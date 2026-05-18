"""M5 — Causal patching at an arbitrary structural anchor (16-anchor v2 schema).

Generalisation of `patch_turn4.py`. Where `patch_turn4` patches only at
`pre_answer_qN` positions and loads source residuals from per-turn
activation files, this script patches at any of the 16 v2 anchors and
loads source residuals from the v2 capture file
(`runs/positional_residuals/<scale>_<variant>_n80_v2/<run_id>.pt`).

Designed for the M5 scale-comparison patch sweep at 27B:

- 27B grants explicit class commitment at `end_ready` (LR LOO 0.51 @ L16,
  3.55x chance) that 12B lacks. This script patches end_ready (or any
  other anchor) to test whether that legibility is load-bearing — i.e.
  whether patching source's `end_ready` residual into a target run flips
  the target's reveal across 4 intervening dialogue turns.
- For each `(src_class, tgt_class)` pair: replace target's residual at
  the chosen anchor + layers with source's stored residual, generate the
  reveal greedily, measure first-step argmax flip rate AND
  `logit[src_class] - logit[tgt_class]` delta vs. an unpatched baseline.

Same metric architecture as `patch_turn4.py`. Different anchor + source-
residual format.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twenty_q.banks import load_bank
from twenty_q.config import MODEL_MAIN
from twenty_q.dialogue import (
    REVEAL_USER_MESSAGE,
    ModelHandle,
    _build_chat_input_ids,
    _history_to_chat_turns,
    load_model,
    parse_reveal_to_canonical,
)
from twenty_q.manifest import RunManifest
from twenty_q.permutations import Permutation
from twenty_q.prompts import self_chosen_prompt

# Reuse the capture script's anchor-finding logic so positions in this
# script match the positions used to compute the source residuals.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_positional_residuals import (  # noqa: E402
    ANCHOR_LABELS_AT_EOT,
    _find_anchors,
)

# All 16 v2 anchor labels in the order they appear in the capture .pt files.
V2_ANCHOR_LABELS = [
    "end_user_prompt",
    "end_ready",
    "end_user_q1",
    "pre_answer_q1",
    "end_model_q1",
    "end_user_q2",
    "pre_answer_q2",
    "end_model_q2",
    "end_user_q3",
    "pre_answer_q3",
    "end_model_q3",
    "end_user_q4",
    "pre_answer_q4",
    "end_model_q4",
    "end_reveal_user",
    "pre_reveal_gen",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True,
                   help="Diagnostic run directory (selfchosen_ready_*).")
    p.add_argument("--src-residuals-dir", required=True,
                   help="v2 capture directory with per-run .pt files. "
                        "Each .pt has keys {anchor_labels, anchor_positions, "
                        "residuals (K, n_layers+1, hidden), seq_len, class, run_id}.")
    p.add_argument("--anchor", required=True, choices=V2_ANCHOR_LABELS,
                   help="Structural anchor to patch at.")
    p.add_argument("--model", default=MODEL_MAIN)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "bfloat16", "float16"])
    p.add_argument("--layers", type=str, required=True,
                   help="Comma-separated 1-indexed residual-block layers, "
                        "e.g. '12,13,14,15,16,17,18,19,20'. All listed layers "
                        "are patched at the anchor position simultaneously. "
                        "Indexed into hidden_states tuple: 0=embedding, "
                        "1=block-0 output, ..., L=block-(L-1) output. The v2 "
                        "capture stores residuals[L] = block-(L-1) output, "
                        "so layer index here matches the capture index used "
                        "by `probe_positional_anchors.py`.")
    p.add_argument("--n-source-per-class", type=int, default=5)
    p.add_argument("--n-target-per-class", type=int, default=5)
    p.add_argument("--prompt-variant", default="default")
    p.add_argument("--out-json", required=True)
    return p.parse_args()


def _load_kept_manifests(run_dir: Path) -> list[RunManifest]:
    attempts = sorted(run_dir.glob("attempt_*"))
    manifests: list[RunManifest] = []
    for a in attempts:
        mp = a / "manifest.json"
        if not mp.exists():
            continue
        m = RunManifest.load(mp)
        if m.reveal_canonical_id is not None:
            manifests.append(m)
    return manifests


def _group_by_class(manifests: list[RunManifest]) -> dict[str, list[RunManifest]]:
    out: dict[str, list[RunManifest]] = {}
    for m in manifests:
        cid = m.reveal_canonical_id
        if cid is None:
            continue
        out.setdefault(cid, []).append(m)
    return out


def _context_with_reveal(
    handle: ModelHandle,
    manifest: RunManifest,
    bank: Any,
    prompt_variant: str,
) -> dict[str, torch.Tensor]:
    """Tokenize the full reveal-ready context: system+user prompt → Ready →
    Q1/A1 → Q2/A2 → Q3/A3 → Q4/A4 → reveal-user message → generation prompt.
    Last token is the pre-reveal-generation position. Mirrors the capture
    script's `_build_full_prefix_inputs` (so anchor positions found via
    `_find_anchors` line up with the v2 capture)."""
    display_names = {c.id: c.display for c in bank.candidates}
    perm = Permutation(order=tuple(manifest.permutation))
    rendered = self_chosen_prompt(perm, display_names, variant=prompt_variant)
    extra = [
        *_history_to_chat_turns(manifest.ready_raw_output, list(manifest.turns)),
        {"role": "user", "content": REVEAL_USER_MESSAGE},
    ]
    return _build_chat_input_ids(handle, rendered, extra_turns=extra)


def _make_patch_hook(position: int, src_residual: torch.Tensor):
    """Forward hook replacing block output at `position` with `src_residual`
    during prefill. Decode steps (shape[1]==1) are left untouched — KV cache
    from the patched prefill is what propagates the intervention forward."""
    def hook(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if hs.shape[1] <= position:
            return output
        new_hs = hs.clone()
        new_hs[:, position, :] = src_residual.to(device=hs.device, dtype=hs.dtype)
        if isinstance(output, tuple):
            return (new_hs,) + tuple(output[1:])
        return new_hs
    return hook


@torch.no_grad()
def _generate_reveal_greedy(
    handle: ModelHandle, model_inputs: dict[str, torch.Tensor], max_new_tokens: int = 48
) -> tuple[str, torch.Tensor]:
    gen = handle.model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=handle.tokenizer.eos_token_id,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
    )
    new_tokens = gen.sequences[0, model_inputs["input_ids"].shape[1]:]
    text = handle.tokenizer.decode(new_tokens, skip_special_tokens=True)
    first_step_logits = gen.scores[0][0].detach().to("cpu", dtype=torch.float32)
    return text, first_step_logits


def _build_class_first_token_ids(
    handle: ModelHandle, realized: list[str], bank: Any
) -> dict[str, int]:
    """Map each realized class id to the first token id of its display name
    when emitted as the start of a reveal answer (with a leading space)."""
    display_by_id = {c.id: c.display for c in bank.candidates}
    out: dict[str, int] = {}
    for cid in realized:
        display = display_by_id[cid]
        token_str = " " + display.capitalize()
        ids = handle.tokenizer.encode(token_str, add_special_tokens=False)
        if not ids:
            raise RuntimeError(f"Empty tokenization for {token_str!r}")
        out[cid] = ids[0]
    return out


def _locate_layer_list(model, max_block_idx: int):
    for path in ("model.layers", "model.language_model.layers",
                 "language_model.model.layers", "language_model.layers"):
        obj = model
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok and hasattr(obj, "__len__") and len(obj) > max_block_idx:
            return obj, f"model.{path}"
    import torch.nn as nn
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and name.endswith(".layers") and len(mod) > max_block_idx:
            return mod, name
    return None, None


def _load_src_residuals(
    src_residuals_dir: Path,
    run_id: str,
    anchor: str,
    layers: list[int],
) -> dict[int, torch.Tensor]:
    """Load source residual at (anchor, layer) for `layers` from the v2 capture.
    Returns {layer_idx: tensor of shape (hidden,)}.
    """
    pt_path = src_residuals_dir / f"{run_id}.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"missing v2 capture file: {pt_path}")
    d = torch.load(pt_path, map_location="cpu", weights_only=False)
    anchor_labels = list(d["anchor_labels"])
    if anchor not in anchor_labels:
        raise ValueError(
            f"anchor {anchor!r} not in capture file's anchor_labels "
            f"({anchor_labels}) for {pt_path}"
        )
    a_idx = anchor_labels.index(anchor)
    residuals = d["residuals"]  # (K, n_layers+1, hidden)
    if residuals.shape[1] <= max(layers):
        raise ValueError(
            f"layer index out of range: max requested {max(layers)} "
            f"vs residuals shape {tuple(residuals.shape)} for {pt_path}"
        )
    return {L: residuals[a_idx, L].to(torch.float32) for L in layers}


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        print(f"run-dir not found: {run_dir}", file=sys.stderr)
        return 2
    src_dir = Path(args.src_residuals_dir).resolve()
    if not src_dir.exists():
        print(f"src-residuals-dir not found: {src_dir}", file=sys.stderr)
        return 2

    bank = load_bank()
    manifests = _load_kept_manifests(run_dir)
    by_class = _group_by_class(manifests)
    realized = sorted(by_class.keys())
    print(f"Realized classes: {realized}")
    for cid in realized:
        print(f"  {cid}: {len(by_class[cid])} runs")

    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    if any(L < 1 for L in layers):
        print("layers must be >= 1 (0 is embeddings, not a residual block)",
              file=sys.stderr)
        return 2
    print(f"Patching anchor={args.anchor} layers={layers}")

    handle = load_model(args.model, device=args.device, dtype=dtype)
    model = handle.model
    layer_list, layer_path = _locate_layer_list(model, max(layers) - 1)
    if layer_list is None:
        print("Could not locate decoder-layer ModuleList; aborting.",
              file=sys.stderr)
        return 3
    print(f"Found decoder layers at {layer_path} (n={len(layer_list)})")
    target_blocks = {L: layer_list[L - 1] for L in layers}

    class_first_tok = _build_class_first_token_ids(handle, realized, bank)
    print(f"Class first-token ids: {class_first_tok}")

    # Source / target pools — distinct sets so no run patches into itself.
    src_runs: dict[str, list[RunManifest]] = {}
    tgt_runs: dict[str, list[RunManifest]] = {}
    for cid in realized:
        runs = by_class[cid]
        src_runs[cid] = runs[: args.n_source_per_class]
        tgt_runs[cid] = runs[-args.n_target_per_class:]
    print(f"Source runs per class: {args.n_source_per_class}")
    print(f"Target runs per class: {args.n_target_per_class}")

    # 1) No-patch baselines: greedy reveal on each target context.
    baseline_records: list[dict[str, Any]] = []
    baseline_class_logits: dict[str, dict[str, float]] = {}
    t0 = time.time()
    for tgt_class, runs in tgt_runs.items():
        for tgt in runs:
            inputs = _context_with_reveal(handle, tgt, bank, args.prompt_variant)
            raw, first_logits = _generate_reveal_greedy(handle, inputs)
            canon = parse_reveal_to_canonical(raw, bank)
            class_logits = {cid: float(first_logits[tid])
                            for cid, tid in class_first_tok.items()}
            baseline_class_logits[tgt.run_id] = class_logits
            baseline_records.append({
                "tgt_class": tgt_class,
                "tgt_run": tgt.run_id,
                "baseline_reveal_raw": raw.strip(),
                "baseline_canonical": canon,
                "original_reveal_canonical": tgt.reveal_canonical_id,
                "baseline_class_logits": class_logits,
            })
    print(f"Baselines: {len(baseline_records)} runs in {time.time()-t0:.1f}s")

    # 2) Per-target anchor position. Tokenize the full reveal-ready context
    # and run `_find_anchors` to locate the chosen anchor's position.
    # Reuses capture-script logic so positions match those used during
    # source-residual computation.
    pos_index: dict[str, int] = {}
    for tgt_class, runs in tgt_runs.items():
        for tgt in runs:
            inputs = _context_with_reveal(handle, tgt, bank, args.prompt_variant)
            anchors = _find_anchors(handle.tokenizer, inputs["input_ids"])
            if any(k.startswith("__DEBUG_") for k in anchors):
                print(f"  failed anchor lookup for tgt {tgt.run_id}: {anchors}",
                      file=sys.stderr)
                continue
            if args.anchor not in anchors:
                print(f"  anchor {args.anchor!r} not found in tgt {tgt.run_id}",
                      file=sys.stderr)
                continue
            pos_index[tgt.run_id] = int(anchors[args.anchor])

    # 3) Patched trials.
    patched_records: list[dict[str, Any]] = []
    t0 = time.time()
    total_trials = sum(
        len(src_runs[sc]) * len(tgt_runs[tc]) for sc in realized for tc in realized
    )
    trial = 0
    for src_class in realized:
        for src in src_runs[src_class]:
            try:
                src_residuals_per_layer = _load_src_residuals(
                    src_dir, src.run_id, args.anchor, layers
                )
            except (FileNotFoundError, ValueError) as e:
                print(f"  skipping src {src.run_id}: {e}", file=sys.stderr)
                continue

            for tgt_class in realized:
                for tgt in tgt_runs[tgt_class]:
                    trial += 1
                    if tgt.run_id not in pos_index:
                        continue
                    pos = pos_index[tgt.run_id]
                    inputs = _context_with_reveal(handle, tgt, bank,
                                                  args.prompt_variant)
                    if pos >= inputs["input_ids"].shape[1]:
                        print(f"  pos {pos} >= seq len "
                              f"{inputs['input_ids'].shape[1]} for tgt "
                              f"{tgt.run_id}", file=sys.stderr)
                        continue
                    hook_handles = []
                    for L in layers:
                        h = target_blocks[L].register_forward_hook(
                            _make_patch_hook(pos, src_residuals_per_layer[L])
                        )
                        hook_handles.append(h)
                    try:
                        raw, first_logits = _generate_reveal_greedy(handle, inputs)
                    finally:
                        for h in hook_handles:
                            h.remove()
                    canon = parse_reveal_to_canonical(raw, bank)
                    class_logits = {cid: float(first_logits[tid])
                                    for cid, tid in class_first_tok.items()}
                    patched_records.append({
                        "src_class": src_class,
                        "src_run": src.run_id,
                        "tgt_class": tgt_class,
                        "tgt_run": tgt.run_id,
                        "pos": pos,
                        "patched_reveal_raw": raw.strip(),
                        "patched_canonical": canon,
                        "patched_class_logits": class_logits,
                    })
                    if trial % 20 == 0 or trial == total_trials:
                        print(f"  [{trial}/{total_trials}] "
                              f"src={src_class}/{src.run_id} "
                              f"tgt={tgt_class}/{tgt.run_id} → {canon}")
    print(f"Patched trials: {len(patched_records)} in {time.time()-t0:.1f}s")

    # 4) Per-(src, tgt) summaries.
    summaries: dict[str, dict[str, Any]] = {}
    for src_class in realized:
        for tgt_class in realized:
            cell = [r for r in patched_records
                    if r["src_class"] == src_class and r["tgt_class"] == tgt_class]
            n = len(cell)
            flip_to_src = sum(1 for r in cell
                              if r["patched_canonical"] == src_class) / n if n else None
            kept_tgt = sum(1 for r in cell
                           if r["patched_canonical"] == tgt_class) / n if n else None
            unparsed = sum(1 for r in cell if r["patched_canonical"] is None)
            dist: dict[str, int] = {}
            for r in cell:
                c = r["patched_canonical"] or "__unparsed__"
                dist[c] = dist.get(c, 0) + 1
            patch_diffs = [
                r["patched_class_logits"][src_class] - r["patched_class_logits"][tgt_class]
                for r in cell
            ]
            base_diffs = [
                baseline_class_logits[r["tgt_run"]][src_class]
                - baseline_class_logits[r["tgt_run"]][tgt_class]
                for r in cell
            ]
            mean_patch = sum(patch_diffs) / n if n else None
            mean_base = sum(base_diffs) / n if n else None
            summaries[f"{src_class}->{tgt_class}"] = {
                "n": n,
                "flip_to_src": flip_to_src,
                "kept_tgt": kept_tgt,
                "flip_to_other": (
                    1 - flip_to_src - kept_tgt
                    if flip_to_src is not None and kept_tgt is not None
                    else None
                ),
                "unparsed": unparsed,
                "distribution": dist,
                "logit_diff_patched": mean_patch,
                "logit_diff_baseline": mean_base,
                "logit_diff_delta": (
                    mean_patch - mean_base
                    if mean_patch is not None and mean_base is not None
                    else None
                ),
            }

    results = {
        "run_dir": str(run_dir),
        "src_residuals_dir": str(src_dir),
        "model": args.model,
        "torch_dtype": args.dtype,
        "anchor": args.anchor,
        "layers": layers,
        "prompt_variant": args.prompt_variant,
        "n_source_per_class": args.n_source_per_class,
        "n_target_per_class": args.n_target_per_class,
        "realized_classes": realized,
        "class_first_token_ids": class_first_tok,
        "baselines": baseline_records,
        "patched_trials": patched_records,
        "summaries": summaries,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}")

    # Console summary matrices.
    print()
    print(f"Flip-to-source rate matrix (row=src, col=tgt):")
    header = "  src\\tgt | " + " | ".join(f"{c[:8]:>8}" for c in realized)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for src_class in realized:
        row = [f"  {src_class[:8]:>8} |"]
        for tgt_class in realized:
            s = summaries[f"{src_class}->{tgt_class}"]
            v = s["flip_to_src"]
            row.append(f"{(v or 0)*100:7.1f}%")
        print("  " + " | ".join(row[0:1] + row[1:]))

    print()
    print("Logit-diff delta matrix (patched - baseline) for logit[src] - logit[tgt]:")
    print("Positive = patch pushes reveal toward src. Diagonals should be ~0 (self-patch).")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for src_class in realized:
        row = [f"  {src_class[:8]:>8} |"]
        for tgt_class in realized:
            s = summaries[f"{src_class}->{tgt_class}"]
            d = s["logit_diff_delta"]
            row.append(f"{(d or 0):+7.2f} ")
        print("  " + " | ".join(row[0:1] + row[1:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
