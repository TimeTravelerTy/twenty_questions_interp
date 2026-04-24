"""M4 — Causal patching at turn-4 pre-answer residual.

For each (source-class, target-class) pair, replace the target run's
residual-stream output at `layer L`, `turn-4 pre-answer token position`
with the source run's saved L-layer residual (from
`turn_04_activations.pt`), then continue the forward pass through the
end-of-game reveal prompt and parse the canonical reveal.

Outputs:
- Per-trial records: (src_class, tgt_class, src_run, tgt_run, patched_reveal, patched_canonical)
- Per-(src, tgt) summary: flip-to-src rate, flip-to-other rate, kept-tgt rate, unparsed count.
- Per-target baseline (no hook) reveal for sanity.

Scope phase 1: single-layer single-position patch, greedy generation,
default self-chosen 4-question panel (D-31 locked probe position).
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, help="Diagnostic run directory.")
    p.add_argument("--model", default=MODEL_MAIN)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--layer", type=int, default=29,
                   help="Residual-stream layer index to patch (index into "
                        "hidden_states tuple; 0=embedding, 1=block 0 out, ...).")
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


def _build_chat_for_run(
    handle: ModelHandle,
    manifest: RunManifest,
    bank: Any,
    prompt_variant: str,
    extra_turns: list[dict[str, str]] | None = None,
) -> dict[str, torch.Tensor]:
    """Rebuild the chat input_ids for a saved run, optionally adding extra turns."""
    display_names = {c.id: c.display for c in bank.candidates}
    perm = Permutation(order=tuple(manifest.permutation))
    rendered = self_chosen_prompt(perm, display_names, variant=prompt_variant)
    return _build_chat_input_ids(handle, rendered, extra_turns=extra_turns)


def _context_up_to_q4_preanswer(
    handle: ModelHandle, manifest: RunManifest, bank: Any, prompt_variant: str
) -> dict[str, torch.Tensor]:
    """Tokenize context ending at the turn-4 pre-answer position (with
    add_generation_prompt=True), so the last input token is the one at which
    the model would next emit turn 4's answer.
    """
    turns_so_far = list(manifest.turns[:3])  # turns 1..3 (answers)
    turn4_question = manifest.turns[3]  # the user message for turn 4
    extra = _history_to_chat_turns(manifest.ready_raw_output, turns_so_far)
    from twenty_q.prompts import question_turn_prompt  # local import to avoid circulars
    extra.append({"role": "user", "content": question_turn_prompt(turn4_question.question_text)})
    return _build_chat_for_run(handle, manifest, bank, prompt_variant, extra_turns=extra)


def _context_with_turn4_and_reveal(
    handle: ModelHandle, manifest: RunManifest, bank: Any, prompt_variant: str
) -> dict[str, torch.Tensor]:
    """Tokenize context including turn-4 answer + reveal user message, with
    add_generation_prompt=True at the end (ready to generate the reveal).
    """
    all_turns = list(manifest.turns)  # all 4 turns
    extra = _history_to_chat_turns(manifest.ready_raw_output, all_turns)
    extra.append({"role": "user", "content": REVEAL_USER_MESSAGE})
    return _build_chat_for_run(handle, manifest, bank, prompt_variant, extra_turns=extra)


def _make_patch_hook(position: int, src_residual: torch.Tensor):
    """Forward hook: replace block output at position `position` (last dim is hidden)
    with the source residual. src_residual shape: (hidden_size,).
    """
    def hook(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        new_hs = hs.clone()
        new_hs[:, position, :] = src_residual.to(device=hs.device, dtype=hs.dtype)
        if isinstance(output, tuple):
            return (new_hs,) + tuple(output[1:])
        return new_hs
    return hook


@torch.no_grad()
def _generate_reveal_greedy(
    handle: ModelHandle, model_inputs: dict[str, torch.Tensor], max_new_tokens: int = 48
) -> str:
    gen = handle.model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=handle.tokenizer.eos_token_id,
        do_sample=False,
    )
    new_tokens = gen[0, model_inputs["input_ids"].shape[1]:]
    return handle.tokenizer.decode(new_tokens, skip_special_tokens=True)


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        print(f"run-dir not found: {run_dir}", file=sys.stderr)
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
    handle = load_model(args.model, device=args.device, dtype=dtype)
    model = handle.model
    # For Gemma3, hidden_states index L corresponds to the output of layer L-1.
    # The exact attribute path depends on model variant; try common layouts.
    layer_block_idx = args.layer - 1
    if layer_block_idx < 0:
        print("--layer must be >= 1 (0 is embeddings, not a residual block)", file=sys.stderr)
        return 2
    layer_list = None
    for path in ("model.layers", "model.language_model.layers",
                 "language_model.model.layers", "language_model.layers"):
        obj = model
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok and hasattr(obj, "__len__") and len(obj) > layer_block_idx:
            layer_list = obj
            print(f"Found decoder layers at model.{path} (n={len(obj)})")
            break
    if layer_list is None:
        # Last-resort scan: find the first ModuleList whose name ends in `.layers`.
        import torch.nn as nn
        for name, mod in model.named_modules():
            if isinstance(mod, nn.ModuleList) and name.endswith(".layers") and len(mod) > layer_block_idx:
                layer_list = mod
                print(f"Found decoder layers at {name} (n={len(mod)})")
                break
    if layer_list is None:
        print("Could not locate decoder-layer ModuleList; aborting.", file=sys.stderr)
        return 3
    target_block = layer_list[layer_block_idx]

    # Pick n runs per class for source and target (distinct sets so we don't patch a run into itself).
    src_runs: dict[str, list[RunManifest]] = {}
    tgt_runs: dict[str, list[RunManifest]] = {}
    for cid in realized:
        runs = by_class[cid]
        src_runs[cid] = runs[: args.n_source_per_class]
        # Take last-k as targets so src/tgt pools don't overlap when both N are small.
        tgt_runs[cid] = runs[-args.n_target_per_class:]

    print(f"Source runs per class: {args.n_source_per_class}")
    print(f"Target runs per class: {args.n_target_per_class}")

    # ---- 1) No-patch baseline for each target: greedy reveal on the target context. ----
    baseline_records: list[dict[str, Any]] = []
    t0 = time.time()
    for tgt_class, runs in tgt_runs.items():
        for tgt in runs:
            inputs = _context_with_turn4_and_reveal(handle, tgt, bank, args.prompt_variant)
            raw = _generate_reveal_greedy(handle, inputs)
            canon = parse_reveal_to_canonical(raw, bank)
            baseline_records.append({
                "tgt_class": tgt_class,
                "tgt_run": tgt.run_id,
                "baseline_reveal_raw": raw.strip(),
                "baseline_canonical": canon,
                "original_reveal_canonical": tgt.reveal_canonical_id,
            })
    print(f"Baselines: {len(baseline_records)} runs in {time.time()-t0:.1f}s")

    # ---- 2) Compute pre-answer position index for every tgt run. ----
    pos_index: dict[str, int] = {}
    for tgt_class, runs in tgt_runs.items():
        for tgt in runs:
            inputs_pre = _context_up_to_q4_preanswer(handle, tgt, bank, args.prompt_variant)
            pos_index[tgt.run_id] = int(inputs_pre["input_ids"].shape[1] - 1)

    # ---- 3) Patched trials. ----
    patched_records: list[dict[str, Any]] = []
    t0 = time.time()
    total_trials = sum(
        len(src_runs[sc]) * len(tgt_runs[tc]) for sc in realized for tc in realized
    )
    trial = 0
    for src_class in realized:
        for src in src_runs[src_class]:
            src_acts_path = run_dir / src.run_id / "turn_04_activations.pt"
            if not src_acts_path.exists():
                print(f"  missing {src_acts_path}, skipping", file=sys.stderr)
                continue
            src_acts = torch.load(src_acts_path, map_location="cpu")  # (n_layers+1, hidden)
            src_residual_L = src_acts[args.layer].to(torch.float32)

            for tgt_class in realized:
                for tgt in tgt_runs[tgt_class]:
                    trial += 1
                    inputs = _context_with_turn4_and_reveal(handle, tgt, bank, args.prompt_variant)
                    pos = pos_index[tgt.run_id]
                    if pos >= inputs["input_ids"].shape[1]:
                        print(f"  pos {pos} >= seq len {inputs['input_ids'].shape[1]} for tgt {tgt.run_id}", file=sys.stderr)
                        continue
                    hook_h = target_block.register_forward_hook(
                        _make_patch_hook(pos, src_residual_L)
                    )
                    try:
                        raw = _generate_reveal_greedy(handle, inputs)
                    finally:
                        hook_h.remove()
                    canon = parse_reveal_to_canonical(raw, bank)
                    patched_records.append({
                        "src_class": src_class,
                        "src_run": src.run_id,
                        "tgt_class": tgt_class,
                        "tgt_run": tgt.run_id,
                        "pos": pos,
                        "patched_reveal_raw": raw.strip(),
                        "patched_canonical": canon,
                    })
                    if trial % 20 == 0 or trial == total_trials:
                        print(f"  [{trial}/{total_trials}] src={src_class}/{src.run_id} tgt={tgt_class}/{tgt.run_id} → {canon}")
    print(f"Patched trials: {len(patched_records)} in {time.time()-t0:.1f}s")

    # ---- 4) Per-(src, tgt) summaries. ----
    summaries: dict[str, dict[str, Any]] = {}
    for src_class in realized:
        for tgt_class in realized:
            cell = [r for r in patched_records if r["src_class"] == src_class and r["tgt_class"] == tgt_class]
            n = len(cell)
            flip_to_src = sum(1 for r in cell if r["patched_canonical"] == src_class) / n if n else None
            kept_tgt = sum(1 for r in cell if r["patched_canonical"] == tgt_class) / n if n else None
            unparsed = sum(1 for r in cell if r["patched_canonical"] is None)
            dist: dict[str, int] = {}
            for r in cell:
                c = r["patched_canonical"] or "__unparsed__"
                dist[c] = dist.get(c, 0) + 1
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
            }

    results = {
        "run_dir": str(run_dir),
        "model": args.model,
        "torch_dtype": args.dtype,
        "layer": args.layer,
        "prompt_variant": args.prompt_variant,
        "n_source_per_class": args.n_source_per_class,
        "n_target_per_class": args.n_target_per_class,
        "realized_classes": realized,
        "baselines": baseline_records,
        "patched_trials": patched_records,
        "summaries": summaries,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}")

    # Console summary matrix
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
