"""M4 — Causal patching at turn-4 pre-answer residual.

For each (source-class, target-class) pair, replace the target run's
residual-stream output at one or more `layers`, at the `turn-4
pre-answer token position`, with the source run's saved layer residuals
(from `turn_04_activations.pt`), then continue the forward pass through
the end-of-game reveal prompt and read both the parsed reveal token and
the first-step reveal logits.

Two metrics:
1. Categorical flip rate: argmax of the first generated token via
   `parse_reveal_to_canonical`. Discrete; coarse on a 4-class panel.
2. Logit-difference (Heimersheim & Nanda 2024 best practice): at the
   first reveal-generation step, compute
   `logits[src_first_tok] - logits[tgt_first_tok]` per trial. Continuous;
   sensitive to partial effects that don't flip argmax.

Phase 1 used `--layers 29` (single layer single position). Phase 2a
broadens along the residual stream by accepting `--layers L1,L2,...` to
patch a band simultaneously. Position is still single (pre-answer).
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
    p.add_argument("--layer", type=int, default=None,
                   help="Single residual-stream layer index to patch. "
                        "Either --layer or --layers must be provided. "
                        "Index into hidden_states tuple: 0=embedding, "
                        "1=block 0 out, ...")
    p.add_argument("--layers", type=str, default=None,
                   help="Comma-separated layer band, e.g. '27,28,...,48' or "
                        "'29,30,31'. Overrides --layer. All listed layers are "
                        "patched at the pre-answer position simultaneously.")
    p.add_argument("--turn", type=int, default=4, choices=[1, 2, 3, 4],
                   help="Which turn's pre-answer position to patch (1..4). "
                        "Default 4. Loads the corresponding "
                        "`turn_0N_activations.pt` from each src run.")
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


def _context_up_to_qN_preanswer(
    handle: ModelHandle, manifest: RunManifest, bank: Any, prompt_variant: str, n: int
) -> dict[str, torch.Tensor]:
    """Tokenize context ending at the turn-N pre-answer position (with
    add_generation_prompt=True), so the last input token is the one at which
    the model would next emit turn N's answer. n is 1-indexed, in [1, 4].
    """
    if n < 1 or n > 4:
        raise ValueError(f"turn N must be 1..4, got {n}")
    turns_so_far = list(manifest.turns[: n - 1])  # turns 1..N-1 with their answers
    turnN_question = manifest.turns[n - 1]  # the user message for turn N
    extra = _history_to_chat_turns(manifest.ready_raw_output, turns_so_far)
    from twenty_q.prompts import question_turn_prompt  # local import to avoid circulars
    extra.append({"role": "user", "content": question_turn_prompt(turnN_question.question_text)})
    return _build_chat_for_run(handle, manifest, bank, prompt_variant, extra_turns=extra)


def _context_up_to_q4_preanswer(
    handle: ModelHandle, manifest: RunManifest, bank: Any, prompt_variant: str
) -> dict[str, torch.Tensor]:
    """Backward-compat alias for _context_up_to_qN_preanswer(..., n=4)."""
    return _context_up_to_qN_preanswer(handle, manifest, bank, prompt_variant, 4)


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
    """Forward hook: replace block output at `position` with the source residual.

    Only patches during prefill (when `hs.shape[1]` covers the full context and
    includes `position`). During autoregressive decode steps `seq_len == 1` and
    we leave the output untouched — the KV cache from the patched prefill is
    what propagates the intervention forward.
    """
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
    """Greedy decode and also return first-step logits over vocab.

    Returns (decoded_text, first_step_logits) where first_step_logits is a
    1-D tensor on CPU with shape (vocab_size,).
    """
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
    when emitted as the start of a reveal answer (with a leading space).

    Reveal prompt asks for "only the name of that animal", so under SentencePiece
    Gemma tokenizers the first generated token is typically " <Name>" or " <name>".
    We tokenize ' Cow', ' Dog', ' Elephant', ' Horse' (capitalized variant) since
    those are the most common reveal openings in observed runs.
    """
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
    if args.layers is not None:
        layers = [int(x) for x in args.layers.split(",") if x.strip()]
    elif args.layer is not None:
        layers = [args.layer]
    else:
        print("must provide --layer or --layers", file=sys.stderr)
        return 2
    if any(L < 1 for L in layers):
        print("layers must be >= 1 (0 is embeddings, not a residual block)", file=sys.stderr)
        return 2
    print(f"Patching layers: {layers}")

    handle = load_model(args.model, device=args.device, dtype=dtype)
    model = handle.model
    # For Gemma3, hidden_states index L corresponds to the output of layer L-1.
    # The exact attribute path depends on model variant; try common layouts.
    max_block_idx = max(layers) - 1
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
        if ok and hasattr(obj, "__len__") and len(obj) > max_block_idx:
            layer_list = obj
            print(f"Found decoder layers at model.{path} (n={len(obj)})")
            break
    if layer_list is None:
        import torch.nn as nn
        for name, mod in model.named_modules():
            if isinstance(mod, nn.ModuleList) and name.endswith(".layers") and len(mod) > max_block_idx:
                layer_list = mod
                print(f"Found decoder layers at {name} (n={len(mod)})")
                break
    if layer_list is None:
        print("Could not locate decoder-layer ModuleList; aborting.", file=sys.stderr)
        return 3
    target_blocks = {L: layer_list[L - 1] for L in layers}

    class_first_tok = _build_class_first_token_ids(handle, realized, bank)
    print(f"Class first-token ids: {class_first_tok}")

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
    baseline_class_logits: dict[str, dict[str, float]] = {}  # tgt_run -> {class_id: logit}
    t0 = time.time()
    for tgt_class, runs in tgt_runs.items():
        for tgt in runs:
            inputs = _context_with_turn4_and_reveal(handle, tgt, bank, args.prompt_variant)
            raw, first_logits = _generate_reveal_greedy(handle, inputs)
            canon = parse_reveal_to_canonical(raw, bank)
            class_logits = {cid: float(first_logits[tid]) for cid, tid in class_first_tok.items()}
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

    # ---- 2) Compute pre-answer position index for every tgt run at the chosen turn. ----
    pos_index: dict[str, int] = {}
    for tgt_class, runs in tgt_runs.items():
        for tgt in runs:
            inputs_pre = _context_up_to_qN_preanswer(handle, tgt, bank, args.prompt_variant, args.turn)
            pos_index[tgt.run_id] = int(inputs_pre["input_ids"].shape[1] - 1)

    # ---- 3) Patched trials. ----
    patched_records: list[dict[str, Any]] = []
    t0 = time.time()
    total_trials = sum(
        len(src_runs[sc]) * len(tgt_runs[tc]) for sc in realized for tc in realized
    )
    trial = 0
    src_acts_filename = f"turn_{args.turn:02d}_activations.pt"
    print(f"Loading source activations from {src_acts_filename} (turn {args.turn} pre-answer)")
    for src_class in realized:
        for src in src_runs[src_class]:
            src_acts_path = run_dir / src.run_id / src_acts_filename
            if not src_acts_path.exists():
                print(f"  missing {src_acts_path}, skipping", file=sys.stderr)
                continue
            src_acts = torch.load(src_acts_path, map_location="cpu")  # (n_layers+1, hidden)
            src_residuals_per_layer = {L: src_acts[L].to(torch.float32) for L in layers}

            for tgt_class in realized:
                for tgt in tgt_runs[tgt_class]:
                    trial += 1
                    inputs = _context_with_turn4_and_reveal(handle, tgt, bank, args.prompt_variant)
                    pos = pos_index[tgt.run_id]
                    if pos >= inputs["input_ids"].shape[1]:
                        print(f"  pos {pos} >= seq len {inputs['input_ids'].shape[1]} for tgt {tgt.run_id}", file=sys.stderr)
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
                    class_logits = {cid: float(first_logits[tid]) for cid, tid in class_first_tok.items()}
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
            # Logit-diff stats: mean over trials of logit[src_class] - logit[tgt_class].
            # Reference baseline: same difference computed without any patch on the
            # tgt_run only — the natural pre-patch separation between src and tgt logits.
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
        "model": args.model,
        "torch_dtype": args.dtype,
        "layers": layers,
        "turn": args.turn,
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

    # Console summary matrices
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
