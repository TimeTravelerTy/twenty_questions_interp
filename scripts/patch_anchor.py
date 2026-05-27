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
    parse_yes_no,
)
from twenty_q.manifest import RunManifest
from twenty_q.permutations import Permutation
from twenty_q.prompts import question_turn_prompt, self_chosen_prompt

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
    p.add_argument("--anchor", required=True,
                   help="Structural anchor(s) to patch at. Comma-separated for "
                        "simultaneous multi-anchor patching, e.g. "
                        "'end_ready,end_model_q1,end_model_q2,end_model_q3,"
                        "end_model_q4'. All listed anchors are patched in the "
                        "same forward pass at the same --layers band, using the "
                        "same source run. Each value must be one of the 16 v2 "
                        f"anchor labels: {', '.join(V2_ANCHOR_LABELS)}.")
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
    p.add_argument("--answer-rollout", action="store_true",
                   help="If set, after patching at the anchor the model "
                        "regenerates each turn's yes/no answer under the "
                        "patched state (target Q_i text is appended verbatim "
                        "from the manifest; A_i is greedy-decoded by the "
                        "patched model). Reveal is then generated after the "
                        "regenerated turn-4 answer. The patch hook re-fires "
                        "on every per-turn prefill that contains the anchor "
                        "position. Default off: original behavior teacher-"
                        "forces the manifest answers as text and patches only "
                        "the reveal-ready prefill.")
    p.add_argument("--rollout-max-new-tokens", type=int, default=8,
                   help="max_new_tokens for per-turn answer regeneration "
                        "under --answer-rollout. Yes/No is one or two tokens; "
                        "8 leaves headroom for trailing punctuation.")
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


def _build_rollout_context(
    handle: ModelHandle,
    manifest: RunManifest,
    bank: Any,
    prompt_variant: str,
    turn_qs: list[str],
    answers_so_far: list[str],
    include_reveal: bool,
) -> dict[str, torch.Tensor]:
    """Build a chat-input prefix consisting of:
      system+user prompt → Ready → (Q_i, A_i)*k → optionally Q_{k+1} (no A)
      → optionally reveal-user message.

    `turn_qs` is the list of target-manifest question texts visible so far
    (length `len(answers_so_far)` or `len(answers_so_far)+1`). When
    `len(turn_qs) == len(answers_so_far)+1`, the trailing user question is
    open — the next greedy decode under this prefix is the model's
    regenerated answer A_{k+1} under the patched state.

    Position of the `end_ready` anchor in the resulting `input_ids` is
    stable across rollout steps (the prompt+Ready prefix doesn't change),
    so the patched residual at end_ready propagates each step via a fresh
    forward pass.
    """
    if manifest.ready_raw_output is None:
        raise ValueError(f"manifest {manifest.run_id} has no ready_raw_output")
    display_names = {c.id: c.display for c in bank.candidates}
    perm = Permutation(order=tuple(manifest.permutation))
    rendered = self_chosen_prompt(perm, display_names, variant=prompt_variant)
    extra: list[dict[str, str]] = [
        {"role": "assistant", "content": manifest.ready_raw_output.strip()}
    ]
    for i, q in enumerate(turn_qs):
        extra.append({"role": "user", "content": question_turn_prompt(q)})
        if i < len(answers_so_far):
            extra.append({"role": "assistant", "content": answers_so_far[i].strip()})
    if include_reveal:
        extra.append({"role": "user", "content": REVEAL_USER_MESSAGE})
    return _build_chat_input_ids(handle, rendered, extra_turns=extra)


def _register_patch_hooks(
    target_blocks: dict[int, Any],
    layers: list[int],
    positions: dict[str, int],
    src_residuals_per_anchor: dict[str, dict[int, torch.Tensor]],
    anchors_to_patch: list[str],
) -> list[Any]:
    """Install patch hooks for `anchors_to_patch` at the given positions."""
    hooks = []
    for L in layers:
        pos_to_res = {
            positions[a]: src_residuals_per_anchor[a][L]
            for a in anchors_to_patch
        }
        if not pos_to_res:
            continue
        h = target_blocks[L].register_forward_hook(_make_patch_hook(pos_to_res))
        hooks.append(h)
    return hooks


def _rollout_trial(
    handle: ModelHandle,
    manifest: RunManifest,
    bank: Any,
    prompt_variant: str,
    anchors: list[str],
    layers: list[int],
    target_blocks: dict[int, Any],
    src_residuals_per_anchor: dict[str, dict[int, torch.Tensor]] | None,
    rollout_max_new_tokens: int,
    class_first_tok: dict[str, int],
) -> dict[str, Any] | None:
    """Run a single rollout. If `src_residuals_per_anchor` is None, patches
    are disabled (baseline path). Returns None if anchor lookup fails on
    any rollout step."""
    turn_qs = [t.question_text for t in manifest.turns]
    target_answers_bool = [t.answer_bool for t in manifest.turns]
    target_answers_raw = [t.raw_model_output.strip() for t in manifest.turns]

    gen_answers_raw: list[str] = []
    gen_answers_bool: list[bool | None] = []

    for i in range(len(turn_qs)):
        inputs = _build_rollout_context(
            handle, manifest, bank, prompt_variant,
            turn_qs[: i + 1], gen_answers_raw, include_reveal=False,
        )
        found = _find_anchors(handle.tokenizer, inputs["input_ids"])
        if any(k.startswith("__DEBUG_") for k in found):
            return None
        if src_residuals_per_anchor is not None:
            valid = [a for a in anchors if a in found]
            positions = {a: int(found[a]) for a in valid}
            seq_len = inputs["input_ids"].shape[1]
            if positions and max(positions.values()) >= seq_len:
                return None
            hook_handles = _register_patch_hooks(
                target_blocks, layers, positions,
                src_residuals_per_anchor, valid,
            )
        else:
            hook_handles = []
        try:
            raw_ans, _ = _generate_reveal_greedy(
                handle, inputs, max_new_tokens=rollout_max_new_tokens,
            )
        finally:
            for h in hook_handles:
                h.remove()
        gen_answers_raw.append(raw_ans.strip())
        gen_answers_bool.append(parse_yes_no(raw_ans))

    # Reveal generation under the final patched prefix.
    inputs = _build_rollout_context(
        handle, manifest, bank, prompt_variant,
        turn_qs, gen_answers_raw, include_reveal=True,
    )
    found = _find_anchors(handle.tokenizer, inputs["input_ids"])
    if any(k.startswith("__DEBUG_") for k in found):
        return None
    if src_residuals_per_anchor is not None:
        valid = [a for a in anchors if a in found]
        positions = {a: int(found[a]) for a in valid}
        seq_len = inputs["input_ids"].shape[1]
        if positions and max(positions.values()) >= seq_len:
            return None
        hook_handles = _register_patch_hooks(
            target_blocks, layers, positions,
            src_residuals_per_anchor, valid,
        )
    else:
        hook_handles = []
    try:
        raw_reveal, first_logits = _generate_reveal_greedy(handle, inputs)
    finally:
        for h in hook_handles:
            h.remove()

    canon = parse_reveal_to_canonical(raw_reveal, bank)
    class_logits = {cid: float(first_logits[tid])
                    for cid, tid in class_first_tok.items()}
    answer_flips = [
        (gb is not None and tb is not None and gb != tb)
        for gb, tb in zip(gen_answers_bool, target_answers_bool)
    ]
    return {
        "regen_answers_raw": gen_answers_raw,
        "regen_answers_bool": gen_answers_bool,
        "target_answers_raw": target_answers_raw,
        "target_answers_bool": target_answers_bool,
        "answer_flip_mask": answer_flips,
        "n_answer_flips": sum(answer_flips),
        "reveal_raw": raw_reveal.strip(),
        "reveal_canonical": canon,
        "class_logits": class_logits,
    }


def _make_patch_hook(pos_to_residual: dict[int, torch.Tensor]):
    """Forward hook replacing block output at each position in
    `pos_to_residual` with the corresponding source residual during
    prefill. Supports multi-anchor patching: one layer block may be
    patched at several anchor positions in the same forward. Decode
    steps (shape[1]==1) are left untouched — KV cache from the patched
    prefill is what propagates the intervention forward."""
    def hook(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        seq = hs.shape[1]
        new_hs = None
        for position, src_residual in pos_to_residual.items():
            if seq <= position:
                continue
            if new_hs is None:
                new_hs = hs.clone()
            new_hs[:, position, :] = src_residual.to(device=hs.device, dtype=hs.dtype)
        if new_hs is None:
            return output
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
    anchors: list[str],
    layers: list[int],
) -> dict[str, dict[int, torch.Tensor]]:
    """Load source residuals at (anchor, layer) for each anchor in `anchors`
    and each layer in `layers` from the v2 capture.
    Returns {anchor: {layer_idx: tensor of shape (hidden,)}}.
    """
    pt_path = src_residuals_dir / f"{run_id}.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"missing v2 capture file: {pt_path}")
    d = torch.load(pt_path, map_location="cpu", weights_only=False)
    anchor_labels = list(d["anchor_labels"])
    residuals = d["residuals"]  # (K, n_layers+1, hidden)
    if residuals.shape[1] <= max(layers):
        raise ValueError(
            f"layer index out of range: max requested {max(layers)} "
            f"vs residuals shape {tuple(residuals.shape)} for {pt_path}"
        )
    out: dict[str, dict[int, torch.Tensor]] = {}
    for anchor in anchors:
        if anchor not in anchor_labels:
            raise ValueError(
                f"anchor {anchor!r} not in capture file's anchor_labels "
                f"({anchor_labels}) for {pt_path}"
            )
        a_idx = anchor_labels.index(anchor)
        out[anchor] = {L: residuals[a_idx, L].to(torch.float32) for L in layers}
    return out


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
    anchors = [a.strip() for a in args.anchor.split(",") if a.strip()]
    bad = [a for a in anchors if a not in V2_ANCHOR_LABELS]
    if bad:
        print(f"unknown anchor(s) {bad}; must be in {V2_ANCHOR_LABELS}",
              file=sys.stderr)
        return 2
    if not anchors:
        print("no anchors given", file=sys.stderr)
        return 2
    print(f"Patching anchors={anchors} layers={layers}")

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
    # Under --answer-rollout, the baseline also regenerates each turn's
    # answer (no patch) so we can subtract baseline non-determinism from
    # the patched-trial answer-flip counts.
    baseline_records: list[dict[str, Any]] = []
    baseline_class_logits: dict[str, dict[str, float]] = {}
    baseline_rollout: dict[str, dict[str, Any]] = {}
    t0 = time.time()
    for tgt_class, runs in tgt_runs.items():
        for tgt in runs:
            if args.answer_rollout:
                roll = _rollout_trial(
                    handle, tgt, bank, args.prompt_variant,
                    anchors, layers, target_blocks={}, src_residuals_per_anchor=None,
                    rollout_max_new_tokens=args.rollout_max_new_tokens,
                    class_first_tok=class_first_tok,
                )
                if roll is None:
                    print(f"  baseline rollout failed for {tgt.run_id}", file=sys.stderr)
                    continue
                baseline_class_logits[tgt.run_id] = roll["class_logits"]
                baseline_rollout[tgt.run_id] = roll
                baseline_records.append({
                    "tgt_class": tgt_class,
                    "tgt_run": tgt.run_id,
                    "baseline_reveal_raw": roll["reveal_raw"],
                    "baseline_canonical": roll["reveal_canonical"],
                    "original_reveal_canonical": tgt.reveal_canonical_id,
                    "baseline_class_logits": roll["class_logits"],
                    "baseline_regen_answers_raw": roll["regen_answers_raw"],
                    "baseline_regen_answers_bool": roll["regen_answers_bool"],
                    "target_answers_bool": roll["target_answers_bool"],
                    "baseline_n_answer_flips": roll["n_answer_flips"],
                    "baseline_answer_flip_mask": roll["answer_flip_mask"],
                })
            else:
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
    # source-residual computation. Skipped under --answer-rollout, where
    # positions are recomputed per rollout step inside `_rollout_trial`.
    pos_index: dict[str, dict[str, int]] = {}
    if not args.answer_rollout:
        for tgt_class, runs in tgt_runs.items():
            for tgt in runs:
                inputs = _context_with_reveal(handle, tgt, bank, args.prompt_variant)
                found = _find_anchors(handle.tokenizer, inputs["input_ids"])
                if any(k.startswith("__DEBUG_") for k in found):
                    print(f"  failed anchor lookup for tgt {tgt.run_id}: {found}",
                          file=sys.stderr)
                    continue
                missing = [a for a in anchors if a not in found]
                if missing:
                    print(f"  anchor(s) {missing} not found in tgt {tgt.run_id}",
                          file=sys.stderr)
                    continue
                pos_index[tgt.run_id] = {a: int(found[a]) for a in anchors}

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
                src_residuals_per_anchor = _load_src_residuals(
                    src_dir, src.run_id, anchors, layers
                )
            except (FileNotFoundError, ValueError) as e:
                print(f"  skipping src {src.run_id}: {e}", file=sys.stderr)
                continue

            for tgt_class in realized:
                for tgt in tgt_runs[tgt_class]:
                    trial += 1
                    if args.answer_rollout:
                        roll = _rollout_trial(
                            handle, tgt, bank, args.prompt_variant,
                            anchors, layers, target_blocks,
                            src_residuals_per_anchor,
                            rollout_max_new_tokens=args.rollout_max_new_tokens,
                            class_first_tok=class_first_tok,
                        )
                        if roll is None:
                            print(f"  rollout failed for tgt {tgt.run_id}", file=sys.stderr)
                            continue
                        patched_records.append({
                            "src_class": src_class,
                            "src_run": src.run_id,
                            "tgt_class": tgt_class,
                            "tgt_run": tgt.run_id,
                            "patched_reveal_raw": roll["reveal_raw"],
                            "patched_canonical": roll["reveal_canonical"],
                            "patched_class_logits": roll["class_logits"],
                            "regen_answers_raw": roll["regen_answers_raw"],
                            "regen_answers_bool": roll["regen_answers_bool"],
                            "target_answers_bool": roll["target_answers_bool"],
                            "answer_flip_mask": roll["answer_flip_mask"],
                            "n_answer_flips": roll["n_answer_flips"],
                        })
                        if trial % 20 == 0 or trial == total_trials:
                            print(f"  [{trial}/{total_trials}] "
                                  f"src={src_class}/{src.run_id} "
                                  f"tgt={tgt_class}/{tgt.run_id} → "
                                  f"{roll['reveal_canonical']} "
                                  f"(answer-flips={roll['n_answer_flips']})")
                        continue

                    if tgt.run_id not in pos_index:
                        continue
                    positions = pos_index[tgt.run_id]  # {anchor: pos}
                    inputs = _context_with_reveal(handle, tgt, bank,
                                                  args.prompt_variant)
                    seq_len = inputs["input_ids"].shape[1]
                    if max(positions.values()) >= seq_len:
                        print(f"  max anchor pos {max(positions.values())} >= "
                              f"seq len {seq_len} for tgt {tgt.run_id}",
                              file=sys.stderr)
                        continue
                    hook_handles = []
                    for L in layers:
                        pos_to_res = {
                            positions[a]: src_residuals_per_anchor[a][L]
                            for a in anchors
                        }
                        h = target_blocks[L].register_forward_hook(
                            _make_patch_hook(pos_to_res)
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
                        "positions": positions,
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
            cell_summary = {
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
            if args.answer_rollout and n:
                # Per-cell answer-flip stats. Subtract baseline non-determinism
                # (each tgt's no-patch rollout flip count) to isolate the patch
                # contribution.
                patched_flips = [r["n_answer_flips"] for r in cell]
                base_flips = [
                    baseline_rollout[r["tgt_run"]]["n_answer_flips"]
                    if r["tgt_run"] in baseline_rollout else 0
                    for r in cell
                ]
                cell_summary["answer_flips_mean"] = sum(patched_flips) / n
                cell_summary["answer_flips_baseline_mean"] = sum(base_flips) / n
                cell_summary["answer_flips_delta_mean"] = (
                    cell_summary["answer_flips_mean"]
                    - cell_summary["answer_flips_baseline_mean"]
                )
                # Per-turn flip rate (fraction of trials where turn i answer flipped vs target).
                per_turn_patched = [0, 0, 0, 0]
                per_turn_baseline = [0, 0, 0, 0]
                for r in cell:
                    for i, f in enumerate(r["answer_flip_mask"][:4]):
                        if f:
                            per_turn_patched[i] += 1
                    br = baseline_rollout.get(r["tgt_run"])
                    if br is not None:
                        for i, f in enumerate(br["answer_flip_mask"][:4]):
                            if f:
                                per_turn_baseline[i] += 1
                cell_summary["per_turn_flip_rate_patched"] = [x / n for x in per_turn_patched]
                cell_summary["per_turn_flip_rate_baseline"] = [x / n for x in per_turn_baseline]
            summaries[f"{src_class}->{tgt_class}"] = cell_summary

    results = {
        "run_dir": str(run_dir),
        "src_residuals_dir": str(src_dir),
        "model": args.model,
        "torch_dtype": args.dtype,
        "anchors": anchors,
        "layers": layers,
        "prompt_variant": args.prompt_variant,
        "answer_rollout": args.answer_rollout,
        "rollout_max_new_tokens": args.rollout_max_new_tokens,
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

    if args.answer_rollout:
        print()
        print("Answer-flip-delta matrix (patched mean - baseline mean, out of 4 turns):")
        print("Positive = patch shifts the model's regenerated answers vs no-patch rollout.")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for src_class in realized:
            row = [f"  {src_class[:8]:>8} |"]
            for tgt_class in realized:
                s = summaries[f"{src_class}->{tgt_class}"]
                d = s.get("answer_flips_delta_mean")
                row.append(f"{(d or 0):+7.3f} ")
            print("  " + " | ".join(row[0:1] + row[1:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
