"""M5b — Probe-direction steering (Contrastive Activation Addition).

Tests whether the linearly-decodable class direction at a chosen
anchor/layer is causally load-bearing for the reveal. Unlike
`patch_anchor.py` which replaces a single token's residual with a
foreign run's residual, this script:

  1. Derives a class-discriminating direction Δ = μ[src] - μ[tgt]
     from the v2 capture's per-run residuals at (anchor, layer),
     averaged within each class.
  2. Injects α·Δ into the residual at the chosen `--steer-layer` at
     every position (or a scoped subset) during the entire forward
     pass — prefill AND decode.

Stronger than position-patching because the direction is added at
every position the model could possibly attend to, not just one
token's K/V. This is the standard CAA recipe (Panickssery et al.
2023, "Steering Llama 2 via Contrastive Activation Addition"), and
the cleanest "is this direction load-bearing?" test available
without retraining.

Caveat (Steered LLM Activations are Non-Surjective):
steering can push the residual stream into states with no prompt
preimage. A positive result means "this direction is sufficient
when injected" — not necessarily "this is the mechanism the model
normally uses." A null result is more decisive: if even injecting
α·Δ at the probe-peak layer fails to move the reveal, the direction
is unambiguously a readout, not a substrate.

Two modes:
  - add: residual ← residual + α·Δ  (push toward source)
  - ablate: residual ← residual − ⟨residual,d̂⟩·d̂  (remove the
            class-discriminating subspace; tests necessity)
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_positional_residuals import _find_anchors  # noqa: E402
from patch_anchor import (  # noqa: E402
    V2_ANCHOR_LABELS,
    _build_class_first_token_ids,
    _build_rollout_context,
    _context_with_reveal,
    _find_anchors_relaxed,
    _generate_reveal_greedy,
    _group_by_class,
    _load_kept_manifests,
    _locate_layer_list,
)
from twenty_q.dialogue import parse_yes_no  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--src-residuals-dir", required=True,
                   help="v2 capture dir; class means are computed across "
                        "all runs whose `class` matches each candidate.")
    p.add_argument("--direction-anchor", required=True,
                   help=f"Anchor to derive direction at; one of {V2_ANCHOR_LABELS}.")
    p.add_argument("--direction-layer", type=int, required=True,
                   help="Layer index (1=first block output, matches "
                        "patch_anchor's --layers convention) to read class "
                        "means from when computing Δ.")
    p.add_argument("--steer-layer", type=int, default=None,
                   help="Layer to inject at (default: same as --direction-layer).")
    p.add_argument("--steer-scope", default="all",
                   choices=["all", "from_anchor"],
                   help="`all`: inject at every position during every forward "
                        "(prefill + decode). `from_anchor`: only at positions "
                        ">= the direction anchor's position in the reveal-ready "
                        "context (still applies on decode steps).")
    p.add_argument("--mode", default="add", choices=["add", "ablate"])
    p.add_argument("--alpha", type=str, default="1.0",
                   help="Comma-separated list of α coefficients to sweep "
                        "for mode=add. Ignored for mode=ablate. e.g. '1,2,5,10'.")
    p.add_argument("--n-target-per-class", type=int, default=5)
    p.add_argument("--model", default=MODEL_MAIN)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "bfloat16", "float16"])
    p.add_argument("--prompt-variant", default="default")
    p.add_argument("--out-json", required=True)
    p.add_argument("--answer-rollout", action="store_true",
                   help="If set, regenerate each turn's yes/no answer under "
                        "the steering hook (same pattern as patch_anchor.py "
                        "--answer-rollout). The steering hook re-fires on "
                        "every per-turn prefill, biasing A_i decoding under "
                        "the steered residual. Reveal is generated after A_4. "
                        "Tracks answer flips vs the manifest target answers.")
    p.add_argument("--rollout-max-new-tokens", type=int, default=8)
    return p.parse_args()


def _compute_class_means(
    src_dir: Path, anchor: str, layer: int, realized: list[str]
) -> dict[str, torch.Tensor]:
    """Mean residual at (anchor, layer) for each realized class, averaged
    over every v2 capture file whose stored `class` matches the class id.
    Shape per class: (hidden,)."""
    by_class: dict[str, list[torch.Tensor]] = {c: [] for c in realized}
    for pt_path in sorted(src_dir.glob("*.pt")):
        d = torch.load(pt_path, map_location="cpu", weights_only=False)
        cid = d.get("class")
        if cid not in by_class:
            continue
        anchor_labels = list(d["anchor_labels"])
        if anchor not in anchor_labels:
            continue
        a_idx = anchor_labels.index(anchor)
        residuals = d["residuals"]  # (K, n_layers+1, hidden)
        if residuals.shape[1] <= layer:
            raise ValueError(
                f"layer {layer} out of range for {pt_path} "
                f"(shape {tuple(residuals.shape)})"
            )
        by_class[cid].append(residuals[a_idx, layer].to(torch.float32))
    means: dict[str, torch.Tensor] = {}
    for c, xs in by_class.items():
        if not xs:
            raise RuntimeError(f"no v2 captures found for class {c!r}")
        means[c] = torch.stack(xs, dim=0).mean(dim=0)
    return means


def _make_steer_hook(
    direction: torch.Tensor,
    alpha: float,
    mode: str,
    start_pos: int | None,
):
    """Forward hook that adds α·Δ (mode=add) or projects out direction
    (mode=ablate) on the layer's residual output. Applies on prefill
    (shape[1]>1, slice positions >= start_pos if given) and on decode
    (shape[1]==1; always applies — the decoded token is at a position
    well past any reasonable start_pos)."""
    d = direction
    d_unit = d / d.norm().clamp_min(1e-6)

    def hook(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        seq = hs.shape[1]
        dev_dir = d.to(device=hs.device, dtype=hs.dtype)
        dev_unit = d_unit.to(device=hs.device, dtype=hs.dtype)
        if seq > 1:
            # Prefill: apply over positions [start_pos:] (or all if None).
            sp = 0 if start_pos is None else min(int(start_pos), seq)
            if mode == "add":
                hs[:, sp:, :] = hs[:, sp:, :] + alpha * dev_dir
            else:  # ablate
                proj = (hs[:, sp:, :] * dev_unit).sum(dim=-1, keepdim=True)
                hs[:, sp:, :] = hs[:, sp:, :] - proj * dev_unit
        else:
            # Decode step: always apply.
            if mode == "add":
                hs[:, :, :] = hs[:, :, :] + alpha * dev_dir
            else:
                proj = (hs * dev_unit).sum(dim=-1, keepdim=True)
                hs[:, :, :] = hs - proj * dev_unit
        if isinstance(output, tuple):
            return (hs,) + tuple(output[1:])
        return hs
    return hook


def _rollout_anchor_start_pos(
    handle: ModelHandle, input_ids: torch.Tensor, direction_anchor: str
) -> int | None:
    """Return the anchor position in a partial rollout context, if present.

    Missing means "do not steer this forward pass yet" for `from_anchor`
    rollout. This matters for late anchors such as `pre_answer_q4`: before
    Q4 exists, steering all positions would silently turn the run into an
    `all`-scope intervention. `pre_reveal_gen` is only present once the
    reveal user turn exists; otherwise every partial prefix has a misleading
    "last token" position.
    """
    if direction_anchor == "pre_reveal_gen":
        reveal_ready = _find_anchors_relaxed(
            handle.tokenizer, input_ids, ["end_reveal_user"]
        )
        if not reveal_ready or "end_reveal_user" not in reveal_ready:
            return None
    found = _find_anchors_relaxed(
        handle.tokenizer, input_ids, [direction_anchor]
    )
    if not found or direction_anchor not in found:
        return None
    return int(found[direction_anchor])


def _steer_rollout_trial(
    handle: ModelHandle,
    manifest: RunManifest,
    bank: Any,
    prompt_variant: str,
    steer_block: Any,
    direction: torch.Tensor,
    alpha: float,
    mode: str,
    direction_anchor: str,
    steer_scope: str,
    rollout_max_new_tokens: int,
    class_first_tok: dict[str, int],
) -> dict[str, Any] | None:
    """Same per-turn loop as patch_anchor._rollout_trial but the
    intervention is a steering hook on `steer_block` rather than a
    position-patch. The hook fires on every per-turn prefill (and on
    decode), so the steering bias enters A_i decoding at each step
    and propagates via the regenerated answer text."""
    turn_qs = [t.question_text for t in manifest.turns]
    target_answers_bool = [t.answer_bool for t in manifest.turns]
    target_answers_raw = [t.raw_model_output.strip() for t in manifest.turns]
    gen_answers_raw: list[str] = []
    gen_answers_bool: list[bool | None] = []

    def _apply_and_decode(inputs, max_new):
        # start_pos for scope=from_anchor: re-locate the anchor in this prefix.
        start_pos = None
        if steer_scope == "from_anchor":
            start_pos = _rollout_anchor_start_pos(
                handle, inputs["input_ids"], direction_anchor
            )
            if start_pos is None:
                return _generate_reveal_greedy(handle, inputs, max_new_tokens=max_new)
        h = steer_block.register_forward_hook(
            _make_steer_hook(direction, alpha, mode, start_pos)
        )
        try:
            return _generate_reveal_greedy(handle, inputs, max_new_tokens=max_new)
        finally:
            h.remove()

    for i in range(len(turn_qs)):
        inputs = _build_rollout_context(
            handle, manifest, bank, prompt_variant,
            turn_qs[: i + 1], gen_answers_raw, include_reveal=False,
        )
        raw_ans, _ = _apply_and_decode(inputs, rollout_max_new_tokens)
        gen_answers_raw.append(raw_ans.strip())
        gen_answers_bool.append(parse_yes_no(raw_ans))

    inputs = _build_rollout_context(
        handle, manifest, bank, prompt_variant,
        turn_qs, gen_answers_raw, include_reveal=True,
    )
    raw_reveal, first_logits = _apply_and_decode(inputs, 48)
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


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    src_dir = Path(args.src_residuals_dir).resolve()
    if not run_dir.exists() or not src_dir.exists():
        print("missing run-dir or src-residuals-dir", file=sys.stderr)
        return 2
    if args.direction_anchor not in V2_ANCHOR_LABELS:
        print(f"unknown anchor {args.direction_anchor!r}", file=sys.stderr)
        return 2
    steer_layer = args.steer_layer if args.steer_layer is not None else args.direction_layer

    bank = load_bank()
    manifests = _load_kept_manifests(run_dir)
    by_class = _group_by_class(manifests)
    realized = sorted(by_class.keys())
    print(f"Realized classes: {realized}")

    print(f"Computing class-mean residuals at anchor={args.direction_anchor} "
          f"layer={args.direction_layer} ...")
    means = _compute_class_means(src_dir, args.direction_anchor,
                                 args.direction_layer, realized)
    for c, mu in means.items():
        print(f"  {c}: ||μ||={mu.norm().item():.3f} (hidden={mu.shape[0]})")

    alphas = [float(x) for x in args.alpha.split(",") if x.strip()]
    if args.mode == "ablate":
        alphas = [1.0]  # alpha unused for ablate
    print(f"alphas: {alphas} mode={args.mode}")

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
             "float16": torch.float16}[args.dtype]
    handle = load_model(args.model, device=args.device, dtype=dtype)
    model = handle.model
    layer_list, layer_path = _locate_layer_list(model, steer_layer - 1)
    if layer_list is None:
        print("Could not locate decoder layers", file=sys.stderr)
        return 3
    print(f"Decoder layers at {layer_path} (n={len(layer_list)}); steering at L{steer_layer}")
    steer_block = layer_list[steer_layer - 1]

    class_first_tok = _build_class_first_token_ids(handle, realized, bank)
    print(f"Class first-token ids: {class_first_tok}")

    tgt_runs: dict[str, list[RunManifest]] = {
        c: by_class[c][-args.n_target_per_class:] for c in realized
    }

    # 1) Baselines.
    print(f"Computing baselines (rollout={args.answer_rollout}) ...")
    baseline_records = []
    baseline_class_logits: dict[str, dict[str, float]] = {}
    baseline_rollout: dict[str, dict[str, Any]] = {}
    t0 = time.time()
    for tgt_class, runs in tgt_runs.items():
        for tgt in runs:
            if args.answer_rollout:
                # Zero direction → hook is no-op; equivalent to no steering.
                roll = _steer_rollout_trial(
                    handle, tgt, bank, args.prompt_variant,
                    steer_block, torch.zeros_like(next(iter(means.values()))),
                    0.0, "add", args.direction_anchor, args.steer_scope,
                    args.rollout_max_new_tokens, class_first_tok,
                )
                if roll is None:
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
                    "baseline_regen_answers_bool": roll["regen_answers_bool"],
                    "target_answers_bool": roll["target_answers_bool"],
                    "baseline_n_answer_flips": roll["n_answer_flips"],
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
    print(f"  baselines: {len(baseline_records)} in {time.time()-t0:.1f}s")

    # 2) Anchor position per target (for steer-scope=from_anchor).
    pos_index: dict[str, int] = {}
    if args.steer_scope == "from_anchor":
        for tgt_class, runs in tgt_runs.items():
            for tgt in runs:
                inputs = _context_with_reveal(handle, tgt, bank, args.prompt_variant)
                found = _find_anchors(handle.tokenizer, inputs["input_ids"])
                if any(k.startswith("__DEBUG_") for k in found):
                    print(f"  anchor lookup failed for {tgt.run_id}", file=sys.stderr)
                    continue
                if args.direction_anchor not in found:
                    continue
                pos_index[tgt.run_id] = int(found[args.direction_anchor])

    # 3) Steered trials: sweep α × (src_class, tgt_class) cells.
    steered_records = []
    t0 = time.time()
    pairs = [(s, t) for s in realized for t in realized if s != t]
    if args.mode == "ablate":
        # For ablate, direction is per (src,tgt) pair but only one α slot.
        pass
    total = sum(len(tgt_runs[t]) for s, t in pairs) * len(alphas)
    trial = 0
    for src_class, tgt_class in pairs:
        direction = means[src_class] - means[tgt_class]  # (hidden,)
        for alpha in alphas:
            for tgt in tgt_runs[tgt_class]:
                trial += 1
                if args.answer_rollout:
                    roll = _steer_rollout_trial(
                        handle, tgt, bank, args.prompt_variant,
                        steer_block, direction, alpha, args.mode,
                        args.direction_anchor, args.steer_scope,
                        args.rollout_max_new_tokens, class_first_tok,
                    )
                    if roll is None:
                        continue
                    steered_records.append({
                        "src_class": src_class,
                        "tgt_class": tgt_class,
                        "tgt_run": tgt.run_id,
                        "alpha": alpha,
                        "direction_norm": float(direction.norm()),
                        "steered_reveal_raw": roll["reveal_raw"],
                        "steered_canonical": roll["reveal_canonical"],
                        "steered_class_logits": roll["class_logits"],
                        "regen_answers_bool": roll["regen_answers_bool"],
                        "target_answers_bool": roll["target_answers_bool"],
                        "answer_flip_mask": roll["answer_flip_mask"],
                        "n_answer_flips": roll["n_answer_flips"],
                    })
                    if trial % 25 == 0 or trial == total:
                        print(f"  [{trial}/{total}] {src_class}->{tgt_class} α={alpha} "
                              f"tgt={tgt.run_id} → {roll['reveal_canonical']} "
                              f"(af={roll['n_answer_flips']})")
                    continue

                start_pos = pos_index.get(tgt.run_id) if args.steer_scope == "from_anchor" else None
                inputs = _context_with_reveal(handle, tgt, bank, args.prompt_variant)
                hook_h = steer_block.register_forward_hook(
                    _make_steer_hook(direction, alpha, args.mode, start_pos)
                )
                try:
                    raw, first_logits = _generate_reveal_greedy(handle, inputs)
                finally:
                    hook_h.remove()
                canon = parse_reveal_to_canonical(raw, bank)
                class_logits = {cid: float(first_logits[tid])
                                for cid, tid in class_first_tok.items()}
                steered_records.append({
                    "src_class": src_class,
                    "tgt_class": tgt_class,
                    "tgt_run": tgt.run_id,
                    "alpha": alpha,
                    "direction_norm": float(direction.norm()),
                    "steered_reveal_raw": raw.strip(),
                    "steered_canonical": canon,
                    "steered_class_logits": class_logits,
                })
                if trial % 25 == 0 or trial == total:
                    print(f"  [{trial}/{total}] {src_class}->{tgt_class} α={alpha} "
                          f"tgt={tgt.run_id} → {canon}")
    print(f"Steered trials: {len(steered_records)} in {time.time()-t0:.1f}s")

    # 4) Per-(src, tgt, α) summaries.
    summaries = {}
    for src_class, tgt_class in pairs:
        for alpha in alphas:
            cell = [r for r in steered_records
                    if r["src_class"] == src_class
                    and r["tgt_class"] == tgt_class
                    and r["alpha"] == alpha]
            n = len(cell)
            if not n:
                continue
            flip_to_src = sum(1 for r in cell
                              if r["steered_canonical"] == src_class) / n
            kept_tgt = sum(1 for r in cell
                           if r["steered_canonical"] == tgt_class) / n
            dist: dict[str, int] = {}
            for r in cell:
                c = r["steered_canonical"] or "__unparsed__"
                dist[c] = dist.get(c, 0) + 1
            patch_diffs = [
                r["steered_class_logits"][src_class]
                - r["steered_class_logits"][tgt_class]
                for r in cell
            ]
            base_diffs = [
                baseline_class_logits[r["tgt_run"]][src_class]
                - baseline_class_logits[r["tgt_run"]][tgt_class]
                for r in cell
            ]
            mean_patch = sum(patch_diffs) / n
            mean_base = sum(base_diffs) / n
            summaries[f"{src_class}->{tgt_class}@α={alpha}"] = {
                "n": n,
                "flip_to_src": flip_to_src,
                "kept_tgt": kept_tgt,
                "flip_to_other": 1 - flip_to_src - kept_tgt,
                "distribution": dist,
                "logit_diff_steered": mean_patch,
                "logit_diff_baseline": mean_base,
                "logit_diff_delta": mean_patch - mean_base,
            }

    results = {
        "run_dir": str(run_dir),
        "src_residuals_dir": str(src_dir),
        "model": args.model,
        "torch_dtype": args.dtype,
        "direction_anchor": args.direction_anchor,
        "direction_layer": args.direction_layer,
        "steer_layer": steer_layer,
        "steer_scope": args.steer_scope,
        "mode": args.mode,
        "alphas": alphas,
        "prompt_variant": args.prompt_variant,
        "n_target_per_class": args.n_target_per_class,
        "realized_classes": realized,
        "class_first_token_ids": class_first_tok,
        "class_mean_norms": {c: float(means[c].norm()) for c in realized},
        "baselines": baseline_records,
        "steered_trials": steered_records,
        "summaries": summaries,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}")

    # Console summary: per-α flip-to-source matrices.
    for alpha in alphas:
        print(f"\nFlip-to-source rate matrix at α={alpha} (mode={args.mode}):")
        header = "  src\\tgt | " + " | ".join(f"{c[:8]:>8}" for c in realized)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for src_class in realized:
            row = [f"  {src_class[:8]:>8} |"]
            for tgt_class in realized:
                if src_class == tgt_class:
                    row.append(f"{'   -   ':>8}")
                    continue
                key = f"{src_class}->{tgt_class}@α={alpha}"
                s = summaries.get(key)
                v = (s["flip_to_src"] if s else 0) or 0
                row.append(f"{v*100:7.1f}%")
            print("  " + " | ".join(row[0:1] + row[1:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
