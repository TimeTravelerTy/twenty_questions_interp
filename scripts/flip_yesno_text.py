"""M4 phase 2d (c-text) — Yes/no answer text-level flip behavioral test.

For each kept run T in a self-chosen collection, regenerate the reveal
under counterfactual dialogues where exactly one yes/no answer text has
been flipped (Yes -> No or No -> Yes). Pure behavior — no patching,
just rebuilding the chat context with one answer literal changed.

This directly tests the M4 phase 2c-iii (D-39) "improvisation"
hypothesis: that the model re-derives the class at reveal time from
accumulated dialogue evidence, rather than reading off a stored
commitment.

Predictions (per D-39):
- If the model uses dialogue evidence: flipping a yes/no answer should
  change the reveal — either to a different class within the attractor
  set, or out to a less-frequent class consistent with the new pattern.
- If the model does NOT use dialogue evidence: flipping has no effect;
  the original reveal class persists.
- Asymmetry across turns is informative: if flipping turn-4 matters
  more than flipping turn-1, the most recent constraint dominates the
  re-derivation. If they matter equally, the model treats the full
  history symmetrically.

Output: per-trial JSON with original class, flipped turn, flipped
answer pattern, and the resulting reveal canonical / raw text.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

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
    p.add_argument("--out-json", required=True, help="Output path for trial records.")
    p.add_argument("--model", default=MODEL_MAIN)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--prompt-variant", default="default")
    p.add_argument("--n-per-class", type=int, default=20,
                   help="Subsample to first K runs of each realized class. Default 20 "
                        "(matches M3 scale-up methodology).")
    p.add_argument("--turns-to-flip", type=str, default="1,2,3,4",
                   help="Comma-separated turn indices (1..4) to flip. Default all 4.")
    return p.parse_args()


def _load_kept_manifests(run_dir: Path) -> list[RunManifest]:
    out: list[RunManifest] = []
    for a in sorted(run_dir.glob("attempt_*")):
        mp = a / "manifest.json"
        if not mp.exists():
            continue
        m = RunManifest.load(mp)
        if m.reveal_canonical_id is not None:
            out.append(m)
    return out


def _flip_turn_n_answer(manifest: RunManifest, n: int) -> tuple[list, str, str]:
    """Return (modified_turns, original_answer_text, flipped_answer_text)."""
    turns = [copy.copy(t) for t in manifest.turns]
    orig_raw = turns[n - 1].raw_model_output.strip()
    orig_lower = orig_raw.lower()
    if orig_lower.startswith("yes"):
        flipped_raw = "No"
    elif orig_lower.startswith("no"):
        flipped_raw = "Yes"
    else:
        return turns, orig_raw, ""  # unparseable; caller should skip
    turns[n - 1].raw_model_output = flipped_raw
    turns[n - 1].answer_bool = not turns[n - 1].answer_bool if turns[n - 1].answer_bool is not None else None
    return turns, orig_raw, flipped_raw


def _build_full_reveal_inputs(
    handle: ModelHandle,
    manifest: RunManifest,
    turns_override: list,
    bank,
    prompt_variant: str,
) -> dict[str, torch.Tensor]:
    """Build the full chat ending at the reveal-generation prompt, using a
    (possibly modified) turn list for the turn answers."""
    display_names = {c.id: c.display for c in bank.candidates}
    perm = Permutation(order=tuple(manifest.permutation))
    rendered = self_chosen_prompt(perm, display_names, variant=prompt_variant)
    extra = [
        *_history_to_chat_turns(manifest.ready_raw_output, turns_override),
        {"role": "user", "content": REVEAL_USER_MESSAGE},
    ]
    return _build_chat_input_ids(handle, rendered, extra_turns=extra)


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
    bank = load_bank()
    turns_to_flip = [int(x) for x in args.turns_to_flip.split(",") if x.strip()]
    if not all(1 <= n <= 4 for n in turns_to_flip):
        print(f"--turns-to-flip must contain values in 1..4, got {turns_to_flip}", file=sys.stderr)
        return 2

    manifests = _load_kept_manifests(run_dir)
    print(f"Found {len(manifests)} kept runs in {run_dir}")

    # Subsample to first n_per_class per realized class
    by_class: dict[str, list[RunManifest]] = {}
    for m in manifests:
        by_class.setdefault(m.reveal_canonical_id, []).append(m)
    realized = sorted(by_class.keys())
    print(f"Realized classes: {realized}")
    selected: list[RunManifest] = []
    for c in realized:
        selected.extend(by_class[c][: args.n_per_class])
    print(f"Subsample (n_per_class={args.n_per_class}): {len(selected)} runs")
    print(f"Turns to flip: {turns_to_flip}")

    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    handle = load_model(args.model, device=args.device, dtype=dtype)

    # ---- Per-run baseline reveal via fresh replay (no flip), for sanity ----
    baselines: list[dict] = []
    t0 = time.time()
    for m in selected:
        inputs = _build_full_reveal_inputs(handle, m, list(m.turns), bank, args.prompt_variant)
        raw = _generate_reveal_greedy(handle, inputs)
        canon = parse_reveal_to_canonical(raw, bank)
        baselines.append({
            "run_id": m.run_id,
            "class": m.reveal_canonical_id,
            "original_pattern": [t.answer_bool for t in m.turns],
            "baseline_replay_canonical": canon,
            "baseline_replay_raw": raw.strip(),
            "ondisk_canonical": m.reveal_canonical_id,
        })
    print(f"Baselines: {len(baselines)} runs in {time.time()-t0:.1f}s")

    # ---- Flipped-turn trials ----
    trials: list[dict] = []
    t0 = time.time()
    total = len(selected) * len(turns_to_flip)
    i = 0
    for m in selected:
        for n in turns_to_flip:
            i += 1
            modified_turns, orig_ans, flipped_ans = _flip_turn_n_answer(m, n)
            if not flipped_ans:
                trials.append({
                    "run_id": m.run_id, "class": m.reveal_canonical_id,
                    "flipped_turn": n, "skipped": True,
                    "reason": f"unparseable original answer: {orig_ans!r}",
                })
                continue
            inputs = _build_full_reveal_inputs(handle, m, modified_turns, bank, args.prompt_variant)
            raw = _generate_reveal_greedy(handle, inputs)
            canon = parse_reveal_to_canonical(raw, bank)
            orig_pattern = [t.answer_bool for t in m.turns]
            flipped_pattern = list(orig_pattern)
            flipped_pattern[n - 1] = not flipped_pattern[n - 1] if flipped_pattern[n - 1] is not None else None
            trials.append({
                "run_id": m.run_id,
                "class": m.reveal_canonical_id,
                "flipped_turn": n,
                "original_answer_text": orig_ans,
                "flipped_answer_text": flipped_ans,
                "original_pattern": orig_pattern,
                "flipped_pattern": flipped_pattern,
                "flipped_canonical": canon,
                "flipped_raw": raw.strip(),
            })
            if i % 40 == 0 or i == total:
                print(f"  [{i}/{total}] {m.reveal_canonical_id}/{m.run_id} "
                      f"flip_t{n}: {orig_ans}->{flipped_ans} -> {canon}", flush=True)
    print(f"Flipped trials: {len(trials)} in {time.time()-t0:.1f}s")

    # ---- Per-class × per-flipped-turn summary ----
    summary: dict[str, dict] = {}
    for cls in realized:
        for n in turns_to_flip:
            cell = [t for t in trials
                    if t.get("class") == cls
                    and t.get("flipped_turn") == n
                    and not t.get("skipped")]
            if not cell:
                continue
            kept_class = sum(1 for t in cell if t["flipped_canonical"] == cls) / len(cell)
            unparsed = sum(1 for t in cell if t["flipped_canonical"] is None)
            dist: dict[str, int] = {}
            for t in cell:
                k = t["flipped_canonical"] or "__unparsed__"
                dist[k] = dist.get(k, 0) + 1
            summary[f"{cls}/flip_t{n}"] = {
                "n": len(cell),
                "kept_class_rate": kept_class,
                "unparsed": unparsed,
                "distribution": dist,
            }

    out = {
        "run_dir": str(run_dir),
        "model": args.model,
        "torch_dtype": args.dtype,
        "prompt_variant": args.prompt_variant,
        "n_per_class": args.n_per_class,
        "turns_to_flip": turns_to_flip,
        "realized_classes": realized,
        "baselines": baselines,
        "trials": trials,
        "summary": summary,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out_json}")

    # Console summary table
    print()
    print(f"Kept-class rate matrix (rows=class, cols=turn flipped):")
    header = "  class    " + " | ".join(f"  T{n}  " for n in turns_to_flip)
    print(header)
    for cls in realized:
        row = [f"  {cls:8s}"]
        for n in turns_to_flip:
            s = summary.get(f"{cls}/flip_t{n}")
            if s is None:
                row.append("  --  ")
            else:
                row.append(f"{s['kept_class_rate']*100:5.1f}%")
        print(" | ".join(row))

    print()
    print("Output distributions per (class, flipped_turn):")
    for k in sorted(summary.keys()):
        s = summary[k]
        print(f"  {k:20s}  n={s['n']:2d} kept={s['kept_class_rate']*100:5.1f}%  dist={s['distribution']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
