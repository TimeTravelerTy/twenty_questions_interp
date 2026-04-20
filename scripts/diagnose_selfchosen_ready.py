"""Self-chosen Ready-state diagnostic for the M3 smoke bank.

This is the direct follow-through from STATUS.md after D-21/H-rotation.
By default it uses the same 4 candidates and primary question set used in the
`verbalized_index` persistence diagnostic, but it can also run on the full
20-animal bank via `--candidates all`.

The workflow is the same in either regime:

1. Present the requested candidate subset.
2. Capture the Ready-state activations.
3. Run the primary yes/no questions.
4. Ask for the secret at the end so each run can be labeled by the model's own
   chosen candidate.
5. Compute Ready-state NC LOO and within-vs-between contrast across the
   realized revealed secrets.
6. Optionally compare those Ready-state statistics against persistence
   State A vs State B to determine which regime self-chosen resembles.

Usage:
    uv run python scripts/diagnose_selfchosen_ready.py --device auto --dtype bfloat16
    uv run python scripts/diagnose_selfchosen_ready.py --candidates all --max-attempts 200
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from twenty_q.banks import Bank, load_bank, resolve_id_selector, subset_bank
from twenty_q.config import MODEL_MAIN
from twenty_q.dialogue import (
    capture_ready_state,
    collect_question_turns,
    elicit_reveal_after_turns,
    load_model,
    parse_ready,
    parse_reveal_to_canonical,
)
from twenty_q.manifest import RunManifest
from twenty_q.permutations import shuffle_candidates
from twenty_q.prompts import self_chosen_prompt
from twenty_q.readouts import (
    layerwise_loo_accuracy_nearest_centroid,
    within_between_contrast,
)

DEFAULT_CANDIDATES = ("tiger", "eagle", "frog", "salmon")
DEFAULT_PRIMARY_QUESTION_IDS = (
    "is_mammal",
    "is_bird",
    "lives_primarily_in_water",
    "has_four_legs",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL_MAIN)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--out-dir", default="runs/diag/selfchosen_ready_smoke")
    p.add_argument(
        "--candidates",
        default=",".join(DEFAULT_CANDIDATES),
        help="Comma-separated candidate ids, or 'all' for the full bank.",
    )
    p.add_argument(
        "--question-ids",
        default=",".join(DEFAULT_PRIMARY_QUESTION_IDS),
        help="Comma-separated question ids, or 'all' for the full question bank.",
    )
    p.add_argument(
        "--n-per-candidate",
        type=int,
        default=2,
        help="Keep running until this many reveal-parsed runs exist for each candidate.",
    )
    p.add_argument(
        "--max-attempts",
        type=int,
        default=40,
        help="Hard cap on attempted self-chosen runs.",
    )
    p.add_argument("--seed-offset", type=int, default=2_000_000)
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help=(
            "Decoding temperature for generation. 0.0 = greedy (default). "
            ">0 enables do_sample=True with this temperature across Ready, "
            "question-turn, and reveal generations — intended to break the "
            "greedy choice collapse observed in the T=0 smoke."
        ),
    )
    p.add_argument(
        "--persistence-results",
        default="runs/diag/persistence_smoke/results.json",
        help="Optional diagnose_persistence results.json for A-vs-B comparison.",
    )
    return p.parse_args()


def _run_one(
    handle,
    bank: Bank,
    seed: int,
    run_id: str,
    out_dir: Path,
    temperature: float = 0.0,
) -> tuple[dict[str, Any], torch.Tensor]:
    display_names = {c.id: c.display for c in bank.candidates}
    perm = shuffle_candidates(bank.candidate_ids, seed=seed)
    rendered = self_chosen_prompt(perm, display_names)

    # With temperature sampling we still want per-attempt reproducibility,
    # so seed the generators before each stochastic generate call.
    if temperature and temperature > 0.0:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    ready_states, ready_raw = capture_ready_state(
        handle, rendered, temperature=temperature
    )

    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    act_path = run_dir / "activations.pt"
    torch.save(ready_states, act_path)

    turns, turn_activation_paths = collect_question_turns(
        handle=handle,
        rendered=rendered,
        ready_output=ready_raw,
        questions=list(bank.questions),
        run_dir=run_dir,
        temperature=temperature,
    )
    reveal_raw = elicit_reveal_after_turns(
        handle, rendered, ready_raw, turns, temperature=temperature
    )
    reveal_canonical = parse_reveal_to_canonical(reveal_raw, bank)

    answers: list[dict[str, Any]] = []
    n_correct = 0
    n_parsed = 0
    for turn in turns:
        expected = None
        correct = None
        if reveal_canonical is not None:
            expected = bool(bank.answer(reveal_canonical, turn.question_id))
            if turn.answer_bool is not None:
                correct = turn.answer_bool == expected
                n_parsed += 1
                n_correct += int(correct)
        answers.append(
            {
                "qid": turn.question_id,
                "question_text": turn.question_text,
                "raw": turn.raw_model_output,
                "parsed": turn.answer_bool,
                "bank": expected,
                "correct": correct,
            }
        )

    manifest = RunManifest(
        run_id=run_id,
        condition="self_chosen",
        model_name=handle.model_name,
        model_revision=handle.model_revision,
        tokenizer_revision=handle.tokenizer_revision,
        torch_dtype=handle.torch_dtype,
        device=handle.device,
        prompt_template_id=rendered.template_id,
        seed=seed,
        decoding_params={
            "do_sample": bool(temperature and temperature > 0.0),
            "temperature": float(temperature),
            "max_new_tokens": 8,
            "reveal_tokens": 48,
        },
        permutation=list(perm.order),
        ready_raw_output=ready_raw,
        ready_parse_ok=parse_ready(ready_raw),
        turns=turns,
        end_of_game_reveal=reveal_raw,
        reveal_canonical_id=reveal_canonical,
        activation_paths={i: str(act_path) for i in range(ready_states.shape[0])},
        turn_activation_paths=turn_activation_paths,
        hidden_size=int(ready_states.shape[-1]),
    )
    manifest_path = run_dir / "manifest.json"
    manifest.save(manifest_path)

    row = {
        "run_id": run_id,
        "seed": seed,
        "permutation": list(perm.order),
        "ready_raw": ready_raw,
        "ready_ok": manifest.ready_parse_ok,
        "reveal_raw": reveal_raw,
        "reveal_canonical_id": reveal_canonical,
        "ready_state_path": str(act_path),
        "manifest_path": str(manifest_path),
        "answers": answers,
        "n_answer_parsed": n_parsed,
        "n_answer_correct": n_correct,
        "pct_answer_correct": (n_correct / n_parsed) if n_parsed else None,
    }
    with (run_dir / "result.json").open("w") as f:
        json.dump(row, f, indent=2, default=str)
    return row, ready_states


def _choose_closer(value: float | None, a: float | None, b: float | None) -> str:
    if value is None or a is None or b is None:
        return "unknown"
    da = abs(value - a)
    db = abs(value - b)
    if abs(da - db) < 1e-12:
        return "tie"
    return "state_a" if da < db else "state_b"


def _compare_against_persistence(
    ready_nc_by_layer: list[float],
    ready_contrast: dict[str, Any],
    persistence_path: Path,
) -> dict[str, Any] | None:
    if not persistence_path.exists():
        return None
    with persistence_path.open() as f:
        persistence = json.load(f)

    p = persistence["persistence"]
    layer = int(p["post13_best_layer_by_a"])
    self_nc = ready_nc_by_layer[layer]
    state_a_nc = float(p["nc_loo_state_a_by_layer"][layer])
    state_b_nc = float(p["nc_loo_state_b_by_layer"][layer])
    self_contrast = ready_contrast.get("contrast_post13")
    state_a_contrast = p["state_a_within_between"].get("contrast_post13")
    state_b_contrast = p["state_b_within_between"].get("contrast_post13")

    nc_vote = _choose_closer(self_nc, state_a_nc, state_b_nc)
    contrast_vote = _choose_closer(self_contrast, state_a_contrast, state_b_contrast)
    overall = nc_vote if nc_vote == contrast_vote else "mixed"
    return {
        "persistence_results_path": str(persistence_path),
        "reference_layer": layer,
        "ready_nc_at_reference_layer": self_nc,
        "reference_nc": {
            "state_a": state_a_nc,
            "state_b": state_b_nc,
        },
        "ready_contrast_post13": self_contrast,
        "reference_contrast_post13": {
            "state_a": state_a_contrast,
            "state_b": state_b_contrast,
        },
        "vote_by_metric": {
            "nc_at_reference_layer": nc_vote,
            "contrast_post13": contrast_vote,
        },
        "overall": overall,
    }


def main() -> int:
    args = parse_args()
    full_bank = load_bank()
    candidate_ids = resolve_id_selector(
        args.candidates, full_bank.candidate_ids, label="candidate"
    )
    question_ids = resolve_id_selector(
        args.question_ids, full_bank.question_ids, label="question"
    )
    bank = subset_bank(full_bank, candidate_ids=candidate_ids, question_ids=question_ids)

    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]

    print(f"Loading {args.model} on {args.device} ({args.dtype}) ...")
    t0 = time.time()
    handle = load_model(args.model, device=args.device, dtype=dtype)
    print(f"  loaded in {time.time() - t0:.1f}s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kept_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    ready_states_by_cid: dict[str, list[torch.Tensor]] = {cid: [] for cid in candidate_ids}
    counts = {cid: 0 for cid in candidate_ids}

    started = time.time()
    attempt = 0
    while min(counts.values()) < args.n_per_candidate and attempt < args.max_attempts:
        run_id = f"attempt_{attempt:03d}"
        seed = args.seed_offset + attempt
        row, ready_states = _run_one(
            handle=handle,
            bank=bank,
            seed=seed,
            run_id=run_id,
            out_dir=out_dir,
            temperature=args.temperature,
        )
        reveal = row["reveal_canonical_id"]
        kept = reveal in counts and counts[reveal] < args.n_per_candidate
        row["kept_for_analysis"] = kept
        row["drop_reason"] = None if kept else (
            "unparsed_or_unknown_reveal"
            if reveal not in counts
            else "quota_already_filled"
        )
        if kept:
            counts[reveal] += 1
            kept_rows.append(row)
            ready_states_by_cid[reveal].append(ready_states)
        all_rows.append(row)
        answer_blurb = (
            f"{row['n_answer_correct']}/{row['n_answer_parsed']}"
            if row["n_answer_parsed"]
            else "n/a"
        )
        print(
            f"  [{attempt + 1}/{args.max_attempts}] {run_id} "
            f"ready={row['ready_raw']!r} reveal={reveal!r} "
            f"answers={answer_blurb} kept={kept}"
        )
        attempt += 1

    complete = min(counts.values()) >= args.n_per_candidate
    realized_class_ids = [
        cid for cid in candidate_ids if counts[cid] >= args.n_per_candidate
    ]
    partial = (not complete) and len(realized_class_ids) >= 2
    results: dict[str, Any] = {
        "model": args.model,
        "model_revision": handle.model_revision,
        "tokenizer_revision": handle.tokenizer_revision,
        "torch_dtype": args.dtype,
        "temperature": float(args.temperature),
        "candidate_ids": list(candidate_ids),
        "question_ids": list(question_ids),
        "n_per_candidate_target": args.n_per_candidate,
        "max_attempts": args.max_attempts,
        "attempts_run": attempt,
        "complete": complete,
        "partial_analysis_emitted": partial,
        "realized_class_ids": realized_class_ids,
        "counts_kept_by_candidate": counts,
        "all_rows": all_rows,
        "kept_rows": kept_rows,
    }

    if complete or partial:
        # Balance each realized class to exactly n_per_candidate runs so NC LOO
        # and contrast are not dominated by any one class.
        analysis_class_ids = realized_class_ids
        balanced_states_by_cid: dict[str, list[torch.Tensor]] = {
            cid: ready_states_by_cid[cid][: args.n_per_candidate]
            for cid in analysis_class_ids
        }
        ordered_states = [
            state for cid in analysis_class_ids for state in balanced_states_by_cid[cid]
        ]
        labels = [
            cid for cid in analysis_class_ids for _ in balanced_states_by_cid[cid]
        ]
        ready_nc_by_layer = layerwise_loo_accuracy_nearest_centroid(
            ordered_states, labels, list(analysis_class_ids)
        )
        ready_contrast = within_between_contrast(balanced_states_by_cid)
        n_layers = len(ready_nc_by_layer)
        best_layer = 13 + int(np.argmax(ready_nc_by_layer[13:])) if n_layers > 13 else int(
            np.argmax(ready_nc_by_layer)
        )

        per_candidate: dict[str, dict[str, Any]] = {}
        n_answer_total = 0
        n_answer_correct = 0
        for cid in candidate_ids:
            kept = [row for row in kept_rows if row["reveal_canonical_id"] == cid]
            per_candidate[cid] = {
                "n_runs": len(kept),
                "n_ready_ok": sum(int(row["ready_ok"]) for row in kept),
                "n_answer_parsed": sum(int(row["n_answer_parsed"]) for row in kept),
                "n_answer_correct": sum(int(row["n_answer_correct"]) for row in kept),
            }
            per_candidate[cid]["pct_answer_correct"] = (
                per_candidate[cid]["n_answer_correct"] / per_candidate[cid]["n_answer_parsed"]
                if per_candidate[cid]["n_answer_parsed"]
                else None
            )
            n_answer_total += per_candidate[cid]["n_answer_parsed"]
            n_answer_correct += per_candidate[cid]["n_answer_correct"]

        comparison = _compare_against_persistence(
            ready_nc_by_layer,
            ready_contrast,
            Path(args.persistence_results),
        )

        results["ready_analysis"] = {
            "class_ids": analysis_class_ids,
            "n_per_class_balanced": args.n_per_candidate,
            "n_layers": n_layers,
            "nc_loo_by_layer": ready_nc_by_layer,
            "within_between": ready_contrast,
            "post13_best_layer": best_layer,
            "post13_best_nc": ready_nc_by_layer[best_layer],
            "comparison_to_persistence": comparison,
        }
        results["correctness"] = {
            "n_answer_parsed": n_answer_total,
            "n_answer_correct": n_answer_correct,
            "pct_answer_correct": (
                n_answer_correct / n_answer_total if n_answer_total else None
            ),
            "per_candidate": per_candidate,
        }

    out_path = out_dir / "results.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"  {attempt} attempts in {time.time() - started:.1f}s")
    print(f"  kept counts: {counts}")
    print(f"  realized classes: {realized_class_ids}")
    if complete or partial:
        ready = results["ready_analysis"]
        suffix = (
            ""
            if complete
            else f" (PARTIAL — {len(realized_class_ids)}/{len(candidate_ids)} classes)"
        )
        header = f"Self-chosen Ready summary{suffix}"
        print(f"\n== {header} ==")
        print(f"  classes: {ready['class_ids']}")
        print(
            f"  layer {ready['post13_best_layer']} best post-13 NC: "
            f"{ready['post13_best_nc']:.2%}"
        )
        print(
            "  post-13 within-between contrast: "
            f"{ready['within_between'].get('contrast_post13', float('nan')):+.2e}"
        )
        if results.get("correctness"):
            corr = results["correctness"]
            pct = corr["pct_answer_correct"]
            print(
                f"  primary correctness: {corr['n_answer_correct']}/{corr['n_answer_parsed']} "
                f"({(pct * 100.0) if pct is not None else float('nan'):.1f}%)"
            )
        comparison = ready.get("comparison_to_persistence")
        if comparison is not None:
            print(
                "  resembles: "
                f"{comparison['overall']} "
                f"(votes={comparison['vote_by_metric']})"
            )
    else:
        print(
            "\nFewer than 2 realized classes met quota; no ready-state analysis "
            f"emitted. Wrote raw results to {out_path}",
            file=sys.stderr,
        )
        return 1

    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
