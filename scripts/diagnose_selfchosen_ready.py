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
import math
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
from twenty_q.prompts import SELF_CHOSEN_VARIANTS, self_chosen_prompt
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
    p.add_argument(
        "--stop-when-n-classes-hit-quota",
        type=int,
        default=None,
        help=(
            "Optional early-stop target for diversity runs. If set, stop once "
            "at least this many classes have reached --n-per-candidate, "
            "instead of waiting for the full candidate list."
        ),
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
        "--prompt-variant",
        default="default",
        choices=list(SELF_CHOSEN_VARIANTS),
        help=(
            "Self-chosen prompt wording. 'less_obvious' adds a nudge aimed at "
            "breaking the 12B {elephant,cow,dog,horse} attractor (D-33)."
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
    prompt_variant: str = "default",
) -> tuple[dict[str, Any], torch.Tensor]:
    display_names = {c.id: c.display for c in bank.candidates}
    perm = shuffle_candidates(bank.candidate_ids, seed=seed)
    rendered = self_chosen_prompt(perm, display_names, variant=prompt_variant)

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


def _classes_at_quota(counts: dict[str, int], quota: int) -> list[str]:
    return [cid for cid, count in counts.items() if count >= quota]


def _should_stop_collection(
    counts: dict[str, int],
    quota: int,
    *,
    stop_when_n_classes_hit_quota: int | None,
) -> bool:
    classes_at_quota = _classes_at_quota(counts, quota)
    target = stop_when_n_classes_hit_quota or len(counts)
    return len(classes_at_quota) >= target


def _summarize_attempt_distribution(
    all_rows: list[dict[str, Any]],
    *,
    candidate_ids: list[str],
    n_questions: int,
) -> dict[str, Any]:
    parsed_counts = {cid: 0 for cid in candidate_ids}
    n_ready_ok = 0
    n_reveal_parsed = 0
    n_answer_slots = 0
    n_answer_parsed = 0
    n_answer_correct = 0

    for row in all_rows:
        n_ready_ok += int(bool(row.get("ready_ok")))
        reveal = row.get("reveal_canonical_id")
        if reveal in parsed_counts:
            parsed_counts[reveal] += 1
            n_reveal_parsed += 1
        n_answer_slots += n_questions
        n_answer_parsed += int(row.get("n_answer_parsed") or 0)
        n_answer_correct += int(row.get("n_answer_correct") or 0)

    parsed_total = sum(parsed_counts.values())
    probs = [count / parsed_total for count in parsed_counts.values() if count > 0]
    entropy_bits = -sum(p * math.log2(p) for p in probs) if probs else None
    top_count = max(parsed_counts.values(), default=0)
    return {
        "counts_by_candidate": parsed_counts,
        "n_distinct_candidates_parsed": sum(int(count > 0) for count in parsed_counts.values()),
        "ready_parse_success": (n_ready_ok / len(all_rows)) if all_rows else None,
        "reveal_parse_success": (n_reveal_parsed / len(all_rows)) if all_rows else None,
        "answer_parse_success": (n_answer_parsed / n_answer_slots) if n_answer_slots else None,
        "answer_correct_on_parsed": (
            n_answer_correct / n_answer_parsed if n_answer_parsed else None
        ),
        "parsed_reveal_entropy_bits": entropy_bits,
        "parsed_reveal_effective_classes": (2 ** entropy_bits) if entropy_bits is not None else None,
        "parsed_reveal_top1_share": (top_count / parsed_total) if parsed_total else None,
    }


def _compare_against_persistence(
    analysis_class_ids: list[str],
    ready_nc_by_layer: list[float],
    ready_contrast: dict[str, Any],
    persistence_path: Path,
) -> dict[str, Any] | None:
    if not persistence_path.exists():
        return None
    with persistence_path.open() as f:
        persistence = json.load(f)

    persistence_class_ids = list(persistence.get("class_ids", []))
    if set(persistence_class_ids) != set(analysis_class_ids):
        return {
            "persistence_results_path": str(persistence_path),
            "comparison_skipped_reason": (
                "class_ids_mismatch: self_chosen="
                f"{analysis_class_ids} persistence={persistence_class_ids}"
            ),
        }

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
    stop_when_n_classes_hit_quota = args.stop_when_n_classes_hit_quota
    if stop_when_n_classes_hit_quota is not None:
        if stop_when_n_classes_hit_quota < 1 or stop_when_n_classes_hit_quota > len(candidate_ids):
            print(
                "--stop-when-n-classes-hit-quota must be between 1 and the number "
                f"of candidate ids ({len(candidate_ids)}).",
                file=sys.stderr,
            )
            return 2

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
    while (
        not _should_stop_collection(
            counts,
            args.n_per_candidate,
            stop_when_n_classes_hit_quota=stop_when_n_classes_hit_quota,
        )
        and attempt < args.max_attempts
    ):
        run_id = f"attempt_{attempt:03d}"
        seed = args.seed_offset + attempt
        row, ready_states = _run_one(
            handle=handle,
            bank=bank,
            seed=seed,
            run_id=run_id,
            out_dir=out_dir,
            temperature=args.temperature,
            prompt_variant=args.prompt_variant,
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

    classes_at_quota = _classes_at_quota(counts, args.n_per_candidate)
    complete = len(classes_at_quota) == len(candidate_ids)
    realized_class_ids = classes_at_quota
    partial = (not complete) and len(realized_class_ids) >= 2
    attempt_distribution = _summarize_attempt_distribution(
        all_rows, candidate_ids=list(candidate_ids), n_questions=len(question_ids)
    )
    results: dict[str, Any] = {
        "model": args.model,
        "model_revision": handle.model_revision,
        "tokenizer_revision": handle.tokenizer_revision,
        "torch_dtype": args.dtype,
        "temperature": float(args.temperature),
        "prompt_variant": args.prompt_variant,
        "candidate_ids": list(candidate_ids),
        "question_ids": list(question_ids),
        "n_per_candidate_target": args.n_per_candidate,
        "max_attempts": args.max_attempts,
        "stop_when_n_classes_hit_quota": stop_when_n_classes_hit_quota,
        "attempts_run": attempt,
        "complete": complete,
        "partial_analysis_emitted": partial,
        "realized_class_ids": realized_class_ids,
        "n_classes_at_quota": len(realized_class_ids),
        "counts_kept_by_candidate": counts,
        "attempt_distribution": attempt_distribution,
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
            list(analysis_class_ids),
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
    print(
        "  attempt distribution: "
        f"{attempt_distribution['n_distinct_candidates_parsed']} parsed classes, "
        f"reveal parse={(attempt_distribution['reveal_parse_success'] or 0.0):.1%}, "
        f"ready parse={(attempt_distribution['ready_parse_success'] or 0.0):.1%}, "
        f"top1 share={(attempt_distribution['parsed_reveal_top1_share'] or 0.0):.1%}, "
        f"effective classes={((attempt_distribution['parsed_reveal_effective_classes']) or 0.0):.2f}"
    )
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
            if "overall" in comparison:
                print(
                    "  resembles: "
                    f"{comparison['overall']} "
                    f"(votes={comparison['vote_by_metric']})"
                )
            else:
                print(
                    "  persistence comparison skipped: "
                    f"{comparison['comparison_skipped_reason']}"
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
