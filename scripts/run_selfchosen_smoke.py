"""Batch-run self-chosen smoke dialogues.

M2 purpose only: capture Ready-position hidden states + optional end-of-game
reveal. For M3 smoke tests, `--question-ids` continues with fixed question
turns and captures the pre-answer state at each turn.

Usage:
    uv run python scripts/run_selfchosen_smoke.py --n 40
    uv run python scripts/run_selfchosen_smoke.py --n 2 --question-ids is_mammal,can_fly
"""
from __future__ import annotations

import argparse
import sys
import time

from twenty_q.banks import load_bank
from twenty_q.config import MODEL_DEBUG, SELFCHOSEN_RUNS_DIR
from twenty_q.dialogue import load_model, run_selfchosen_dialogue
from twenty_q.permutations import shuffle_candidates


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL_DEBUG)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--question-ids", default="",
                   help="Comma-separated question ids to ask after Ready; empty = Ready-only.")
    p.add_argument("--seed-offset", type=int, default=1_000_000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-reveal", action="store_true",
                   help="Skip the end-of-game reveal step.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bank = load_bank()
    question_lookup = {q.id: q for q in bank.questions}
    question_ids = [x.strip() for x in args.question_ids.split(",") if x.strip()]
    unknown_questions = sorted(set(question_ids) - set(question_lookup))
    if unknown_questions:
        print(f"Unknown question ids: {unknown_questions}", file=sys.stderr)
        return 2
    questions = [question_lookup[qid] for qid in question_ids]

    print(f"Loading {args.model} on {args.device} ...")
    t0 = time.time()
    handle = load_model(args.model, device=args.device)
    print(f"  loaded in {time.time() - t0:.1f}s")

    failures: list[str] = []
    reveal_hits = 0
    parse_ok_count = 0
    for k in range(args.n):
        seed = args.seed_offset + k
        perm = shuffle_candidates(bank.candidate_ids, seed=seed)
        run_id = f"sc_{k:03d}"
        t0 = time.time()
        try:
            manifest = run_selfchosen_dialogue(
                handle=handle,
                bank=bank,
                perm=perm,
                seed=seed,
                run_id=run_id,
                out_dir=SELFCHOSEN_RUNS_DIR,
                elicit_reveal_after=not args.no_reveal,
                questions=questions or None,
            )
        except Exception as e:  # noqa: BLE001
            failures.append(f"{run_id}: {type(e).__name__}: {e}")
            print(f"  [{k+1}/{args.n}] {run_id}  FAILED: {e}")
            continue
        parse_ok_count += int(manifest.ready_parse_ok or 0)
        reveal_hits += int(manifest.reveal_canonical_id is not None)
        dt = time.time() - t0
        reveal_blurb = (
            f"  reveal->{manifest.reveal_canonical_id!r}"
            if manifest.end_of_game_reveal is not None
            else ""
        )
        print(
            f"  [{k+1}/{args.n}] {run_id}  "
            f"ready={manifest.ready_raw_output!r}{reveal_blurb}  {dt:.1f}s"
        )

    print(
        f"\nDone: {args.n - len(failures)}/{args.n}.  "
        f"Ready-parse: {parse_ok_count}/{args.n}.  "
        f"Reveal-parsed: {reveal_hits}/{args.n}.  "
        f"Failures: {len(failures)}."
    )
    for f in failures:
        print(f"  {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
