"""Batch-run calibration dialogues.

Defaults to 8 runs per candidate (160 total) on Gemma 3 1B / CPU. Each run
captures per-layer residual-stream activations at the Ready position plus a
`RunManifest` under `runs/calibration/<run_id>/`. Pass `--question-ids` to
continue past Ready and capture pre-answer activations on later turns.

Usage:
    uv run python scripts/run_calibration.py --n-per-candidate 8
    uv run python scripts/run_calibration.py --n-per-candidate 1 --candidates tiger,cat
    uv run python scripts/run_calibration.py --n-per-candidate 1 \
        --candidates tiger --question-ids is_mammal,can_swim
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time

import torch

from twenty_q.banks import load_bank
from pathlib import Path

from twenty_q.config import CALIBRATION_RUNS_DIR, MODEL_DEBUG
from twenty_q.dialogue import load_model, run_calibration_dialogue
from twenty_q.permutations import shuffle_candidates


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL_DEBUG)
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--n-per-candidate", type=int, default=8)
    p.add_argument(
        "--out-dir",
        default=str(CALIBRATION_RUNS_DIR),
        help="Directory to write run subdirectories into.",
    )
    p.add_argument(
        "--schema",
        default="index",
        choices=["index", "name_paraphrase"],
        help="Calibration schema for Ready-state collection.",
    )
    p.add_argument("--candidates", default="",
                   help="Comma-separated candidate ids to restrict to; empty = all 20.")
    p.add_argument("--question-ids", default="",
                   help="Comma-separated question ids to ask after Ready; empty = Ready-only.")
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bank = load_bank()

    target_ids = (
        tuple(x.strip() for x in args.candidates.split(",") if x.strip())
        if args.candidates
        else bank.candidate_ids
    )
    unknown = set(target_ids) - set(bank.candidate_ids)
    if unknown:
        print(f"Unknown candidate ids: {sorted(unknown)}", file=sys.stderr)
        return 2
    question_lookup = {q.id: q for q in bank.questions}
    question_ids = [x.strip() for x in args.question_ids.split(",") if x.strip()]
    unknown_questions = sorted(set(question_ids) - set(question_lookup))
    if unknown_questions:
        print(f"Unknown question ids: {unknown_questions}", file=sys.stderr)
        return 2
    questions = [question_lookup[qid] for qid in question_ids]

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

    total = len(target_ids) * args.n_per_candidate
    done = 0
    failures: list[str] = []
    for cid in target_ids:
        for k in range(args.n_per_candidate):
            # hashlib (not built-in hash) — process-stable, reproducible across runs.
            digest = hashlib.sha256(f"{cid}:{k}".encode()).digest()
            seed = (args.seed_offset + int.from_bytes(digest[:4], "big")) % (2**31)
            perm = shuffle_candidates(bank.candidate_ids, seed=seed)
            run_id = f"cal_{cid}_{k:02d}"
            t0 = time.time()
            try:
                manifest = run_calibration_dialogue(
                    handle=handle,
                    bank=bank,
                    secret_canonical_id=cid,
                    perm=perm,
                    seed=seed,
                    run_id=run_id,
                    out_dir=out_dir,
                    questions=questions or None,
                    schema=args.schema,
                )
            except Exception as e:  # noqa: BLE001
                failures.append(f"{run_id}: {type(e).__name__}: {e}")
                print(f"  [{done+1}/{total}] {run_id}  FAILED: {e}")
                done += 1
                continue
            ok = "ok" if manifest.ready_parse_ok else "parse!"
            dt = time.time() - t0
            done += 1
            print(
                f"  [{done}/{total}] {run_id}  "
                f"ready={manifest.ready_raw_output!r} ({ok})  {dt:.1f}s"
            )

    print(f"\nDone: {done}/{total}. Failures: {len(failures)}.")
    for f in failures:
        print(f"  {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
