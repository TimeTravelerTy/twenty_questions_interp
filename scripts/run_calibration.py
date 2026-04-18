"""Batch-run calibration dialogues.

Defaults to 8 runs per candidate (160 total) on Gemma 3 1B / CPU. Each run
captures per-layer residual-stream activations at the Ready position plus a
`RunManifest` under `runs/calibration/<run_id>/`.

Usage:
    uv run python scripts/run_calibration.py --n-per-candidate 8
    uv run python scripts/run_calibration.py --n-per-candidate 1 --candidates tiger,cat
"""
from __future__ import annotations

import argparse
import sys
import time

from twenty_q.banks import load_bank
from twenty_q.config import CALIBRATION_RUNS_DIR, MODEL_DEBUG
from twenty_q.dialogue import load_model, run_calibration_dialogue
from twenty_q.permutations import shuffle_candidates


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL_DEBUG)
    p.add_argument("--n-per-candidate", type=int, default=8)
    p.add_argument("--candidates", default="",
                   help="Comma-separated candidate ids to restrict to; empty = all 20.")
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

    print(f"Loading {args.model} on {args.device} ...")
    t0 = time.time()
    handle = load_model(args.model, device=args.device)
    print(f"  loaded in {time.time() - t0:.1f}s")

    total = len(target_ids) * args.n_per_candidate
    done = 0
    failures: list[str] = []
    for cid in target_ids:
        for k in range(args.n_per_candidate):
            seed = args.seed_offset + hash((cid, k)) % (2**31)
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
                    out_dir=CALIBRATION_RUNS_DIR,
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
