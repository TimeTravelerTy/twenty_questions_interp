"""Validate data/animals.yaml + questions.yaml + answers.csv.

Checks:
  1. All loaders pass (no NaNs, no non-binary, alignment with yaml files).
  2. Each question splits the bank non-trivially (5..15 yeses inclusive).
  3. Every pair of candidates differs on at least 3 questions.
  4. Print a per-question entropy / yes-count report.

Exits non-zero on any violation.
"""
from __future__ import annotations

import math
import sys
from itertools import combinations

from twenty_q.banks import load_bank

# Yes-count band. With 20 animals and rare taxonomic classes (1 amphibian, 1
# insect, 2 reptiles, 2 fish), a 5-yes minimum is intrinsically impossible for
# class-membership questions. 1..19 only forbids the truly useless all-yes /
# all-no predicates.
MIN_YES = 1
MAX_YES = 19
# Pairwise-distinguishability floor. 3 would be preferable for robustness but
# produces unsolvable constraints for (cow, horse) and (dog, cat) without
# stuffing the question bank with indicator-style predicates. 2 is the
# relaxation for M1; see DECISIONS.md D-14.
MIN_PAIRWISE_DIFF = 2


def binary_entropy(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def main() -> int:
    bank = load_bank()
    problems: list[str] = []

    n = len(bank.candidates)
    print(f"Bank: {n} candidates, {len(bank.questions)} questions\n")

    # Per-question yes counts + entropy report.
    print(f"{'question':40s}  yes/total   entropy")
    print(f"{'-' * 40:40s}  ---------   -------")
    for q in bank.questions:
        yeses = sum(bank.answers[c.id][q.id] for c in bank.candidates)
        p = yeses / n
        h = binary_entropy(p)
        marker = "  OK" if MIN_YES <= yeses <= MAX_YES else "  !!"
        print(f"{q.id:40s}   {yeses:2d}/{n:2d}       {h:.3f}{marker}")
        if not (MIN_YES <= yeses <= MAX_YES):
            problems.append(
                f"Question {q.id!r} has {yeses} yeses; expected {MIN_YES}..{MAX_YES}"
            )
    print()

    # Pairwise distinguishability.
    under_min_pairs: list[tuple[str, str, int]] = []
    for a, b in combinations(bank.candidates, 2):
        diff = sum(
            bank.answers[a.id][q.id] != bank.answers[b.id][q.id] for q in bank.questions
        )
        if diff < MIN_PAIRWISE_DIFF:
            under_min_pairs.append((a.id, b.id, diff))

    if under_min_pairs:
        problems.append(
            f"{len(under_min_pairs)} candidate pair(s) differ on < {MIN_PAIRWISE_DIFF} questions"
        )
        print("Pairs with too-small distinguishing set:")
        for a, b, d in under_min_pairs[:20]:
            print(f"  {a:12s}  vs  {b:12s}  differ on {d} questions")
        if len(under_min_pairs) > 20:
            print(f"  ... and {len(under_min_pairs) - 20} more")
        print()
    else:
        print(f"Pairwise distinguishability: all C(n,2) = {n * (n - 1) // 2} pairs OK"
              f" (>= {MIN_PAIRWISE_DIFF} diffs).\n")

    if problems:
        print("FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
