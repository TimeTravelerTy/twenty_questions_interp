# data/

Canonical artifacts for the study. Every `RunManifest` pins the model + tokenizer
revisions it was built against, but assumes the files in this directory are stable
*within* a manifest's lifetime. Changing a file here invalidates prior runs — log
the change in `docs/DECISIONS.md` and bump a version string.

## Files

- `animals.yaml` — 20-candidate bank (M1.1a).
- `questions.yaml` — ~28 binary predicates (M1.1b).
- `answers.csv` — `A(c, q)` table, manually verified (M1.1c).

## Build process

1. Draft animals + questions for attribute diversity (see M1 in plan).
2. Fill `answers.csv` with best-effort labels.
3. Manual review of edge cases (e.g., orca = mammal, penguin ≠ flies).
4. `python scripts/validate_answers.py` — enforces no NaNs, question balance
   (1..19 yeses; excludes only all-yes/all-no predicates), pairwise
   distinguishability ≥ 2 (see DECISIONS.md D-14), per-question entropy report.
