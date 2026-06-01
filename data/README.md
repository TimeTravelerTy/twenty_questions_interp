# data/

Canonical artifacts for the study. Every `RunManifest` pins the model and tokenizer
revisions it was built against, but assumes the files in this directory are stable
*within* a manifest's lifetime. Changing a file here invalidates prior runs, so bump
a version string when you do.

## Files

- `animals.yaml` — 20-candidate bank.
- `questions.yaml` — ~28 binary predicates.
- `answers.csv` — `A(c, q)` table, manually verified.

## Build process

1. Draft animals and questions for attribute diversity.
2. Fill `answers.csv` with best-effort labels.
3. Manual review of edge cases (e.g., orca = mammal, penguin does not fly).
4. `python scripts/validate_answers.py` — enforces no NaNs, question balance
   (1..19 yeses; excludes only all-yes/all-no predicates), pairwise
   distinguishability ≥ 2, and a per-question entropy report.
