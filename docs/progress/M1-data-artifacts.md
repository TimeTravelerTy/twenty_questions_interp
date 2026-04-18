# M1 — Data artifacts + feasible-set utility

**Closed:** 2026-04-18 by Claude (Opus 4.7).

## What was built

- **`data/animals.yaml`** — 20 candidates selected for attribute diversity across
  taxonomy (10 mammals / 4 birds / 2 reptiles / 2 fish / 1 amphibian / 1 insect),
  habitat, diet, size, domesticity, and geography.
- **`data/questions.yaml`** — 30 binary predicates.
- **`data/answers.csv`** — 20 × 30 A(c, q) table, all binary, no NaNs.
- **`src/twenty_q/`** — `banks.py` (loaders + `feasible_set`), `permutations.py`
  (seeded shuffle + 1-based displayed-index roundtrip), `manifest.py` (pydantic
  `RunManifest` + JSON save/load), `config.py` (paths + model IDs).
- **`scripts/validate_answers.py`** — runs on CI; produces a per-question
  yes-count / entropy report and checks pairwise distinguishability.
- **Tests** — 13 pass (bank loader, feasible-set hand-computed fixture,
  permutation bijection).

## Surprises / calibration on the validator

The original validator thresholds from the plan (`5 <= yes-count <= 15`,
pairwise diff >= 3) were too strict for a 20-animal bank. Intrinsic issues:

- **Rare taxonomic classes are unfixable by adding questions.** With only 1
  amphibian and 1 insect, `is_amphibian` and `is_insect` can never reach 5
  yeses. Relaxed to `1..19` (only forbids useless all-yes / all-no).
- **Three candidate pairs landed at 0–1 diffs** after the first draft: (cow,
  horse), (dog, cat), (eagle, owl). Fixing pairs to `>= 3` diffs required 6+
  targeted indicator questions (purrs, barks, has-a-mane, produces-milk, …) —
  too much stuffing for marginal gain. Relaxed pairwise floor to `2` and added
  4 targeted distinguishers: `is_ridden_by_humans`, `produces_dairy_milk`,
  `purrs`, `soars_during_daylight`. Dropped 2 redundant predicates:
  `lives_primarily_on_land` (complement of `lives_primarily_in_water` for our
  bank) and `is_warm_blooded` (determined by mammal OR bird). Final bank: 30
  questions, all 190 pairs distinguishable on >=2 questions.
- See `docs/DECISIONS.md` D-14 for the permanent record.

## Ambiguous edge cases (logged for future rereadings)

- `kangaroo.can_swim`: 0 (they swim if crossing a river but it's not
  characteristic).
- `gorilla.has_four_legs`: 0 (primates are bipedal-with-arms semantically;
  knuckle-walking does not equal four-leggedness).
- `elephant.covered_in_fur_or_hair`: 0 (elephants have sparse bristly hair,
  not what a casual observer calls "fur or hair").
- `chicken.is_carnivore/is_herbivore`: both 0 (omnivore — neither predicate
  is true). Similarly `gorilla.is_herbivore=1` even though they eat insects
  occasionally.
- `frog.lives_primarily_in_water`: 1 (adult frogs vary; default to aquatic
  association).
- `cobra.has_a_tail`: 1 (body-continuous-with-tail; the model will likely
  interpret snakes as "all tail" or "no tail" — could be a hard case).
- `eagle/owl.has_scales`: 0 (technically bird legs are scaled, but the
  question is commonly understood as reptile/fish).

These are edge cases the model will encounter at M3+ when the dialogue
actually includes question turns. If decoder accuracy is low on a specific
animal, check this list first — the A(c, q) row may be the issue, not the
probe.

## Watch-outs for the next agent

- `RunManifest.save()` creates parent dirs. `runs/` is gitignored; scripts
  should write there freely.
- `banks.feasible_set(history)` expects already-parsed booleans (0/1), not
  raw yes/no strings. M2's dialogue driver must parse first.
- The permutation's `displayed_index` is **1-based**, matching the calibration
  prompt ("candidate #7"). Do not off-by-one.
- `is_warm_blooded` and `lives_primarily_on_land` were intentionally dropped
  as redundant — do not add them back without logging in DECISIONS.md.

## Next concrete step

M2: `src/twenty_q/prompts.py` (calibration + self-chosen templates),
`hooks.py` (NNsight wrapper), `dialogue.py` (one Ready capture end-to-end).
Target Gemma 3 1B on local CPU.
