# M3 — 12B name_paraphrase pilot: Ready-state readouts

**Run date:** 2026-04-21 by Claude (Opus 4.7).
**Model:** `google/gemma-3-12b-it`, H100 bfloat16.
**Calibration pilot:** job `7226576`, 100/100 `Ready` parsed, no failures.
**Artifacts:** `runs/calibration/12b_name_paraphrase_4way_pilot_20260420/` (100 runs);
`runs/m3_12b_pilot_readouts.json`; this file.

## Scope

First dense Ready-state readouts on a 12B name_paraphrase calibration pilot
(25 runs × {elephant, cow, dog, horse} = 100 runs) using the discriminative
6-question panel validated in the matched-controls run `7226547`
(`is_carnivore, is_larger_than_human, is_domesticated, lives_in_africa,`
`produces_dairy_milk, is_ridden_by_humans`).

The goal is purely to check that calibration data at the standardized 12B
schema produces sensible probes — a plumbing check, not the scientific claim.
The scientific test is transfer to self-chosen Ready, covered in a separate
follow-up.

## Result

4-way NC/LR LOO per layer (chance 0.25) at Ready across all 49 hidden states:

| layer band | NC LOO | LR LOO | attribute decoders |
|---|---|---|---|
| 0 (embedding) | 0.25 | 0.00 | 0.75 (majority) |
| 1–5 (early) | 0.23–0.52 | 0.62–0.99 | 0.81–0.99 |
| **6–7** | 0.52–0.66 | **1.00** | 0.93–1.00 |
| 8–16 | 0.59–0.93 | ≥0.99 | ≥0.93 |
| **17–26** | 0.93–0.99 | ≥0.99 | 1.00 |
| **27–48** | **1.00** | **1.00** | **1.00** |

See `M3-12b-pilot-readouts-detail.md` for the full per-layer table (auto-generated
by `scripts/decode_ready.py`).

Topline:

- **Linear probe saturates by L6.** LR LOO hits 1.00 at L6 and stays there.
  Candidate identity is *linearly* available in the residual stream from
  roughly the first quarter of the network onward.
- **Nearest-centroid catches up by L17 and saturates by L27.** NC LOO climbs
  0.25 → 0.66 @ L7 → 0.93 @ L16 → 0.98 @ L17 → 1.00 from L27. The LR/NC gap
  over L6–L26 is the interesting structural finding below.
- **Binary attribute decoders saturate at L7.** All six questions in the panel
  are LOO-perfect from L7 onward. This is a plumbing confirmation at the pilot
  size (4 classes × bank lookup → attribute is a deterministic function of
  class, so any perfect class decoder forces this); it does not independently
  validate attribute decoding.

## Interpretation

The LR ≫ NC gap over early-to-mid layers is the non-trivial structural
signal. At L6–L26 a linear separator works perfectly while centroid distance
misclassifies 2–40% of held-out runs. That implies the four candidates live
on a low-dimensional separable surface but with non-spherical / non-isotropic
per-class clusters — there are directions along which identity is encoded
that are not well-captured by the per-class mean. By L27 the residual stream
has tightened into class centroids; by L48 all probes saturate.

For the blog narrative, this is the first clean demonstration that at 12B the
Ready-state residual *does* carry candidate identity, and it does so early
and linearly. 4B could not be made to produce this shape (see
[M3-selfchosen-20bank.md](M3-selfchosen-20bank.md): self-chosen was ~104×
weaker than persistence State B on contrast even at the 20-bank prompt).

## Caveats

1. **Saturation makes differential statements hard.** At 4 classes × 25 runs,
   identity is easy. Scaling to a broader candidate set (e.g. the full 20-bank)
   is required to differentiate where linear separability breaks.
2. **Calibration is not the scientific claim.** These probes are trained on
   `name_paraphrase` calibration where the secret is literally named in the
   prompt. Per D-06/D-23 the headline claim requires transfer to self-chosen
   Ready, not calibration LOO.
3. **Attribute decoders are downstream of class.** At 4 classes × 25 runs, a
   perfect class decoder trivially implies perfect attribute decoders by bank
   lookup. Attribute decoding becomes informative only with many more classes
   or when class identity is imperfectly decodable.

## Secondary: fixed a class-set bug in `decode_ready.py`

The original implementation passed `bank.candidate_ids` (all 20) to the
nearest-centroid fit regardless of which classes the calibration data
actually contained. For any subset pilot, centroids for the 16 unused
classes collapsed to NaN and poisoned every argmin, so NC LOO read 0.00
across all layers on the first run of this pilot. Fixed in `decode_ready.py`
to use `sorted(set(cal_secrets))`. M2 (all 20 classes) is unaffected.

## Next step

Scientific: fit readouts on the calibration pilot at the L6–L17 band, then
collect a matching 12B self-chosen Ready run on `{elephant, cow, dog, horse}`
and test whether the trained centroid/LR probe *transfers* to self-chosen
predictions. That is the D-23 reinforcement: probes must be fit at the
position they're evaluated at, and we need to measure how bad calibration-→-
self-chosen transfer actually is at 12B. This is the point of the whole
M3 → M4 arc.
