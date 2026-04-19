# M3 — H-persistence diagnostic (verbalized_index, 4B)

**Run date:** 2026-04-19 by Claude (Opus 4.7).
**Model:** `google/gemma-3-4b-it` on TSUBAME H100, float32.
**Job:** `tq_m3_persistence` (`7218322`). Repo HEAD at run time: `f77bc6c`.
**Artifacts:** `runs/diag/persistence_smoke_20260419/` (local + TSUBAME).

## Scope

Direct test of the H-persistence hypothesis proposed in
`M3-3cond-binding-smoke.md` and D-19: *does the instantiated-entity
representation survive the chat-turn boundary at 4B, even when index→name
retrieval succeeds?*

Per run of `verbalized_index` (4 candidates × 2 seeds = 8 runs), we capture
two all-layer hidden states at parallel "model about to speak" positions:

- **State A** — end of turn-1 user prompt, before the model verbalizes the
  name. Immediately after index→name retrieval.
- **State B** — end of turn-2 user prompt, before the model emits `Ready`.
  A full assistant (name) + user (lock-in) turn has elapsed since A.

Both captures use `add_generation_prompt=True`, so they are directly
comparable. We then run the primary question set after Ready to confirm the
usual answer-drift pattern on this run.

Metrics per layer ℓ:

- NC LOO accuracy at A (chance 25%).
- NC LOO accuracy at B.
- **Cross-state transfer:** fit centroids on A, predict B. This is the
  persistence test — if A and B encode the entity in the same way, cross
  accuracy should match within-state accuracy.
- Within-vs-between cosine contrast at A and B separately (signal amplitude).

## Topline results

| metric | value |
|---|---|
| verbalization match (retrieval) | **8/8** |
| primary correctness | **23/32 = 71.9%** (reproducing 3cond smoke) |
| layer 21 NC-A | **100%** |
| layer 21 NC-B | **100%** |
| **layer 21 cross A→B** | **37.5%** |
| post-13 within-between contrast, A | **+1.28e-02** |
| post-13 within-between contrast, B | **+5.17e-04** (~25× weaker) |
| post-13 cross A→B mean | **38.6%** |

Full per-layer numbers live in
`runs/diag/persistence_smoke_20260419/results.json`.

## The shape of the result

The original H-persistence claim — "the entity representation does not
reliably persist across the chat-turn boundary" — is **too strong**. The
entity *identity* does persist: NC LOO is 100% at both A and B from layer 21
onward, and 100% at B from layer 6 onward. The model, at both timepoints,
holds a latent that separates the four candidates into distinct clusters.

What does *not* persist is the **geometry**. Two concrete dissociations:

1. **Cross-state transfer fails in the middle layers.** At layer 21
   (where A first reaches 100% NC LOO), cross A→B is **37.5%** — above
   chance (25%) but far below within-state decoding. Middle-layer
   sub-bands (17–30) all show cross ≤ 75% despite within-state ≥ 88–100%.
   Centroids of the entity learned at A do not classify B.
2. **Amplitude collapses.** The within-vs-between cosine contrast is
   **~25× smaller at B than at A** (1.28e-02 vs 5.17e-04, post-L13 mean).
   The entity signal is still there but nearly drowned in components
   specific to the lock-in / Ready context.

Only at the very final layer (L34, post-norm) does cross transfer reach
100% — i.e., in fully-normalized space the entity directions realign. But
the mid-network layers that matter for attribute readouts and SAE features
(the layers the M4/M5 plan depends on) are where the rotation happens.

## Answer drift corroborates

Primary correctness on the same 8 runs:

| cid | name_match | ready | primary correct |
|---|---|---|---|
| tiger | 2/2 | 2/2 | **8/8** |
| eagle | 2/2 | 2/2 | 6/8 |
| frog | 2/2 | 2/2 | 3/8 |
| salmon | 2/2 | 2/2 | 6/8 |

Wrong-answer inventory (identical to the 3cond smoke):

```
eagle_00/01  is_mammal: Yes (bank No) ✗
frog_00       is_mammal: Yes, is_bird: Yes, has_four_legs: No ✗✗✗
frog_01       is_mammal: Yes, has_four_legs: No ✗✗
salmon_00/01 is_mammal: Yes (bank No) ✗
```

Every non-tiger run calls itself a mammal at least once, even though
retrieval was perfect and the entity is separable at B by NC. This is
what rotation + magnitude collapse predicts: the B-state representation no
longer aligns with the attribute-readout directions that encode
"mammal-ness," so the answer distribution is dominated by generic priors.

## Revised hypothesis — H-rotation/H-collapse

Replacing H-persistence with a sharper, data-consistent version:

> **H-rotation.** At 4B, the entity representation is preserved as
> identity across a chat-turn boundary but undergoes substantial rotation
> and ~25× magnitude collapse in the mid-network layers where attribute
> readouts live. Consequences:
>
> - Identity probes (NC on the candidate list) work at either A or B.
> - Attribute probes trained at one position do not transfer to the other.
> - Answer behavior downstream of B is dominated by candidate-list priors
>   because the rotated/attenuated state no longer projects cleanly onto
>   attribute directions.

This is directly testable: fit a binary `is_mammal` probe at A, apply at B
(expect drop); do the converse (also expect drop). We skipped this in this
run because the cross-state NC result already establishes the rotation
mechanism. Worth doing as a follow-up if it matters for the blog claim.

## Implications for the pipeline

- **Probe-training location matters a lot.** M2's "fit all readouts at the
  Ready position" was the right call, but under M3's turnful dialogue, the
  "Ready position" that survived calibration is one full chat turn removed
  from the point of retrieval. Readouts must be fit where they will be
  evaluated (self-chosen Ready) and not transferred across dialogue
  positions naively.
- **The M4/M5 SAE feature case studies need to pick a capture point and
  stick with it.** Transferring features between positions within a
  dialogue is not free at 4B.
- **Calibration-only accuracy is a lagging indicator of representation
  quality.** At 71.9% primary correctness we still have 100% NC
  separability at B, so a probe-training pipeline built on B-state
  activations can in principle be clean even when the model's overt
  answers are wrong. This is good news for D-01 (calibration is infra):
  imperfect yes/no answers at 4B do not automatically contaminate probe
  inputs.

## Implications for the scientific claim

The blog claim is about self-chosen. The interesting prediction now is:
does self-chosen at 4B look more like A (clean, high-contrast, identity
decodable with big amplitude) or more like B (decodable but
geometrically-collapsed)?

If self-chosen — where the "choice" happens internally without a
verbalization turn — produces something closer to State B, the blog
story needs to center on "the model holds a latent secret that is
decodable but not cleanly actionable." That's still interesting, but a
different shape of claim than "the model fluently represents its secret
as if it were speaking it out loud."

Next step in STATUS will move to the small self-chosen 4B smoke on this
same primary question set to check which regime it resembles.

## What this does NOT conclude

- This result is about 4B. At 12B or larger, competence likely shifts;
  retrieval may survive the boundary with less geometric distortion.
  Worth re-running this exact diagnostic when we move up the ladder
  (D-05).
- The bank's documented disputed cells
  (`frog.has_four_legs`, `eagle.can_swim`; see
  `docs/progress/M3-binding-bank-audit.md`) still account for part of the
  surface correctness miss. A broader bank audit is on the backlog, but
  not prioritized here because the representational finding does not
  depend on exact table correctness — the separability is measured on
  candidate-ID labels, not attribute answers.
- Only `verbalized_index` was tested. Whether name-based conditions also
  show rotation/collapse is unclear; the contrast magnitudes in the 3cond
  smoke are similar (~6.7e-04 for name_paraphrase at B), which at least
  suggests the B-side amplitude collapse is condition-agnostic. A clean
  two-condition re-run with A captured for all conditions would answer
  this, if it becomes load-bearing.

## Decision

H-persistence is replaced by H-rotation (above). The next concrete step
in STATUS moves from "mechanistic persistence test" (done) to the
self-chosen 4B smoke — the remaining item in STATUS step 4.

Calibration remains infrastructure (D-01). Self-chosen is still the
headline (D-01). D-06 remains in effect. No change to the bank.
