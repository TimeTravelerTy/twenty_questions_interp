# M3 — self-chosen Ready-state smoke (4B)

**Run date:** 2026-04-19 by Claude (Opus 4.7).
**Model:** `google/gemma-3-4b-it` on TSUBAME H100, bfloat16.
**Job:** `tq_m3_selfchosen_ready` (`7218660`). Repo HEAD at run time: `0a203f5`.
**Artifacts:** `runs/diag/selfchosen_ready_smoke_20260419/` (local + TSUBAME).

## Scope

The D-21/D-22 follow-through: does the self-chosen Ready-state geometry at 4B
look like persistence State A (clean, probe-ready) or State B (decodable but
rotated / amplitude-collapsed)? Same 4 candidates (`tiger, eagle, frog,
salmon`) and primary question set (`is_mammal, is_bird,
lives_primarily_in_water, has_four_legs`) as `diagnose_persistence.py`, so
results are directly comparable. Labels come from a post-question reveal.

## Topline finding 1 — choice distribution collapses to 2/4 classes

| reveal | count (of 40 attempts) | position-0 bias |
|---|---|---|
| **salmon** | **33** | 15/15 when salmon is first; 7/11 when tiger first; 11/11 when eagle first |
| **frog**   | **7**  | 5/5 when frog is first; 2/10 when tiger first |
| tiger | 0 | — |
| eagle | 0 | — |

Greedy decoding + candidate shuffling was insufficient to force variety. The
model shows:

1. A **self-bias toward "unusual" candidates** (salmon and frog, the two
   non-mammal-or-bird items). Tiger and eagle are never chosen.
2. A **first-position lock-in**: when salmon or frog is permuted to position 0,
   the model selects it 100% of the time. Tiger and eagle at position 0 default
   to salmon or frog instead.

This is its own interesting result: under greedy decoding with the current
self-chosen prompt template, "choice" is largely a deterministic function of
permutation × category prior. Not a fair sampling of a latent distribution.

Immediate consequence: the 4-way NC LOO + A-vs-B vote the script was designed
to compute is ill-posed (only 2 realized classes). We did an ad-hoc 2-class
analysis instead — `runs/diag/selfchosen_ready_smoke_20260419/analysis_2class.json`.

## Topline finding 2 — self-chosen Ready is even more collapsed than State B

Salmon subsampled to 7 (matching frog count) for a balanced 14-run
NC LOO + within-between contrast. Persistence rows restricted to the same 2
classes (2 each, untouched).

| metric | self-chosen Ready | State A | State B |
|---|---|---|---|
| post-13 best NC LOO | **100% @ L29** | 100% @ L21 | 100% @ L13+ |
| NC LOO at L21 (A's peak) | **64%** | 100% | 100% |
| post-13 within-between contrast | **+7.85e-05** | +1.31e-02 | +5.26e-04 |
| ratio vs A | 1× | **166×** | 25× |
| ratio vs B | 1× | 25× | **6.7×** |

Per-layer contrast at selected depths:

| layer | self-chosen | A | B |
|---|---|---|---|
| L17 | +5.96e-08 | +9.94e-05 | +2.33e-05 |
| L21 | +1.25e-06 | +9.46e-04 | +3.17e-05 |
| L25 | +3.98e-05 | +1.00e-02 | +5.22e-05 |
| L29 | +4.08e-05 | +1.13e-02 | +5.94e-05 |
| L33 | +9.71e-05 | +1.49e-02 | +7.44e-04 |

The entity-identity signal is present at self-chosen Ready (NC hits 100% at
L29) but the within-vs-between cosine geometry is an additional ~6.7× weaker
than State B across post-13 layers, and **identity does not become
NC-decodable until the very final block** (L29), not at the mid layers where
persistence A and B both peak.

This refutes the friendlier hypothesis from D-21 (that self-chosen *might*
look like A). Self-chosen Ready is the **weakest** regime of the three:

```
  A (post-verbalization) > B (pre-Ready after lock-in) > self-chosen Ready
        1×                        1/25×                       1/166×
```

## Revised picture for the scientific claim

The previous working story — "the model fluently represents its secret at
Ready" — does not survive this data. The sharpened story is:

> At Gemma 3 4B, in the self-chosen condition, the model does hold a latent
> identity for its chosen candidate, but that latent is **only decodable at
> the final mid-network / late-network layers (≥ L29 in this smoke)** and is
> **geometrically much weaker** than even the already-attenuated lock-in
> state (B). Attribute probes trained at calibration positions that resemble
> State A or State B would transfer poorly.

This matters for the M4 pipeline plan: probes should be fit at self-chosen
Ready directly, not transferred from elsewhere in the dialogue. And the M5
SAE-feature case studies need to locate features at layers ≥ L29 if they want
to capture self-chosen identity.

## Caveats that bound the claim

- **n = 2 per candidate on the persistence side, 7 per class on self-chosen
  after balancing**. NC LOO and contrast are coarse at these sample sizes.
  A larger self-chosen replication (below) is the obvious next step.
- **float32 vs bfloat16.** Persistence was float32; self-chosen was bfloat16.
  The 166× contrast gap is too large to be explained by dtype alone, but the
  comparison should be replicated at matched dtype if it becomes
  load-bearing.
- **2-class binary** is easier than 4-way. The fact that NC reaches 100% at
  L29 for salmon-vs-frog is a weaker claim than 4-way 100% separability.
- **Position bias + category bias** in choice distribution means "self-chosen"
  here is closer to "biased-forced-choice" than to sampling from an internal
  distribution. A temperature-sampled follow-up would disentangle this.
- **Single condition (4-candidate prompt).** Whether self-chosen at the full
  20-candidate prompt is *more* or *less* collapsed is not known from this
  smoke.

## What this does NOT conclude

- Does not conclude that self-chosen is uninterpretable — identity IS
  decodable, just at late layers with weak amplitude.
- Does not close D-05 (bigger-model ladder). At 12B+ the self-chosen
  distribution may widen and the geometry may be cleaner; the rotation/
  collapse shape is explicitly model-scale-dependent in the 4B story.
- Does not close the bank audit backlog; correctness on the realized classes
  (salmon, frog) still inherits the disputed `frog.has_four_legs` cell.

## Decision

D-23 (this session): pipeline plan updated — probes fit at self-chosen Ready,
not transferred; SAE case studies target L29+. Follow-up run proposed: larger
self-chosen smoke with (a) temperature sampling to break the greedy collapse
and/or (b) expanded candidate list, so NC LOO can use 4+ realized classes.
