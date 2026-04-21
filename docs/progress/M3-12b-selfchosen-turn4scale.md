# M3 — 12B self-chosen turn-4 pre-answer: scale-up crystallizes the signal

**Run date:** 2026-04-21 by Claude (Opus 4.7).
**Source collection:** `runs/diag/selfchosen_ready_20bank_12b_turn4scale_20260421/`
from job `7232075` (600 attempts, 80 kept = 20 × `{elephant,cow,dog,horse}`).
**Scored positions:** pre-answer activations at turns 1 and 4, all 49 layers.
**Machine-readable report:** `runs/m3_12b_selfchosen_turn4scale.json`.
**Detail table:** `docs/progress/M3-12b-selfchosen-turn4scale-detail.md`.

## Scope

Follow-up to `M3-12b-selfchosen-turns.md`, which showed that on the n=40 pilot
from job `7230807`, turn-4 pre-answer was the first self-chosen position at
12B where NC/LR LOO rose materially above chance (NC max 0.625, LR max 0.60;
L27–48 means NC 0.549, LR 0.539). The pilot was right at the edge of
"probe-usable", so the question was whether 2× more data would push the
signal into robust territory or expose it as noise.

This scale-up doubles the per-class count to 20 (total 80 kept runs) and
keeps every other parameter identical: same 20-bank self-chosen prompt,
same 4-question panel, same model (`google/gemma-3-12b-it`), T=0.0.

## Result

Chance is **25%** (4 realized classes: `elephant, cow, dog, horse`).
LOO summaries across all non-L0 layers:

| turn | NC mean | NC median | NC max | LR mean | LR median | LR max |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.342 | 0.388 | 0.500 | 0.349 | 0.350 | 0.487 |
| 4 | **0.428** | **0.500** | **0.662** | **0.509** | **0.562** | **0.787** |

Layers L27–48 (the late band already identified in the n=40 pilot):

| turn | NC L27–48 mean | LR L27–48 mean |
|---|---:|---:|
| 1 | 0.418 | 0.431 |
| 4 | **0.558** | **0.731** |

Turn-4 peaks (for reference; best layers stable across NC and LR):

- NC max **0.662 @ L29**
- LR max **0.787 @ L31**
- Second peak: LR 0.76 @ L29 and 0.76 @ L47

## Comparison to the n=40 pilot

| metric | pilot (n=40) | scale-up (n=80) | delta |
|---|---:|---:|---:|
| turn-4 NC L27–48 mean | 0.549 | 0.558 | +0.009 |
| turn-4 LR L27–48 mean | 0.539 | **0.731** | **+0.192** |
| turn-4 NC max | 0.625 | 0.662 | +0.037 |
| turn-4 LR max | 0.600 | **0.787** | **+0.187** |

NC barely moves. LR jumps ~0.19 in both mean and max. That pattern is
exactly what we would expect if the class signal is linearly separable but
the LR fit at n=40 was regularization-starved on 3840-dim features. The
scale-up does not create a signal; it sharpens one that was already
present in the pilot.

## Interpretation

1. **Turn-4 pre-answer is the self-chosen probe position at 12B.** LR 0.79
   at L31 is ~3.2× chance. The STATUS threshold for locking this position
   ("rises into the ~70% regime") is cleared on LR and exceeded on the
   whole L27–L48 band (LR mean 0.731).
2. **The signal is coherent across depth, not a single-layer artifact.**
   LR is 0.68–0.79 across L26–L48; NC is 0.50–0.66 across the same band.
   That rules out multiple-testing noise as the explanation.
3. **Turn 1 is meaningful but much weaker.** Turn-1 L27–48 LR mean is
   0.431, clearly above chance but far below turn-4's 0.731. So the
   commitment-strengthening story is real: answering yes/no questions
   does sharpen the latent class representation.
4. **Public history still cannot explain this.** The 4-question panel is
   degenerate (`1,0,0,1`) for every realized class in this subset, so the
   late-layer decode at turn 4 cannot be leakage from public yes/no
   history. This is a latent-state result.
5. **LR ≫ NC at turn 4 (late band 0.73 vs 0.56)** mirrors the 12B
   calibration pattern from `M3-12b-pilot-readouts.md`: classes live on a
   linearly separable surface before their centroids become spherical.

## Decision consequence

1. **Lock turn-4 pre-answer, late layers (L26–L48, peaks near L29–L31) as
   the self-chosen probe position for M4.** Causal patching, SAE feature
   case studies, and the blog-post readout all use this position.
2. **Do not keep sweeping alternative self-chosen positions** unless a
   specific causal-patching experiment needs a different residual index.
3. **The next bottleneck is class diversity, not position.** Every 12B
   self-chosen collection realizes only `{elephant, cow, dog, horse}`.
   Whether LR 0.79 at L31 survives when we force broader realization
   (temperature > 0, prompt variants) is the next question. But the
   position question is now resolved.

## Artifacts

- `runs/m3_12b_selfchosen_turn4scale.json`
- `docs/progress/M3-12b-selfchosen-turn4scale-detail.md`
- Source runs (local subset): `runs/diag/selfchosen_ready_20bank_12b_turn4scale_20260421/`
  (80 kept runs, Ready + turn_01 + turn_04 activations only; full
  per-turn tensors remain on TSUBAME)
