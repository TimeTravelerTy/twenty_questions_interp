# M3 — 12B self-chosen Ready direct-fit is also weak

**Run date:** 2026-04-21 by Claude (Opus 4.7).
**Model:** `google/gemma-3-12b-it`, H100, bfloat16.
**Collection:** job `7230807`, 300 attempts, 40 kept after balancing to 10/class.
**Artifacts:** `runs/diag/selfchosen_ready_20bank_12b_directfit_20260421/`,
`runs/m3_12b_selfchosen_directfit.json`, per-layer table in
`docs/progress/M3-12b-selfchosen-direct-detail.md`.

## Scope

After the `M3-12b-selfchosen-transfer.md` verdict (calibration→self-chosen
transfer is at chance at 12B, D-28), the decisive follow-up is: does fitting
probes *directly* at self-chosen Ready recover the signal? If direct-fit is
strong, the bottleneck is just probe-training location (D-23) and the path
forward is self-chosen-trained probes. If direct-fit is also weak, the
bottleneck is self-chosen Ready geometry itself, and the H-rotation story
from 4B (D-21) — where State B carries the class code but Ready does not —
extends upward.

Collection: 20-bank prompt, T=0.0, 300 attempts, quota 10/class.
Realized classes: **`{elephant, cow, dog, horse}`** (same as every previous
12B self-chosen run). 40 kept runs balanced 10/class.

Scoring: leave-one-run-out NC and LR at every layer on the kept 40 runs,
class set restricted to the realized 4.

## Result

Per-class chance is **25%**. Summary across all 49 layers (L0 embedding
excluded from mean/median):

|            | mean | median | max             |
|------------|------|--------|-----------------|
| **NC LOO** | 0.23 | 0.22   | **0.45 @ L14**  |
| **LR LOO** | 0.27 | 0.28   | **0.45 @ L4**   |

Best layers (top 5 by NC, then by LR):

- NC: L14 0.45 · L25 0.42 · L17 0.40 · L44 0.38 · L45 0.38
- LR: L4 0.45 · L18 0.42 · L16 0.38 · L5 0.35 · L19 0.35

## Interpretation

1. **Direct-fit at self-chosen Ready is barely above chance.** The mean LOO
   accuracy over layers is at chance for NC and only marginally above for
   LR. Individual-layer maxima reach ~1.8× chance, but the best-layer
   positions for NC and LR disagree (L14 vs L4), which is consistent with
   these being noise peaks rather than a coherent mid-network class code.
2. **Compare to 12B *calibration* Ready on the same 4 classes.** LR LOO is
   **1.00 from L6** and NC LOO is **1.00 from L27**
   (`M3-12b-pilot-readouts.md`). Calibration saturates; self-chosen does not
   even approach it.
3. **This is the decisive negative result for the transfer-is-the-bottleneck
   hypothesis.** D-23 told us probes must be fit *where* they're evaluated.
   Doing exactly that at self-chosen Ready on 12B does not produce a strong
   readout. So the issue is not just "calibration geometry is different from
   self-chosen geometry"; the issue is that **self-chosen Ready at 12B, on
   the currently realized class distribution, does not carry a reliably
   decodable class signal**.
4. **The H-rotation story from D-21 is now the leading candidate
   again at 12B.** At 4B, self-chosen Ready was 104× weaker than
   matched persistence State B on contrast, but the same prompt at
   State A/B carried clear class structure. If 12B behaves the same way,
   Ready is simply the wrong position to probe for self-chosen identity —
   we need State A or State B.

## Caveats and what this does *not* prove

- **n = 40 over 4 classes is small.** At 3840-dim residual stream, LR is
  severely underdetermined even with default L2 regularization. A stronger
  regularizer, PCA first, or more runs per class could lift LR. NC is less
  sensitive to dimensionality but also noisy at 10 samples/class.
- **Only 4 of 20 classes realize.** The same four have now shown up in
  every 12B self-chosen run: `{elephant, cow, dog, horse}`. They are all
  mammals and mostly domesticated land animals — semantically tight.
  Broader realization might improve separability, but narrowing to this
  subset is not a fair representation of the 20-way task.
- **Per-layer maxima are not disciplined.** Because we picked the peak
  across 49 layers, the ~45% numbers are capped by layer-sweep multiple
  testing. A pre-registered layer from calibration (say L6 for LR, L27 for
  NC) gives NC=0.23 at L27 and LR=0.33 at L6 — both at or near chance.

## Decision consequence

Direct-fit at self-chosen Ready is not probe-ready at 12B. The scientific
path forward splits:

1. **Shift the probing position, not the training regime.** Rather than
   keep chasing Ready-state self-chosen readouts, capture self-chosen
   **State A / State B** activations at 12B and repeat the geometry
   analysis. This is the H-rotation replication test: D-21 said at 4B,
   Ready ≪ State B; does that still hold at 12B?
2. **Improve self-chosen sample size / diversity only if (1) is strong.**
   Scaling to 20+ runs/class or broadening the realized distribution is
   only worthwhile if we can first show the geometry is there somewhere.

## Artifacts

- `docs/progress/M3-12b-selfchosen-direct-detail.md` — full per-layer LOO
  table for NC and LR across all 49 layers.
- `runs/m3_12b_selfchosen_directfit.json` — machine-readable report.
- Collection: `runs/diag/selfchosen_ready_20bank_12b_directfit_20260421/`
  (40 kept runs pulled locally; full 300-attempt dir lives on TSUBAME at
  `/gs/fs/tga-sip_arase/tyrone/twenty_questions_interp/`).
