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
4. **Mid-dialogue pre-answer positions are the next thing to test.** An
   earlier draft of this section suggested State A / State B — that was a
   category error. State A / State B only exist when the model puts the
   secret name into context, which self-chosen forbids by construction
   (see D-29). Within the self-chosen regime, the remaining lever on
   *position* is not post-verbalization but mid-dialogue: by the time the
   model has answered a few yes/no questions, the commitment has been
   exercised and the chosen class may be more crystallized than at Ready.
   `diagnose_selfchosen_ready.py` already captures
   `turn_0k_activations.pt` for k=1..4 at each pre-answer position, so
   this test can run directly on the 40 kept runs without new collection.

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

1. **Sweep mid-dialogue pre-answer positions on the existing data.**
   `turn_01..turn_04_activations.pt` are already captured per kept run.
   Rerun LOO NC + LR at each of those positions and compare against the
   Ready baseline in this doc. Real signal would look like a layer band
   that is consistent across turns and sharpens with turn number; the
   Ready pattern (isolated single-layer peaks, NC and LR peaks at
   different layers) would look like noise. No new collection needed.
2. **If mid-dialogue is also weak, shift to class diversity, not more
   positions.** The narrow realized class set `{elephant,cow,dog,horse}`
   is the most likely next bottleneck — all mammals, mostly domesticated.
   Forcing realization diversity (T>0, seed sweep, prompt variants) is
   cheaper than trying stronger regularization or PCA on 40 points.

Explicitly *not* on the path: probing "State A / State B" in the
self-chosen condition. Those positions were defined (D-21) as post-name-
verbalization residual-stream states in the *persistence* regime. Self-
chosen by construction never names the secret in context, so there is no
analogous position to probe. A prior revision of this section proposed
exactly that experiment; it was incoherent.

## Artifacts

- `docs/progress/M3-12b-selfchosen-direct-detail.md` — full per-layer LOO
  table for NC and LR across all 49 layers.
- `runs/m3_12b_selfchosen_directfit.json` — machine-readable report.
- Collection: `runs/diag/selfchosen_ready_20bank_12b_directfit_20260421/`
  (40 kept runs pulled locally; full 300-attempt dir lives on TSUBAME at
  `/gs/fs/tga-sip_arase/tyrone/twenty_questions_interp/`).
