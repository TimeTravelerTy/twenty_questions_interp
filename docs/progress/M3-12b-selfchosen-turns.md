# M3 — 12B self-chosen class signal appears at turn 4 pre-answer, not at Ready

**Run date:** 2026-04-21 by Codex.
**Source collection:** `runs/diag/selfchosen_ready_20bank_12b_directfit_20260421/`
from job `7230807` (300 attempts, 40 kept = 10 x `{elephant,cow,dog,horse}`).
**Scored positions:** pre-answer activations at turns 1..4, all 49 layers.
**Machine-readable report:** `runs/m3_12b_selfchosen_turns.json`.
**Detail table:** `docs/progress/M3-12b-selfchosen-turns-detail.md`.

## Scope

After `M3-12b-selfchosen-direct.md` showed that **Ready-state** direct-fit is
barely above chance at 12B, the cheapest informative next test was the one
STATUS proposed: look at the **pre-answer residual stream** later in the same
self-chosen dialogues.

This uses the existing kept 40-run dataset only; `diagnose_selfchosen_ready.py`
had already captured `turn_01..turn_04_activations.pt` for each run.

## Result

Chance is **25%** (4 realized classes). Turn-wise summaries across all non-L0
layers:

| turn | NC mean | NC median | NC best | NC best layer | LR mean | LR median | LR best | LR best layer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.31 | 0.35 | 0.45 | 9 | 0.31 | 0.31 | 0.475 | 47 |
| 2 | 0.21 | 0.20 | 0.50 | 16 | 0.26 | 0.26 | 0.40 | 46 |
| 3 | 0.23 | 0.20 | 0.45 | 18 | 0.23 | 0.23 | 0.375 | 15 |
| 4 | **0.40** | **0.46** | **0.625** | **44** | **0.40** | **0.45** | **0.60** | **42** |

Ready baseline from `M3-12b-selfchosen-direct.md` for comparison:

- Ready NC: mean **0.23**, max **0.45 @ L14**
- Ready LR: mean **0.27**, max **0.45 @ L4**

So the turn-4 pre-answer position is materially stronger than Ready by every
summary statistic:

- NC mean: **0.40 vs 0.23**
- LR mean: **0.40 vs 0.27**
- NC best: **0.625 vs 0.45**
- LR best: **0.60 vs 0.45**

The strongest signal is also a coherent **late-layer band**, not a single noisy
peak. Over layers 27..48:

- turn 4 NC mean = **0.549**
- turn 4 LR mean = **0.539**

The peak band is broad:

- NC is **0.53–0.62** from roughly **L27–L48**
- LR is **0.42–0.60** from roughly **L28–L48**

## Interpretation

1. **Ready really was too early.** The class signal is much stronger once the
   model has been forced to answer several yes/no questions.
2. **The useful self-chosen probe position at 12B is turn-4 pre-answer, not
   Ready.** This is now the best in-regime position we have tested.
3. **The strengthening is not monotone across turns.** Turn 1 is already better
   than Ready, turns 2 and 3 are weak, and turn 4 becomes strong. So the right
   lesson is not "later is always better"; it is "this position is question/
   context dependent, and turn 4 is the first clearly probe-usable one in the
   current regime."
4. **The signal cannot be explained by the public answer history on this kept
   subset.** The current 4-question panel is degenerate on
   `{elephant,cow,dog,horse}`: every class gives the same answer pattern
   `1,0,0,1`. So the turn-4 decode is not just reading off public dialogue.
   That makes this a cleaner latent-state result than the Ready failure alone.

## Decision consequence

1. **Move the self-chosen probe position from Ready to turn-4 pre-answer** for
   the 12B branch.
2. **Do not keep spending cycles on Ready-state sweeps.** That question is now
   answered.
3. **Next productive artifact:** a larger 12B self-chosen 20-bank collection,
   scored primarily at turn 4 late layers (`~L42–L48`), to see whether this
   rises from "real but moderate" to genuinely robust.

## Artifacts

- `runs/m3_12b_selfchosen_turns.json`
- `docs/progress/M3-12b-selfchosen-turns-detail.md`
