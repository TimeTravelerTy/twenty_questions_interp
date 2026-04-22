# M3 — 12B self-chosen diversity at T=0.7: temperature does not break the 4-class attractor

**Run date:** 2026-04-22 by Claude (Opus 4.7).
**Source collection:** `runs/diag/selfchosen_ready_20bank_12b_diversity_T07_20260422/`
from job `7237460` (1500 attempts, T=0.7, 20-bank, early-stop at 8
classes × 20 quota, h_rt 12h).
**Machine-readable report:** `runs/diag/.../results.json` (only the
attempt-distribution block is meaningful here — the Ready analysis is
the same 4-class {elephant,cow,dog,horse} set from prior runs).

## Scope

D-32 predicted that moving from T=0.0 to T=0.7 on the 20-bank would
broaden the realized class distribution beyond the
`{elephant, cow, dog, horse}` attractor seen in every 12B greedy self-
chosen collection so far (pilot, direct-fit, turn-4 scale-up). A
broader realized set is the only way to test whether turn-4 LR LOO
(0.787 @ L31 on n=80, chance 0.25 at 4 classes) degrades gracefully
as chance drops — e.g. to 0.125 at 8 classes.

This run is the T=0.7 probe with the new early-stop + attempt-distribution
diagnostics from D-32.

## Result

T=0.7 does **not** broaden the attractor. Across all 1500 attempts:

| class | count | share |
|---|---:|---:|
| horse | 572 | 38.1% |
| cow | 520 | 34.7% |
| elephant | 307 | 20.5% |
| dog | 101 | 6.7% |
| **all 16 others** | **0** | **0%** |

- Reveal parse success: **100%** (1500/1500).
- Ready parse success: **100%**.
- Answer parse success: **100%** (4/4 every run).
- Distinct parsed classes: **4**.
- Top-1 share: **38.1%** (horse).
- Entropy: **1.79 bits**; effective classes: **3.46**.

The candidate list is permuted per seed, so the attractor is not
list-position — it is the same 4 animals (`elephant, cow, dog, horse`)
regardless of where they appear in the presented list.

## Interpretation

1. **Temperature is not a usable knob for diversity here.** At 12B on
   this prompt, the first-token posterior over the Ready commitment is
   concentrated enough on these 4 animals that T=0.7 still samples only
   from them — not 1500 attempts produced a single realization of
   tiger, kangaroo, bat, dolphin, gorilla, cat, eagle, penguin,
   chicken, owl, cobra, crocodile, frog, shark, salmon, or bee.
2. **The attractor is prompt-induced, not sampling noise.** A 4-way
   collapse under T=0.7 means the model's prior over "which animal to
   keep as secret" given this exact prompt is effectively a
   4-class distribution. 4B under the same prompt collapsed to a
   different 2–4 class attractor ({salmon, frog} etc.) — so the
   attractor identity is both prompt- and scale-dependent.
3. **The M3 self-chosen problem at 12B is therefore genuinely a 4-class
   problem on this prompt.** It is not a methodological gap that we
   only have 4 realized classes — the model itself does not realize
   more under this condition. The n=80 turn-4 result
   (LR LOO 0.787 @ L31, chance 0.25) characterizes decodability on
   the model's *realized* distribution.
4. **The scale-axis hypothesis sharpens.** If 4B collapses to one set
   and 12B to a different set, the interesting scale-up question is
   whether 27B+ has a broader or narrower attractor. The "when does
   commitment become decodable" question and the "how diverse is the
   attractor" question are both worth tracking.

## Decision consequence

1. **Do not chase more temperature.** T=0.7 is already the most
   diversity we can extract from this prompt at 12B; T=1.0+ would
   start degrading instruction-following before it broadens the class
   set further (the posterior concentration is not a sharpness issue).
2. **Option A (preferred) — accept 4 classes as the headline.** Lock
   the M3 self-chosen result as "turn-4 pre-answer, LR LOO 0.787 @
   L31, on the model's realized 4-class distribution" and move to M4
   (causal patching + SAE case studies at L29–L31 turn 4).
3. **Option B (optional, cheap) — try one round of prompt variants**
   (e.g., swap the opening sentence, drop the explicit candidate list,
   or bias the framing toward novelty) as a side experiment before
   M4, to check whether the attractor is prompt-fragile. This is a
   scientific curiosity, not a blocker for M4.

## Artifacts

- `runs/diag/selfchosen_ready_20bank_12b_diversity_T07_20260422/results.json`
  (attempt_distribution block)
- Job: `7237460`, 1500 attempts in 1126.6s (~0.75s/attempt on 1 H100).
