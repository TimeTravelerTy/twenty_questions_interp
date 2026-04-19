# M3 — self-chosen Ready-state smoke, T=0.7 (4B)

**Run date:** 2026-04-19 by Claude (Opus 4.7).
**Model:** `google/gemma-3-4b-it` on TSUBAME H100, bfloat16.
**Job:** `tq_m3_selfchosen_T07` (`7219788`). Repo HEAD at run time: `65e482e`.
**Artifacts:** `runs/diag/selfchosen_ready_T07_20260419/` (local + TSUBAME).

## Scope

Follow-up to `M3-selfchosen-ready-smoke.md`: does temperature sampling (T=0.7
across Ready, question, and reveal generations) break the greedy choice
collapse that made the T=0 self-chosen run a 2-class degenerate case? And
does it close the geometric gap to persistence State B?

Same 4 candidates (`tiger, eagle, frog, salmon`) and primary question set.
120 attempts, target 4 kept per class.

## Topline finding 1 — choice collapse survives T=0.7

120 attempts → **tiger 0, eagle 0, frog 24, salmon 96**. Still only 2/4
realized classes. Not a greedy artifact: this is the model's actual choice
distribution under the self-chosen prompt at 4B. Sampling at T=0.7 does not
surface tiger or eagle at all.

Position-0 breakdown (reveal | position-0 candidate | count):

| position-0 | reveal = salmon | reveal = frog | total |
|---|---|---|---|
| salmon | 39 | 0 | 39 |
| eagle  | 31 | 0 | 31 |
| tiger  | 20 | 4 | 24 |
| frog   | 6  | 20 | 26 |

Readings:
- **salmon is the dominant attractor.** It wins even when eagle or tiger is
  shown first (31/31 for eagle@0, 20/24 for tiger@0).
- **Only frog at position-0 reliably flips the choice away from salmon**
  (20/26 pick frog). But salmon still "steals" 6/26 even then.
- **Tiger and eagle are essentially zero-probability picks** under the
  self-chosen prompt at 4B — no permutation or sampling perturbation
  surfaces them in 120 attempts.

Scientific implication: greedy decoding wasn't the cause of the collapse.
The model has a strong prior over which of the 4 candidates is plausible
as "a secret animal," and it is approximately a delta on salmon with some
probability mass on frog. The 4-candidate subset is not well-sampled by
the self-chosen condition.

## Topline finding 2 — T=0.7 tightens the geometry but does not close the gap to State B

Balanced 2-class analysis (frog n=4, salmon n=4), restricted to the realized
classes.

| metric | T=0.7 | T=0 | State A | State B |
|---|---|---|---|---|
| post-13 best NC LOO | 100% @ **L24** | 100% @ L29 | 100% @ L21 | 100% @ L13+ |
| NC LOO at L21 (A's peak) | **75%** | 64% | 100% | 100% |
| post-13 within-between contrast | **+1.14e-04** | +7.85e-05 | +1.31e-02 | +5.26e-04 |
| ratio vs T=0 | 1.45× | 1× | 167× | 6.7× |
| ratio vs State B | 1 / 4.5× | 1 / 6.7× | 25× | 1× |

Two things moved in the right direction vs T=0:

1. **Best-layer NC shifted earlier** (L29 → L24). Identity becomes decodable
   at slightly mid-network depths under sampling, not just at the very last
   blocks.
2. **Contrast grew 1.45×.** Still ~4.5× weaker than State B and ~115× weaker
   than State A, but the monotone story is preserved and partly narrowed.

The comparison vote against persistence is now **mixed** (NC tie at the
reference layer L21; contrast still firmly State B) vs T=0's clean
state_b / state_b vote. The self-chosen latent is getting closer to B as
sampling widens the forward pass noise, but the geometry is still weaker
than the lock-in persistence state at matched sample size.

## Topline finding 3 — sampling degrades question-answering fluency

Primary-question correctness dropped: **17/32 (53%) at T=0.7** vs higher
at T=0 (per-candidate 50–56% now; no free lunch from sampling). The
bank-disputed `frog.has_four_legs` cell and general attribute-ambiguity
are now compounded by sampling noise in the answer token.

This is important for the M3 scientific interpretation: at T=0.7 the
model is less trustworthy on yes/no questions, so any downstream
"question-conditioned" probe needs to bake in that calibration and
correctness are coupled.

## Revised picture

- **The 4-candidate self-chosen setup is biased at 4B regardless of
  decoding temperature.** Category and identity priors (salmon > frog >>
  {tiger, eagle}) swamp any per-permutation variation. "Self-chosen" on
  this small panel is de facto a 2-animal judgement.
- **The Ready-state identity latent *is* present for frog vs salmon at
  self-chosen**, decodable via NC at 100% from L24 (T=0.7) or L29 (T=0).
  But the cosine geometry is weaker than persistence State B by 4.5–6.7×,
  and much weaker than State A.
- **The 4-class hypothesis test (4-way NC LOO, 4-class A-vs-B vote) is not
  feasible on this 4-candidate subset** at 4B. Needs either (a) a wider
  candidate pool where sampling actually surfaces variety, or (b) a
  strict per-class quota enforced by resampling even when the natural
  distribution is narrow.

## Decision implications

- **D-24 (this session):** the 4-candidate self-chosen smoke is now
  scientifically closed as a test of the 4-way identity question at 4B.
  Further improvement on the self-chosen vs persistence comparison
  requires widening the prompt's candidate list (hypothesis: the full
  20-candidate panel will broaden the choice distribution simply by
  diluting the salmon attractor) or switching to a stronger model where
  competence + prior sharpness trade differently (D-05 ladder).
- **M4 pipeline unaffected from D-23:** probes still fit at self-chosen
  Ready directly (not transferred from calibration); SAE case studies
  still target layers ≥ L24.

## Caveats

- Still only 2 realized classes; balanced n=4 per class is small.
- Sampling is per-run (torch.manual_seed(seed) before each generate); the
  hidden state itself is still a deterministic function of the prompt. The
  "distribution over candidates" that temperature sampling surfaces is the
  reveal-prompt distribution, not truly a distribution over Ready-state
  latents. This limits what temperature can ever do in this diagnostic.
- bfloat16 (matches T=0); no dtype confound added here.

## What this does NOT conclude

- Does not close the hypothesis that the full 20-candidate self-chosen
  prompt will broaden the distribution. That's the next test.
- Does not close D-05. At 12B+ the prior may sharpen or redistribute; the
  "salmon attractor" at 4B may be model-specific.
- Does not update the bank-audit backlog.
