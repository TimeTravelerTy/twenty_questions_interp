# M5b — Probe-direction steering at 27B end_ready L16

**Status:** **DONE (2026-05-28).** Job 7782428, 576 trials in 77s
after model load. Result: the probe direction is **weakly causal for
one semantically-similar class pair (tiger→gorilla)** but
overwhelmingly null elsewhere, even at α=10 (10× the natural class-
mean separation magnitude). This is structurally distinct from
findings like Arditi et al.'s refusal-direction result, where a
single contrastive direction broadly controls behavior. For class
commitment in the 20-Questions setup, the probe-decoded direction
is overwhelmingly epiphenomenal, with a narrow exception consistent
with a semantically-meaningful sub-axis between similar classes.

## The test

CAA recipe (Panickssery et al. 2023): derive
`Δ = μ[src_class] − μ[tgt_class]` from class-mean residuals at
(`end_ready`, L16) — the probe peak position with LR LOO 0.508 (3.55×
chance). Inject `α · Δ` into the residual at L16 at every position
during the entire forward pass (prefill + decode). Sweep `α ∈ {1,
3, 10}`. Greedy reveal, measure flip-to-source.

Stronger than position-patching: the direction is added at every
position the model could attend to (not just one token's K/V), and
on every forward step including the reveal's first-token decode.
Direct test of "is this direction sufficient when injected?"

## Aggregate

| α | kept-target | flip-to-source | drift-to-other |
|---|---:|---:|---:|
| 1.0 | 185/192 (96.4%) | 1/192 (0.5%) | 6/192 (3.1%) |
| 3.0 | 183/192 (95.3%) | 2/192 (1.0%) | 7/192 (3.6%) |
| 10.0 | 179/192 (93.2%) | 2/192 (1.0%) | 11/192 (5.7%) |

Baseline non-determinism: 0/32 (clean). Class-mean L16 residual
norms are ~3418 across all 7 classes (uniform magnitude), so the
contrastive direction Δ has norm small relative to that — α=10
amplifies it but doesn't blow up the residual.

Drift-to-other rises with α but flip-to-source barely moves. As α
grows the steering is producing *unrelated* drift, not pushing
output toward the chosen source class — consistent with the
direction starting to push the residual off-distribution rather
than activating a clean class attractor.

## The one textbook-clean cell

**tiger → gorilla**, the only cell with monotonic α-dependence:

| α | flips to tiger | distribution |
|---|---:|---|
| 1.0 | 0/5 | {gorilla: 5} |
| 3.0 | 1/5 | {gorilla: 4, tiger: 1} |
| 10.0 | 2/5 | {gorilla: 2, tiger: 2, horse: 1} |

Monotonic gorilla → tiger flipping, with α=10 also showing one
off-target drift (attempt_403 → horse). This is the closest thing
to a positive causal finding for the probe direction: tiger and
gorilla are both large mammals, possibly sharing a
semantically-meaningful sub-axis that the contrastive direction
captures.

## The other "movement" cells are noise

**cow → horse**: flip-to-source 20% at α=1 and α=3, *only* at
`attempt_593` (the known boundary-degenerate horse target M5
documented). At α=10 it stops flipping. Not a causal steering
result — it's the same fragility surfaced under any small
perturbation.

Nothing else moves.

## Why this is the right kind of null

Recent literature on probe-direction causality is mixed:

- [Arditi et al. (NeurIPS 2024)](https://arxiv.org/abs/2406.11717)
  showed refusal in LLMs is mediated by a single direction —
  ablating it broadly removes refusal behavior. Strong CAA-style
  positive.
- [Pre-CoT probes paper](https://arxiv.org/pdf/2603.01437) reports
  probe directions for pre-decoded answers flip outputs >50% of the
  time under steering. Strong positive.
- Our result here: the probe direction for *class commitment* in a
  20-Questions self-chosen setup is causally weak — 1.0% flip-to-
  source at α=10, with one cell showing weak monotonic α-dependence.

The mechanism difference is plausible: refusal is a binary policy
the model implements via a (largely) single-direction mechanism;
class commitment in 20-Questions is determined by the prompt's
20-bank permutation and re-derived from the prompt at every later
position via attention. The probe finds a direction because the
*choice* projects into the residual stream as a side-effect; that
projection is not the mechanism the model uses to maintain the
choice.

## Caveat

Per [Steered LLM Activations are Non-Surjective](https://arxiv.org/html/2604.09839v1):
steering pushes the residual into states that may have no prompt
preimage. The drift-to-other at α=10 (5.7% of trials going to
unrelated classes) is the empirical signature — we're moving into
off-distribution territory before achieving clean flip-to-source.
A *null* under this protocol is therefore decisive ("even injecting
α·Δ broadly fails to redirect the reveal"); a *positive* would need
the off-distribution caveat. Our result is decisive null with one
semantic-pair exception.

## Open follow-ups

- **Position-bisection of commitment formation**: between L16
  end_ready (steering null) and L58 pre_reveal_gen (steering
  positive), the model transitions from "summary in progress" to
  "committed for readout." Steering with rollout at intermediate
  anchors — `pre_answer_q1..q4` (LR LOO 0.574–0.672 @ L38), or at
  `end_model_q4` — would locate the **earliest position where
  steering flips the reveal**, which is when the model has actually
  "decided." This is the more rigorous probe-direction-causality
  story the project would want for the blog.
- **Ablate mode at L16 with rollout**: necessity test. Project the
  direction out at L16 across the whole rollout. If the model still
  reveals the target, the direction is fully epiphenomenal in both
  the sufficiency *and* necessity senses.
- **Larger α at L16**: see when total breakdown happens. Drift→other
  at α=10 is already 7.8%; α=30 would likely produce uniform-random
  output without any clean flip-to-src, confirming that no amount
  of L16 amplification can extract a commitment that isn't there.
- **CCM** (Contrastive Causal Mediation,
  [ICLR 2025](https://openreview.net/forum?id=bUXa74EiOL)): the
  rigorous modern alternative to raw steering, explicitly handles
  the off-distribution concern via mediation analysis. Overkill for
  the M5b null story, but worth knowing if the bisection step above
  finds a positive somewhere mid-dialogue.

## What this is and isn't evidence for

**What M5b is evidence for** (in plain terms):
1. The model does NOT hold the secret animal "in mind" the way a
   human would — there's no early-formed commitment in the residual
   at end_ready that propagates causally to the reveal.
2. The class identity is encoded primarily in the prompt itself
   (which 20 animals, in what order) and re-derived from prompt
   context by attention at each later position.
3. By the time we reach the literal pre-reveal position, the
   model's last few layers do hold the class causally — that's the
   readout assembly, not a stored commitment.

**What M5b is NOT evidence for**:
- That there's no commitment anywhere in the model's forward pass.
  The L58 steering positive proves there IS a load-bearing class
  direction late in the network. We just haven't localized where
  the transition happens (see bisection follow-up).
- That the probe directions are useless. The L58 probe at LR 0.820
  successfully identifies a causally load-bearing direction; the
  L16 probe identifies a summary-but-not-substrate direction.
  Probes-as-readouts are valid; the warning is that probe accuracy
  alone doesn't establish causality (Heimersheim & Nanda).

## L16 steering with answer-rollout (the decisive null)

**Job 7783019 (2026-05-28).** Same direction as the original L16 run
but with `--answer-rollout` enabled and `--steer-scope from_anchor`
(inject only at positions ≥ end_ready, leaving the prompt encoding
untouched). The hook re-fires on every per-turn prefill, so the
steering bias enters A_1..A_4 decoding at each step — the model
gets to commit to or against the source direction across the
dialogue, not just at the reveal-token logits.

This fixes the same confound rollout fixed for patching: in the
original L16 steering run the manifest's (Q, A) history was teacher-
forced, so the model could re-derive class from visible answers and
the steering only got one shot at the reveal-token logits.

**Result: even more null than the non-rollout L16 steering.**

| α | kept | flip→src | drift→other | answer-flip slots |
|---|---:|---:|---:|---:|
| 1.0 | 188/192 (97.9%) | **0/192 (0%)** | 4/192 (2.1%) | 0/768 (0%) |
| 3.0 | 188/192 (97.9%) | 1/192 (0.5%) | 3/192 (1.6%) | 0/768 (0%) |
| 10.0 | 175/192 (91.1%) | 2/192 (1.0%) | 15/192 (7.8%) | 3/768 (0.39%) |

Baselines: 32/32 fully deterministic, 0 reveal drift, 0 answer
flips. The only non-zero answer-flip cell is at α=10:
`gorilla → shark/attempt_431 → reveal=tiger` (3/4 answers flipped,
regenerated `[T,F,F,T]`). Same fragile target (`attempt_431`),
same tiger attractor, same `[T,F,F,T]` fingerprint that surfaced
under late-band multi-anchor patching. **A class-pair-specific
fragility of one target, not a steering signal.**

The cow→horse 20% at α=3,10 is still attempt_593-only (0 answer
flips in both cases). The tiger→gorilla 20% at α=10 is a single
trial.

So the L16 direction is decisively epiphenomenal under the
strongest available protocol: direct injection of `α·(μ_src − μ_tgt)`
at the probe peak layer, scoped to positions ≥ end_ready, with
rollout giving the steering 5 forward passes to bias the model's
own decisions. **The model regenerates the same answers 99.6% of
the time, and the reveal flips to source 0–1% of the time.** This
directly answers the question "is the model committing to the
class at end_ready in a way the probe direction captures?" — no.

## Steering at pre_reveal_gen L58 (the strongest probe peak)

**Job 7782821 (2026-05-28).** Same script, but at the position+layer
with the strongest probe signal anywhere in the 16-anchor table:
**pre_reveal_gen L58, LR LOO 0.820 (5.74× chance)**. Scope =
`from_anchor` — inject only at positions ≥ pre_reveal_gen (the
literal last prefill token before reveal decoding + every decode
step). Cleaner than the L16 `scope=all` run, which perturbed the
entire prompt encoding.

**Aggregate flips broadly — but largely tautologically.**

| α | kept-target | flip→src | drift→other |
|---|---:|---:|---:|
| 1.0 | 16/192 (8.3%) | **123/192 (64.1%)** | 53/192 (27.6%) |
| 3.0 | 0/192 (0.0%) | **154/192 (80.2%)** | 38/192 (19.8%) |
| 10.0 | 0/192 (0.0%) | **144/192 (75.0%)** | 48/192 (25.0%) |

This is qualitatively different from L16's 0.5-1.0% flip rate. But
the interpretation isn't "we found the load-bearing direction" —
it's "the L58 contrastive direction at pre_reveal_gen is, by
construction, the unembed direction for the class token."

### Why L58 flips and what it does / doesn't tell us

The L58 residual at pre_reveal_gen passes through **four more
transformer blocks** (L59, L60, L61, L62) plus the final layer norm
and the unembedding matrix to produce the reveal-token logits. So
the L58 steering isn't directly perturbing the unembed input — the
direction has to survive four blocks of nonlinear processing
(attention, MLP) to still drive output. The fact that it broadly
does (80% flip→src at α=3) is real evidence that the class signal
at L58 is *causally* held in the last 4 layers in a way an additive
direction can override.

But this **doesn't speak to early commitment**. The improvisation
thesis (M4) and the M5 patch null at end_ready say: the model does
not maintain a class-specific commitment in the residual at
end_ready that propagates forward; instead, it re-derives class via
attention over the prompt at every later position. By the time the
model has reached `pre_reveal_gen` (the last token before reveal
decoding), it has just finished that re-derivation — it has
committed *now*, in the residual at that position, for the next
~4 layers of readout processing. The L58 positive says exactly
that: at the moment of readout, the class signal is held in the
late residual. It doesn't say the model held it 4 turns ago at
end_ready.

Compare:
- **End_ready L1-L62 full-residual patch (M5 7737409)**: replace
  the *entire* residual stack at end_ready with a source's. Null.
  The model recomputes.
- **End_ready L16 steering with rollout (this work, 7783019)**:
  add α·Δ at L16 across positions ≥ end_ready, throughout the
  whole dialogue rollout. Null at the answer level (0–0.4% slots)
  and the reveal level (0–1%).
- **pre_reveal_gen L58 steering**: 80% reveal flip at α=3.

The early-position null + late-position positive is consistent
and tells a unified story: the model re-derives the class at each
forward pass; the result of that re-derivation is increasingly
decodable as you approach the output, and is held causally in the
late residual just before readout. The probe at L16 is detecting a
*summary* of an in-progress re-derivation that doesn't propagate;
the probe at L58 is detecting the *finished* commitment that does
drive the next 4 layers. (And steering at L58 confirms that probe
points to a load-bearing direction at the late position.)

### Per-source-class pattern: not all directions are clean

At α=3 (where almost everything pegs to 0% or 100%), the per-source
output distributions reveal the structure:

| src direction | flip→src @ α=3 | output distribution |
|---|---:|---|
| horse | 100% | {horse: 27} |
| tiger | 100% | {tiger: 27} |
| cow | 93% | {cow: 25, horse: 2} |
| dog | 93% | {dog: 25, horse: 2} |
| elephant | 93% | {elephant: 25, horse: 2} |
| gorilla | 93% | {gorilla: 25, horse: 2} |
| **shark** | **0%** | **{salmon: 30}** |

The shark anomaly is the most informative: at α=1, shark steering
flips 100% to shark; at α=3, it flips 100% to **salmon** (a class
that exists in the 20-bank but not in any of our realized targets).
The contrastive direction `μ_shark − μ_X` at L58, when amplified
~3×, overshoots into a region of residual space whose nearest
unembed-class token is "salmon" rather than "shark." This is the
[steering off-manifold critique](https://arxiv.org/html/2604.09839v1)
fired in miniature: enough of the direction takes you to a state
with no realistic prompt preimage.

The "horse leakage" pattern (every other class's α=3 steering shows
2/27 trials going to horse) suggests that any reveal-aligned
direction added at α≥3 has some component that aligns with the
horse unembed direction in a generic way — possibly a class-
agnostic "be confident about an animal" effect that the model
discharges into the most common/easiest class token.

### Combined with L16: a coherent story

This isn't a contradiction of the L16 null — it's the complement
that *closes* the story:

- **L16 (end_ready)**: probe finds class direction at LR 0.508.
  Steering at L16 essentially null. → The L16 direction is decodable
  but doesn't propagate forward as commitment.
- **L58 (pre_reveal_gen)**: probe finds class direction at LR 0.820.
  Steering at L58 broadly positive. → The L58 direction *is* the
  unembed direction; perturbing it perturbs the readout. But this
  doesn't imply the L58 direction was *propagated* from L16 — the
  full-residual L1-L62 patch at end_ready (M5 job 7737409, null)
  rules that out.

The picture: the model computes the class identity afresh at each
later position by attending back over the prompt and its
accumulated dialogue, and the result of that computation shows up
in increasingly-decodable form as you approach the output. The
probe directions at intermediate positions are *summaries* of that
computation, not *substrates* of it; they can't be patched or
steered at early positions because the model isn't reading from
them — it's recomputing.

The L58 positive is consistent with: the model has finished
deciding by L58 of pre_reveal_gen (which is why probe accuracy is
so high), and tilting that decision via direct addition at the
readout layer trivially changes the output. This is the *expected*
behavior of any classifier whose last hidden layer carries the
class signal.

## Artifacts

- Code: [scripts/steer_class_direction.py](../../scripts/steer_class_direction.py)
- L16 steering output: `runs/m5b_steer_27b_default_endready_L16_add.json` (job 7782428, gpu_1, 77s steered + 6min model load)
- L58 steering output: `runs/m5b_steer_27b_default_prereveal_L58_add_from_anchor.json` (job 7782821, gpu_1, 464s steered + 5min model load)
- Per-class probe analysis (queued as 7782923): `runs/m5b_class_dirs_27b_*.json` — used to validate the per-source steering pattern matches per-class probe accuracy.
