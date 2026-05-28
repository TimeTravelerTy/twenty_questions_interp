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

- **Ablate mode**: project the direction out at L16. Tests
  *necessity* rather than sufficiency. If the model still reveals
  the same class with the direction zeroed, the direction is fully
  epiphenomenal.
- **Layer sweep**: try L38 (pre_answer_q4 peak, LR 0.672) and L58
  (pre_reveal_gen peak, LR 0.820 — the strongest probe signal
  anywhere). Late-layer directions might be more causal.
- **Larger α**: see when total breakdown happens. If α=30 produces
  ~uniform random output, the L16 direction is purely additive
  noise; if it produces clean flip-to-source, the issue was just
  amplitude.
- **CCM** (Contrastive Causal Mediation,
  [ICLR 2025](https://openreview.net/forum?id=bUXa74EiOL)): the
  modern method that explicitly avoids the off-distribution issue
  via mediation analysis. Overkill for first pass but the rigorous
  follow-up if a steering positive emerges.

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

### Why L58 ≈ unembed

The L58 residual at pre_reveal_gen passes through one final layer
norm and the unembedding matrix to produce the reveal-token logits.
The fact that an LR probe achieves 0.820 LOO accuracy from that
residual means **the linear projection from L58 residual to class
identity is nearly perfect** — i.e. the class direction in L58
residual space is very close to the unembed direction for the class
token. Adding `α·(μ_src − μ_tgt)` at that exact position+layer is
essentially "add a vector that points toward the source-class token
in unembed space." It would be surprising if it *didn't* flip.

So the L58 positive isn't evidence of hidden commitment — it's a
readout-layer perturbation that almost can't fail.

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
