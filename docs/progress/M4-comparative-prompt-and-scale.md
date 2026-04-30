# M4 — Comparative analysis across prompt variants and scale (D-42)

**Run dates:** 2026-04-26 to 2026-04-28 by Claude (Opus 4.7), with
methodological direction from the user.
**Five conditions analyzed:**

| condition | model | prompt | collection job | flip-lens job |
|---|---|---|---|---|
| 12B default | gemma-3-12b-it | default | (M3 scale-up `7232075`) | `7267006` |
| 12B commit_strong | gemma-3-12b-it | commit_strong | `7272281` | `7272417` |
| 12B internal_locus | gemma-3-12b-it | internal_locus | `7274562` | `7274689` |
| 12B introspection_aware | gemma-3-12b-it | introspection_aware | `7274563` | `7274690` |
| 12B lipsum_filler | gemma-3-12b-it | lipsum_filler | `7287532` | (skipped — flip-text deprioritized) |
| 27B default | gemma-3-27b-it | default | `7272415` | `7272708` |

## Caveat on flip-text behavioral evidence

D-40 established that flipping a yes/no answer text changes the
reveal class in 74.7% of trials (12B default), and D-41 followed up
with logit-lens to address the "suppressed pre-commitment" worry.
User raised a sharper version of the doubt: even if behavior shifts
under counterfactual dialogue, that does not necessarily falsify a
prior commitment. The same pattern would arise from confabulation —
**a model that committed and then lost access to that commitment
under perturbed evidence would behaviorally look identical to a
model that never committed and is improvising.** Compare to humans:
change someone's memory of a past event and they will smoothly
confabulate a coherent narrative; this does not mean the original
memory was never formed.

For the comparative analysis below, **flip-text behavioral evidence
(kept-class rate, output entropy under flip) is treated as
illustrative, not load-bearing.** The load-bearing evidence is:

1. **Patching null (D-35–D-38).** No single-position residual patch
   at any pre-answer position, at any layer scope (L29 → L27-L48 →
   L1-L48 → all 4 turns), flips reveals on stable targets. 0/2280
   trials.
2. **Positional probe at end_ready (D-39).** LR LOO 1.20× chance at
   12B default. Class is at chance at the supposed commitment site
   in the residual stream.
3. **Lens-based cross-class baseline (D-41).** At L25-L30, own-class
   logit elevation in own-runs vs non-own-runs is ~0 logits. The
   mid-layer "elevation" of attractor classes above neutral is a
   generic prior, not run-specific commitment.
4. **Rank check at L30.** The single class ranked #1 at L30 in
   flipped trials is the same class regardless of the run's actual
   identity (horse at 12B; tiger at 27B). The original class is
   never specifically elevated above its peer attractors at any
   layer where the lens is meaningfully readable.

These four pieces of evidence test whether a class commitment is
**present in the residual stream** in a way that a downstream layer
can read. They do not depend on behavioral interpretation, and the
confabulation worry does not apply.

## Cross-condition data tables

### 1. Behavioral attractor (which classes does the model realize at greedy)

| condition | realized classes | top-1 share | effective classes |
|---|---|---:|---:|
| 12B default | cow, dog, elephant, horse | 38.5% | 3.46 |
| 12B commit_strong | cow, dog, elephant, horse | 35.8% | 3.70 |
| 12B internal_locus | cow, dog, elephant, horse | similar | similar |
| 12B introspection_aware | **cow, elephant, horse** (dog dropped) | similar | similar |
| 12B lipsum_filler | cow, dog, elephant, horse | 43.0% | 3.13 |
| 12B less_obvious (D-34, ref) | + gorilla, kangaroo, penguin (6 cls) | 38.5% | 3.90 |
| **27B default** | + tiger, gorilla, shark (7 cls) | 40.0% | 4.27 |

The attractor identity is **mutable** under prompt and scale. 12B
introspection_aware loses dog entirely (3-class attractor); 27B
gains tiger and gorilla without any prompt nudge.

### 2. Output entropy under flip-text (illustrative, see caveat above)

| condition | mean cell entropy (nats) | mixed cells (H>1) | sharp cells (H<0.3) |
|---|---:|---:|---:|
| 12B default | 0.845 | 7/16 | 3/16 |
| 12B commit_strong | 0.794 | 7/16 | 4/16 |
| 12B internal_locus | 0.893 | 8/16 | 3/16 |
| 12B introspection_aware | 0.697 | 5/16 | 2/16 |
| 12B lipsum_filler | n/a | n/a | n/a |
| **27B default** | **0.137** | **0/28** | **23/28** |

(For lipsum_filler, flip-text was skipped on user direction — flip-text
behavioral evidence is now treated as illustrative-only per the
confabulation caveat above; running it for one more variant adds no
load-bearing information.)

(Maximum entropy = ln(20) = 2.996 for uniform-over-bank.)

### 3. Cross-class baseline at L30 (mean own-vs-non-own logit diff in baseline runs)

| condition | mean L30 diff |
|---|---:|
| 12B default | +0.30 |
| 12B commit_strong | +0.27 |
| 12B internal_locus | +0.28 |
| 12B introspection_aware | +0.35 |
| 12B lipsum_filler | +0.39 |
| 27B default | +0.01 |

All five conditions show ~0 class-specific elevation in the residual
stream at the mid-layer where the lens first becomes readable. **No
condition shows a "commitment-like" signal at L25-L30.** 27B is
striking: the mid-layer signal is even *flatter* than 12B
(+0.01 vs +0.30) — class identity emerges later in the network.

### 4. Rank-1 class at L30 in flipped trials (which class wins regardless of run)

| condition | dominant rank-1 class |
|---|---|
| 12B default | horse (in 9/14 cells) |
| 12B commit_strong | horse (in 8/14 cells) |
| 12B internal_locus | horse (typically) |
| 12B introspection_aware | horse (typically) |
| 12B lipsum_filler | **horse (100% share at L30 across all 600 baseline runs)** |
| 27B default | **tiger** (in 19/20 cells) |

In every condition, the rank-1 class at L30 is determined by which
class has the strongest model prior, NOT by which class the run
"committed" to. 27B's strongest prior shifts from horse (12B) to
tiger (27B) — same generic-prior dominance pattern, different
identity.

### 5. Decodability progression along the dialogue (12B-default summary, applies to all)

```
end_user_prompt    LR ~1.30× chance — basically nothing
end_ready          LR ~1.20× chance — basically nothing (key result)
end_user_q1..q3    LR ~1.5× chance — weak
end_model_q1..q3   LR ~1.5× chance — weak
end_user_q4        LR ~2.2× chance — moderate
end_model_q4       LR ~2.2× chance — moderate
end_reveal_user    LR ~2.2× chance — moderate
pre_reveal_gen     LR 3.7× chance — strong (partly tautological)
```

12B commit_strong: end_ready 1.35× (within noise of default 1.20×).
12B internal_locus / introspection_aware: lens-based proxy ~equiv to
default. 27B: lens-based proxy at L25-L30 even flatter than 12B.

The decodability profile is consistent across all 5 conditions:
class signal is at chance through Ready, weak through mid-dialogue,
and crystallizes only at reveal-token generation time.

(Caveat: the formal probe-anchors LR LOO numbers exist for 12B
default and 12B commit_strong. For 12B internal_locus and 12B
introspection_aware, only the lens-based cross-class proxy is
available — same conclusion at the unembed-readable subspace, but
not the strictly-stronger probe on the full residual. The 27B
formal probe-anchors hit walltime at 3h30m and was not retried; the
lens-based proxy is the substitute.)

## Three robustness axes for the improvisation hypothesis

### Prompt axis (5 12B variants)

The hypothesis "12B commits at end_ready" was tested under five
distinct prompt manipulations:

- `default` — baseline ("commit to it as your secret")
- `commit_strong` — explicit by-Ready instruction ("by the time you
  reply with Ready, you must have already chosen and locked in")
- `internal_locus` — externalize-via-imagined-writing ("imagine
  writing the name on paper inside your mind, like ink on paper")
- `introspection_aware` — explicit mech-interp framing ("we will
  use the logit lens technique to read out which animal you chose;
  do not suppress the representation in late layers")
- `lipsum_filler` — Vogel et al. 2026 "Latent Introspection"
  matched filler: ~500-token placeholder block between the choice
  instruction and the Ready cue, giving the model substantially
  more prefill compute substrate before end_ready is read.

**All five show the same pattern:** end_ready / pre_reveal_gen
mid-layer cross-class diff ~0 (range +0.27 to +0.39 at L30 across
the four matched-bank 12B variants; rank-1 = horse with near-100%
share at L30 regardless of run). None of these prompt manipulations
induce a run-specific residual-stream commitment at end_ready.

The introspection_aware variant is striking as the most direct
null: a prompt that explicitly references the experimenter's
analysis methodology and asks the model to keep its representation
focal and non-suppressed produces no class-specific signal at
L25-L30. Either the model lacks introspective fine-grained control
over its own residual stream regardless of prompting, or the
late-network dialogue-integration mechanism is dominant enough to
override any introspective effort.

The lipsum_filler variant is striking as the cleanest "more compute"
null: ~500 extra tokens of prefill before end_ready do not move the
mid-layer signal off chance (+0.39 at L30, vs +0.30 baseline) and
do not change the rank-1 generic-prior pattern (horse, 100% share).
If a latent commitment could form given enough prefill compute, this
is where we'd expect to see it — and we do not. Neither variant
supports the "commitment-is-just-prompt-fragile" or
"commitment-is-compute-starved" readings.

### Scale axis (12B → 27B default)

Behavioral output entropy collapses 6× (0.845 → 0.137 nats). 27B's
flip-text consistency is dramatically sharper: T1 → crocodile in
~100% of trials regardless of run, T3 → dolphin in ~100%, T4 →
gorilla in ~100% across all 6 starting classes.

**But the residual-stream mechanism does NOT change with scale.**
Cross-class L30 diff goes from +0.30 logits (12B) to +0.01 (27B) —
mid-network signal is *even flatter* at scale, not sharper. End_ready
remains at chance. Late-network L48 differentiation goes from
~5-15 logits (12B) to ~10-46 logits (27B) — *much* steeper at the
final layer.

The scale picture: **27B improvises more decisively via sharper
late-network dialogue integration, not via earlier commitment.**
The mechanism is the same, executed with higher confidence in the
final layers. This is a clean form of the scale-axis result we
flagged at project start: scale doesn't shift the model from
improvisation to pre-commitment; it sharpens the consistency engine.

### Attractor axis (which classes the model actually emits)

The model's prior over the attractor cluster changes substantially
across both prompt and scale:

- 12B default: `{cow, dog, elephant, horse}`
- 12B less_obvious: + gorilla, kangaroo, penguin
- 12B introspection_aware: cow/elephant/horse (DOG dropped)
- 12B lipsum_filler: `{cow, dog, elephant, horse}` (top-1 share
  43%, slightly *narrower* than default — extra filler if
  anything tightens the attractor)
- 27B default: `{cow, dog, elephant, gorilla, horse, tiger}` (+ shark
  at low frequency)

**The attractor identity is mutable; the commitment-mechanism is
stable.** None of these prompt or scale manipulations produce the
qualitative shift we'd expect if scale or prompting moved the model
from improvisation to commitment (i.e., end_ready rising above chance,
patching becoming causal, lens trajectory showing a mid-network
commitment plateau). All six conditions (5 prompt variants on 12B
plus 27B default) are in the "improvisation, late dialogue
integration" regime.

## Decision consequence — M4 closure

The improvisation hypothesis is robust under every axis tested:

1. Patching null at every position × layer scope tested.
2. Positional probe at chance at end_ready in 12B (formal) and lens
   proxy at chance across all 5 12B prompt variants and 27B.
3. Cross-class baseline ~0 at L25-L30 in all 6 conditions
   (12B `default` +0.30, `commit_strong` +0.27, `internal_locus`
   +0.28, `introspection_aware` +0.35, `lipsum_filler` +0.39, 27B
   default +0.01 — all small, none crosses any meaningful threshold,
   27B is even *flatter* than 12B).
4. Rank check confirms generic-prior dominance, not run-specific
   commitment, in all 6 conditions (12B: horse rank-1 with
   near-100% share regardless of run; 27B: tiger).

**12B and 27B do not store class commitment in the residual stream
at the supposed commitment site.** Class identity is constructed
late in the network from the accumulated dialogue evidence.

The `lipsum_filler` variant was the natural last prompt-variant
probe and it nulls cleanly: ~500 extra tokens of prefill compute
substrate before end_ready do not produce a mid-layer commitment
signal. The "no commitment without compute substrate" reading is
ruled out alongside "commitment-is-prompt-fragile" and "introspective
control with explicit framing".

**M4 is now closed.** The mechanism is characterized: improvisation
via late-network (L30→L48) dialogue integration; no mid-network
class storage at any point in the prefix. M5 (SAE / transcoder
feature case studies) becomes the priority — asking "which features
encode the yes/no constraint accumulation and the class-derivation
step" is the concrete next question, and the residual-level
characterization is solid enough to support feature-level dissection.

## Methodological caveats

1. **27B formal probe-anchors hit walltime.** The cpu_8 LOO at 122
   runs × 5376 hidden × 63 layers × 12 anchors was budgeted at
   ~150min and ran at ~210min. The lens-based cross-class baseline
   serves as the substitute and gives a strictly weaker but
   directionally consistent signal. If we want the strict probe
   numbers, the rerun would need a layer subset (8 layers ≈ 30min)
   or cpu_16 with 6h walltime.
2. **Per-variant probe completeness.** 12B default and 12B
   commit_strong have full formal probe-anchors data. 12B
   internal_locus, 12B introspection_aware, 12B lipsum_filler, and
   27B have only the lens-based proxy. Same conclusion at every
   readable layer; the difference is whether class info in
   non-unembed-aligned subspaces could be missed.
3. **Confabulation caveat for flip-text.** The 0.137-nats output
   entropy at 27B is genuinely striking but does not directly speak
   to commitment. A confabulating system would also show low entropy
   under counterfactual dialogue. Output entropy is treated here as
   a measure of consistency-engine sharpness, not commitment
   absence.
4. **Lipsum_filler used 600 attempts vs 80 kept for the other 12B
   variants.** The capture script processes all attempts with a
   parsed reveal class, not only the canonical-bank-class kept set.
   For the other variants the kept-vs-non-kept distinction was the
   table reference (n=80); for lipsum_filler all 600 attempts have
   reveal in the kept attractor `{cow, dog, elephant, horse}` so the
   distinction collapses. Numbers are computed over 600 lipsum_filler
   runs vs 80 for the other variants — the lipsum_filler estimate is
   strictly more precise, not less.

## Artifacts

- 5 flip-text-with-lens JSONs in `runs/m4_flip_yesno_text_*_lens.json`
  (lipsum_filler intentionally not run — flip-text deprioritized).
- 2 formal probe JSONs (12B default, 12B commit_strong) in
  `runs/m4_positional_probe_*.json`
- 600-attempt collections per condition in `runs/diag/*`
- TSUBAME captures in `runs/positional_residuals/*` (6 conditions)
- Lens summary for lipsum_filler: `runs/m4_lipsumfiller_lens_summary.json`
  (computed by `scripts/analyze_positional_lens.py`).
- Reference: D-39 (positional probe), D-40 (flip-text behavioral),
  D-41 (logit-lens robustness checks), D-42 (this comparative
  cross-prompt-and-scale analysis).
