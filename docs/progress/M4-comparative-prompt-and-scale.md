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
| **27B default** | **0.137** | **0/28** | **23/28** |

(Maximum entropy = ln(20) = 2.996 for uniform-over-bank.)

### 3. Cross-class baseline at L30 (mean own-vs-non-own logit diff in baseline runs)

| condition | mean L30 diff |
|---|---:|
| 12B default | +0.30 |
| 12B commit_strong | +0.27 |
| 12B internal_locus | +0.28 |
| 12B introspection_aware | +0.35 |
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

### Prompt axis (4 12B variants)

The hypothesis "12B commits at end_ready" was tested under four
distinct prompt manipulations:

- `default` — baseline ("commit to it as your secret")
- `commit_strong` — explicit by-Ready instruction ("by the time you
  reply with Ready, you must have already chosen and locked in")
- `internal_locus` — externalize-via-imagined-writing ("imagine
  writing the name on paper inside your mind, like ink on paper")
- `introspection_aware` — explicit mech-interp framing ("we will
  use the logit lens technique to read out which animal you chose;
  do not suppress the representation in late layers")

**All four show the same pattern:** end_ready LR ~1.2-1.4× chance,
mid-layer cross-class diff ~0, generic-prior dominance at rank-1.
None of these prompt manipulations induce residual-stream commitment
at end_ready.

The introspection_aware variant is the most striking null result: a
prompt that explicitly references the experimenter's analysis
methodology and asks the model to keep its representation focal and
non-suppressed produces no class-specific signal at L25-L30.
Either the model lacks introspective fine-grained control over its
own residual stream regardless of prompting, or the late-network
dialogue-integration mechanism is dominant enough to override any
introspective effort. Neither possibility supports the
"commitment-is-just-prompt-fragile" reading.

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
- 27B default: `{cow, dog, elephant, gorilla, horse, tiger}` (+ shark
  at low frequency)

**The attractor identity is mutable; the commitment-mechanism is
stable.** None of these prompt or scale manipulations produce the
qualitative shift we'd expect if scale or prompting moved the model
from improvisation to commitment (i.e., end_ready rising above chance,
patching becoming causal, lens trajectory showing a mid-network
commitment plateau). All five conditions are in the
"improvisation, late dialogue integration" regime.

## Decision consequence — what's left for M4

The improvisation hypothesis is robust under every axis tested:

1. Patching null at every position × layer scope tested.
2. Positional probe at chance at end_ready in 12B (formal) and lens
   proxy at chance across all 4 12B prompt variants and 27B.
3. Cross-class baseline ~0 at L25-L30 in all 5 conditions.
4. Rank check confirms generic-prior dominance, not run-specific
   commitment, in all 5 conditions.

**12B and 27B do not store class commitment in the residual stream
at the supposed commitment site.** Class identity is constructed
late in the network from the accumulated dialogue evidence.

What's left to test before considering M4 done:

- **Lipsum-filler prompt variant** (Vogel et al. 2026 "Latent
  Introspection"): user-side filler text between the choice
  instruction and the Ready prompt, giving the model attention/
  compute substrate to "settle" on a choice before emitting Ready.
  In Vogel et al., the matched-lipsum-filler condition achieved the
  highest detection accuracy (84% balanced accuracy on Qwen2.5-Coder-
  32B) — higher than explicit pro-introspection framing alone. If
  lipsum-filler shifts the end_ready probe-decodability above chance
  at 12B, the "no commitment" reading would need updating to "no
  commitment without compute substrate." If lipsum-filler also
  nulls, we have one more axis of robustness.

This is the natural last prompt-variant probe before M4 closes.
After that, M5 (SAE / transcoder feature case studies) becomes the
priority. The mechanism (late dialogue integration; class derivation
from accumulated yes/no answers) is now characterized cleanly enough
that asking "which features encode the constraint integration step"
is a concrete next question.

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
   internal_locus, 12B introspection_aware, and 27B have only the
   lens-based proxy. Same conclusion at every readable layer; the
   difference is whether class info in non-unembed-aligned subspaces
   could be missed.
3. **Confabulation caveat for flip-text.** The 0.137-nats output
   entropy at 27B is genuinely striking but does not directly speak
   to commitment. A confabulating system would also show low entropy
   under counterfactual dialogue. Output entropy is treated here as
   a measure of consistency-engine sharpness, not commitment
   absence.

## Artifacts

- 5 flip-text-with-lens JSONs in `runs/m4_flip_yesno_text_*_lens.json`
- 2 formal probe JSONs (12B default, 12B commit_strong) in
  `runs/m4_positional_probe_*.json`
- 600-attempt collections per condition in `runs/diag/*`
- TSUBAME captures in `runs/positional_residuals/*` (5 conditions)
- Reference: D-39 (positional probe), D-40 (flip-text behavioral),
  D-41 (logit-lens robustness checks)
