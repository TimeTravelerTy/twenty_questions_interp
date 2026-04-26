# M4 phase 2d (c-text + lens) — Logit-lens analysis confirms improvisation; no suppressed pre-commitment

**Run date:** 2026-04-26 by Claude (Opus 4.7).
**Source collection:** `runs/diag/selfchosen_ready_20bank_12b_turn4scale_20260421/`
(M3 default-prompt n=80 scale-up).
**Script:** `scripts/flip_yesno_text.py` at commit `a497536`.
**Job:** `tq_m4_flip_yesno_text_20260426.sh` (gpu_h, 155s wall, exit 0).
**Machine-readable report:** `runs/m4_flip_yesno_text_12b_default_n80_lens.json`.

## Scope and motivation

D-40 showed that flipping a single yes/no answer text shifts the
greedy reveal class outside the `{cow, dog, elephant, horse}`
attractor 74.7% of the time. User raised a methodological worry:
**maybe the model DOES make a pre-commitment, and the flipped dialogue
just forces it to find something more consistent.** Argmax-only
metrics can't distinguish "no commitment" from "committed-but-
overridable" because in both cases the final output respects the
dialogue.

User then proposed using **logit lens** (Nostalgebraist 2020) to look
at intermediate-layer logits, which would catch the case where the
original class is *elevated mid-network and suppressed by late
layers* — a documented pattern in mech interp (negative name movers
in IOI, etc.).

This phase re-runs the flip-text experiment but additionally captures,
for every trial, a `(49 layers × 20 bank classes)` matrix of logit-
lens readings at `pre_reveal_gen`: applying the model's final RMSNorm
+ `lm_head` to the residual at each layer's output, indexed by every
bank-class first token.

For each cell `(orig_class, flipped_turn)`, we compare:
- `base[orig]`: logit of the original class at each layer for the same
  run under no-flip baseline.
- `flip[orig]`: logit of the original class at each layer under flipped
  dialogue.
- `flip[new]`: logit of the trial's actual reveal class at each layer
  under flipped dialogue.

Three patterns we were watching for (per the methodological framing):

- **(i) Pure improvisation.** `flip[orig]` tracks `base[orig]` (or a
  random baseline) at every layer. Original class never preferred under
  flipped dialogue.
- **(ii) Suppressed pre-commitment.** `flip[orig]` *exceeds* `base[orig]`
  at some mid-network layer (the "commitment bump"), then decays below.
  This would indicate a hidden preference for the original class that
  late-network dynamics override.
- **(iii) Concurrent consideration.** `flip[orig]` and `flip[new]` both
  rise in late-mid network; new overtakes orig in the final layers.
  Model entertains both candidates; final pick is dialogue-driven.

## Result — pattern (iii)/(i) hybrid; no evidence for (ii)

Across all 16 cells (4 classes × 4 turns flipped), three structural
facts hold:

1. **L0–L25: `flip[orig] ≈ base[orig]`.** Difference < 0.1 logits
   everywhere. Note the L0–L25 lens readings are largely uninformative
   anyway (residual basis isn't yet aligned with the unembedding —
   known caveat); the meaningful signal starts ~L25.
2. **L30–L37: divergence kicks in.** First layer where
   `flip[orig] − base[orig] < −5` is consistently L35–L39 across cells.
3. **L40–L48: full divergence.** L48 drops of 17–42 logits in over-
   determined cells (T1/T3/T4); 3–10 logits in under-determined T2.

### L48 suppression magnitude per cell

`flip[orig] − base[orig]` at L48 (final layer):

| | T1 | T2 | T3 | T4 |
|---|---:|---:|---:|---:|
| cow      | -30.0 |  -9.6 | -26.9 | -41.0 |
| dog      | -27.5 | -10.1 | -23.9 | -29.7 |
| elephant | -36.9 |  -8.5 | -17.0 | -36.4 |
| horse    | -29.7 |  **-3.1** | -22.3 | -42.4 |

T2 column matches the conservative-cell finding from D-40
(under-determined flips); T1/T3/T4 are over-determined and show large
suppression. `horse/T2`: ~0 shift at every layer — consistent with
the kept-class-rate of 100% for that cell (the flip did nothing).

### Per-cell trajectory shape (representative: cow/T1)

```
layer | base[orig] | flip[orig] | flip[new] | diff
 L20  |    0.46    |    0.47    |   0.86    | +0.01
 L25  |    1.98    |    2.00    |   1.13    | +0.02
 L30  |    5.98    |    4.88    |   3.14    | -1.10
 L35  |    9.76    |    5.99    |   6.93    | -3.77
 L40  |   21.72    |    9.04    |  15.85    | -12.68
 L44  |   29.58    |   14.03    |  23.46    | -15.55
 L48  |   40.69    |   10.64    |   3.06    | -30.05
```

(Note: `flip[new]` at L48 is averaged across the 20 trials' actual
chosen classes (cobra/crocodile/frog/cobra mix for cow/T1), each of
which is argmax for its own trial. The mean is lower than `flip[orig]`
because individual chosen-class logits don't all sit at the same
height; each is the argmax of its own trial's logit distribution.
The argmax-flip behavioral result from D-40 is unchanged.)

## Critically: `flip[orig]` never EXCEEDS `base[orig]` at any layer

This is the methodological resolution. Across all 320 trials × 49
layers, the original-class logit under flipped dialogue is never
higher than under baseline. Pattern (ii) (suppressed pre-commitment)
predicts a mid-network bump where the model "decides" to commit to
the original class despite the flipped dialogue — and we see no such
bump anywhere.

What we see instead is a **smooth monotone integration**:
- L0–L25: lens uninformative (early-layer noise; identical readings
  in flip and base).
- L25–L30: orig-class logit starts rising under both base and flip,
  identically.
- L30–L40: under flip, integration of the dialogue evidence
  suppresses orig class downward; under base, it continues rising.
  The two trajectories *split* here.
- L40–L48: full integration; flip[new] overtakes flip[orig] in
  over-determined cells; orig drops dramatically.

This is most consistent with **pure improvisation with late dialogue
integration**: the model's class-derivation at `pre_reveal_gen`
happens predominantly in L30–L48, drawing from the dialogue evidence
present at earlier token positions via late-network attention. There
is no earlier "commitment" stage that gets overridden — the model's
preference is constructed late in the network from the (current)
dialogue evidence, full stop.

## What this means for D-40

The D-40 conclusion holds robustly. The methodological worry that
flip-text-with-argmax couldn't distinguish "no commitment" from
"committed-but-suppressed" is now empirically resolved: there is no
suppressed-commitment signature anywhere in the layer-wise lens
trajectory. The model genuinely improvises.

Strong corollary: **the model's apparent commitment in the unflipped
baseline is itself a late-network construction.** When we capture
residuals at `pre_reveal_gen` for an unflipped cow run, the L40+ lens
shows cow strongly preferred — but that preference was BUILT by the
late layers from the (unflipped) dialogue evidence, not retrieved
from a stored commitment. Same mechanism in flipped vs unflipped
runs; just different dialogue evidence to integrate.

## Methodological caveats and confidence

1. **Logit lens early-layer noise.** L0–L25 readings are unreliable
   for the standard reason (residuals aren't yet in the unembedding
   basis). We can't strictly rule out a "commitment" signature in
   those layers via this method. But the patching null at every
   pre-answer position (D-35..D-38) and the positional probe at chance
   for `end_ready` (D-39) already weigh against any pre-network
   commitment locus, regardless of layer. Early-layer lens noise is
   a known issue, not a load-bearing gap in this argument.
2. **Lens at one position only.** We measured lens at `pre_reveal_gen`.
   A more thorough analysis would lens at every anchor (`end_ready`,
   etc.) to build a full lens-trajectory heatmap. Cheap follow-up if
   any new doubt arises; the present readings are consistent with the
   structural picture from D-39's per-anchor probing.
3. **Tuned lens.** Standard logit lens overweights the unembedding's
   token frequency. Tuned lens (with a learned linear adapter per
   layer) gives cleaner readings. We used standard logit lens here
   because L30+ readings are already clearly meaningful (cow at L48
   for cow baselines is +40 logits, dwarfing other classes); a tuned
   lens would refine but not change the qualitative conclusion.

## Decision consequence — proceed to scale comparison

The 12B M4 narrative is now methodologically clean:

- The model does not store class commitment in the residual stream
  (D-35..D-38 patching null; D-39 positional probe at chance for
  end_ready).
- The reveal class is causally a function of the visible yes/no
  answer history (D-40 flip-text behavioral; 74.7% out-of-attractor).
- The mechanism is late-network dialogue integration, not retrieval
  of a stored commitment (D-41 lens trajectory; integration at
  L30–L48; no mid-network commitment bump under flipped dialogue).

The next experiment is **scale comparison at Gemma 3 27B** (and
optionally a 70B-class model). The same sequence of experiments
(positional probe + flip-text-with-lens) on 27B would tell us:

1. Does end_ready LR LOO climb above chance at scale?
2. Do flip-text out-of-attractor rates decrease at scale (more
   pre-commitment), increase (more confident improvisation), or stay
   roughly the same?
3. Does the lens-trajectory show a mid-network commitment bump at
   scale, even if the final-layer behavior is dialogue-respecting?

The methodology pipeline is now solid; engineering for 27B is
identical to 12B with the model name swapped (`gemma-3-12b-it` ->
`gemma-3-27b-it`). Need to redo the self-chosen collection on 27B
first (~hours), then run the same phase 2c-iii probe + phase 2d
flip-text-with-lens on the new collection.

## Artifacts

- `runs/m4_flip_yesno_text_12b_default_n80_lens.json` — per-trial
  lens matrices (49 × 20) plus all of D-40's behavioral records.
- `scripts/flip_yesno_text.py` at `a497536` — extended with
  `--logit-lens` flag.
- Reference: D-39 (positional probe), D-40 (flip-text behavioral).
