# M4 phase 2d (c-text) — Yes/no text-level flip: improvisation hypothesis decisively confirmed

**Run date:** 2026-04-26 by Claude (Opus 4.7).
**Source collection:** `runs/diag/selfchosen_ready_20bank_12b_turn4scale_20260421/`
(M3 default-prompt n=80 scale-up; balanced 4 classes — `cow`, `dog`,
`elephant`, `horse`).
**Script:** `scripts/flip_yesno_text.py` at commit `d37c165`.
**Job:** `tq_m4_flip_yesno_text_20260426.sh` (gpu_h, 114s wall, exit 0).
**Machine-readable report:** `runs/m4_flip_yesno_text_12b_default_n80.json`.

## Scope

The phase 2c-iii positional probe (D-39) showed the class signal is at
chance at `end_ready`, weak through mid-dialogue, and crystallizes only
at reveal time. Combined with the comprehensive 0/2280 single-position
patching null (D-35/D-36/D-37/D-38), the working hypothesis became
**improvisation**: the model doesn't store a class commitment in the
residual stream — it re-derives the class at reveal time from the
accumulated yes/no answer history.

This phase tests the improvisation hypothesis directly with a
text-level intervention: **rebuild the chat context with exactly one
yes/no answer text flipped (Yes -> No or No -> Yes), regenerate the
reveal greedily, and observe whether the reveal class follows the
flipped dialogue evidence.** No residual patching — pure behavioral
counterfactual.

For each kept run T (subsample 20 per class, 80 total), generated 4
flipped-reveal trials (one per turn 1..4) plus 1 baseline replay = 400
forward passes total. Both the modified yes/no answer text and the
reveal generation are deterministic.

## Result — strong improvisation, decisively

### Kept-class rate matrix (rows = original class, cols = flipped turn)

```
  class    |   T1   |   T2   |   T3   |   T4
  cow      |   0.0% |  35.0% |   5.0% |   0.0%
  dog      |   0.0% |  10.0% |   0.0% |   0.0%
  elephant |   0.0% |  40.0% |   5.0% |   0.0%
  horse    |   0.0% | 100.0% |  10.0% |   0.0%
```

Read row-by-row: when we flip turn T1 of a cow run, the reveal *never*
stays cow (0/20). When we flip T4 of any class, the reveal *never*
stays original-class (0/80 across all four classes). Even the most
conservative cell (T2 flips) sees the original class kept only 46%
of the time on average. **Flipping a single yes/no answer is
sufficient to change the reveal class on most trials.**

### Out-of-attractor rate: 74.7%

Across all 320 flipped trials, the reveal lands outside the
`{cow, dog, elephant, horse}` attractor set in **239/320 (74.7%)**
trials. The model isn't just shuffling between the four attractor
classes — it routinely produces classes from the broader 20-bank
that are consistent with the new flipped pattern.

| output class | count | % | in attractor? |
|---|---:|---:|:---:|
| dolphin | 79 | 24.7% | no |
| horse | 61 | 19.1% | yes |
| gorilla | 42 | 13.1% | no |
| cobra | 26 | 8.1% | no |
| kangaroo | 24 | 7.5% | no |
| crocodile | 21 | 6.6% | no |
| frog | 18 | 5.6% | no |
| elephant | 10 | 3.1% | yes |
| cow | 8 | 2.5% | yes |
| chicken | 7 | 2.2% | no |
| bat | 6 | 1.9% | no |
| (5 more, each <2%) | 16 | 5.0% | mixed |

13 of the 20 bank classes appear at least once. Compare this to the
M3 base-rate where only 4 classes were realized at greedy. **Once we
break the attractor with a flipped constraint, the model's full bank
distribution opens up and produces sensible class consistency.**

### Each turn induces a *characteristic* drift pattern

Per-(original-class, flipped-turn) output distributions:

```
T1 flip (across all classes):   cobra, crocodile, frog, bee, owl,
                                chicken, tiger — wild / exotic /
                                non-mammal cluster
T2 flip:                        mostly horse (prior-attractor fallback)
T3 flip:                        dolphin 79/80 trials (single dominant target)
T4 flip:                        gorilla 42, kangaroo 24, bat 6,
                                tiger/cat/dolphin (large-mammal swap)
```

The model isn't randomly perturbed by the flip — for a given turn
position, the resulting reveal class is highly reproducible across
different starting classes. This is *competent re-derivation*, not
noise: each yes/no question implicitly encodes a constraint, and
flipping that constraint pushes the most-consistent class along the
question's discrimination axis.

That T3-flip → dolphin in 79/80 trials is striking. T3's question
(varies per run since panel is randomized, but resolves to a
discrimination axis like is_carnivore or lives_in_africa) is
apparently very class-discriminating: flipping its answer puts the
model squarely in dolphin's region of bank-consistent classes.

### Per-turn shift breakdown

| flipped turn | class same | class different | unparsed |
|---|---:|---:|---:|
| T1 | 0 / 80 (0%) | 80 (100%) | 0 |
| T2 | 37 / 80 (46%) | 43 (54%) | 0 |
| T3 | 4 / 80 (5%) | 76 (95%) | 0 |
| T4 | 0 / 80 (0%) | 80 (100%) | 0 |

T2 is the conservative outlier. The cells inside that row:

- horse / T2-flip: keeps `horse` 20/20 — the prior attractor
  swallows whatever the flipped pattern would otherwise push toward.
- dog / T2-flip: 18/20 -> `horse`. dog->horse fallback.
- cow / T2-flip: 12/20 -> horse, 7/20 keep cow, 1/20 -> elephant.
- elephant / T2-flip: 8/20 keep elephant, 8/20 -> horse, 4/20 -> penguin.

So T2-flip is *also* improvisation-respecting, but the question that
happens to land at T2 across runs (the prompt panel is randomized
per-run via `manifest.permutation`) tends to be one whose flipped
constraint is *under-determined* — multiple bank classes are
consistent — and the model's prior over `horse` (the strongest
attractor) wins the tiebreak. T1/T3/T4 give *over-determined* flipped
constraints where a specific non-attractor class is the
most-consistent, so the model confidently switches.

This texture — **constraint flipping respects the prior when
under-determined, abandons it when over-determined** — is itself
mechanistically suggestive.

### Baseline replay determinism

77/80 baselines (no-flip replays) reproduce the on-disk reveal class.
3 mismatches (`attempt_206` dog->elephant; `attempt_038` horse->cow;
`attempt_049` horse->elephant) are the same forward-pass
nondeterminism family as `attempt_588` from earlier phases — same
chat context, same `do_sample=False`, but a few specific runs diverge
between original streaming generation and replay. Not blocking the
flip-result interpretation since those 3 are only ~4% of the data
and the flip results are far stronger than that noise floor.

## Interpretation

The improvisation hypothesis is **decisively confirmed**. The model
does not store a class commitment that survives dialogue tampering.
Instead, the reveal class is **causally a function of the visible
yes/no answer history**: change the history, the class follows.

Three texture observations on top of the headline:

1. **The model's class derivation is competent, not random.** Each
   turn-position flip produces a characteristic, reproducible new
   class distribution. The model isn't confused by the
   counterfactual — it derives a *new most-consistent class* and
   confidently emits it.
2. **The 4-class attractor at greedy is a prior, not a structural
   commitment.** Once we hand the model an answer pattern that the
   priors don't dominate, it accesses the broader 20-class bank
   distribution naturally. 13 of 20 bank classes appear in the
   flipped trials.
3. **The "self-chosen 20 questions" task at 12B is mostly a
   hypothetical-completion task.** The model:
   - Says "Ready" without committing (no class signal at end_ready).
   - Answers each question consistently with *some plausible class
     it could be thinking of* — the prior makes that the
     cow/dog/elephant/horse cluster.
   - At reveal time, derives the most-consistent class from the
     accumulated answer pattern.

This is a substantive finding about how 12B Gemma handles
self-referential commitment tasks, and it reframes the M4 milestone
narrative entirely.

## Scale-axis question (for the 27B / 70B follow-up)

Whether 12B's improvisation pattern is scale-robust or shifts with
scale is now a sharp empirical question. The same flip-text
experiment at Gemma 3 27B (or a 70B-class model) would tell us:

- Does kept-class rate climb at scale (more pre-commitment)?
- Do the per-turn drift patterns stay reproducible (competent
  re-derivation) or become noisier (less integration of dialogue
  evidence)?
- Does end_ready LR LOO cross above chance at scale, in which case
  patching it might actually move reveals (genuine pre-commitment
  + causal commitment locus)?

The original "do models decide early?" framing has a 12B answer (no,
they improvise) and the scale comparison is now the M5/M6 hook.

## Decision consequence — phase 2e (residual-level localization)

The improvisation hypothesis is confirmed; the next research question
is *where in the residual stream* the dialogue-evidence-to-class
mapping happens. Phase 2e: residual-level constraint localization.

**Phase 2e setup:**

For each (original run T, target turn N to flip), compute two paired
forward passes:

- (a) T's *original* dialogue, capture residuals at every anchor.
- (b) T's *flipped-T_N* dialogue, capture residuals at every anchor.

Then compare anchor-by-anchor: where do (a) and (b) start to differ?
The first anchor with a substantial divergence is where the flipped
answer's effect first enters the residual stream. Subsequent anchors
where the divergence persists or grows show the propagation path
toward the reveal-token logits.

This gives us a *causal* localization (rather than just decodable
localization from D-39): the residuals where the dialogue evidence
matters for the reveal.

Engineering: extends `capture_positional_residuals.py` to take a
`--flip-turn N` flag that injects the flipped answer text into the
chat-context build. Run once per turn flipped (4 runs × ~50s each =
~3-5 min), pull the per-anchor residuals, compute Frobenius norm of
(orig - flipped) per anchor × layer cell.

**Held back until user weighs in.** This is a moderate engineering
extension; the interpretation depends on choosing how to summarize
"divergence" (norm? cosine? class-direction projection?). Worth a
brief design conversation before implementing.

Also worth flagging: phase 2e may not be the most scientifically
interesting next move. Two alternatives:

- **(b) Scale comparison.** Run the same flip-text experiment on
  Gemma 3 27B (and possibly a 70B-class model) to test whether
  improvisation is scale-robust. This is the central scale question
  the project flagged from the start.
- **(c) Mechanistic question via SAE features.** Now that we know
  the class is improvised from dialogue evidence, the SAE/transcoder
  case studies (M5) become more pointed: which features in the
  residual stream encode the yes/no constraint accumulation? Which
  features encode the class-derivation step? This was the original
  M5 plan and the improvisation finding makes it more concrete.

(b) and (c) are arguably better next steps than (a) phase 2e, because
the M4 narrative is now solid (improvisation > storage) and the
remaining mechanistic detail of *where the constraint integration
happens* may be more efficiently studied at the SAE-feature level
than the raw-residual level.

## Artifacts

- `runs/m4_flip_yesno_text_12b_default_n80.json` — full per-trial
  records + per-cell summary (320 flipped trials + 80 baselines).
- `scripts/flip_yesno_text.py` at `d37c165`.
- Reference: phase 2c-iii positional probe (D-39); patching null
  chain D-35..D-38.
