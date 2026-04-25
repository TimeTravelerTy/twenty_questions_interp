# M4 phase 2a — L27-L48 layer band patch at turn-4 pre-answer is also null

**Run date:** 2026-04-25 by Claude (Opus 4.7).
**Source collection:** `runs/diag/selfchosen_ready_20bank_12b_turn4scale_20260421/`
(M3 default-prompt n=80 scale-up; balanced 4 classes — `cow`, `dog`,
`elephant`, `horse` — 20 runs/class).
**Patch script:** `scripts/patch_turn4.py` at commit `0a4913c`.
**Job:** `tq_m4_patch_turn4_12b_band_20260425.sh` (gpu_h MIG slice, job
`7260288`, ~2 minutes wall after model load).
**Machine-readable report:** `runs/m4_patch_turn4_12b_default_L27-48band.json`.

## Scope

Phase 2a follow-up to the L29 single-layer null (D-35). Same target
collection, same trial design (5 src × 5 tgt per class, 16 cells, 400
patched trials + 20 baselines). The change: instead of patching one
layer, patch **all 22 layers from L27 through L48 simultaneously** at
the same single pre-answer position. Forces the entire late-layer band
to be src's representation. This is the literature-canonical "broaden
first" intervention from Heimersheim & Nanda 2024
(arXiv 2404.15255 §4).

Also added in this phase: continuous **logit-difference metric** at the
first reveal-generation step. For each (src, tgt) cell, capture
`logit[src_first_tok] - logit[tgt_first_tok]` averaged across trials,
with per-target-run baseline subtraction. First-token ids: cow=26107,
dog=21871, elephant=94629, horse=34741 (` Cow`, ` Dog`, ` Elephant`,
` Horse` under the Gemma SentencePiece tokenizer).

## Result — null on both metrics

### Argmax flip-rate

```
  src\tgt |      cow |      dog | elephant |    horse
  ---------------------------------------------------
      cow |   100.0% |     0.0% |     0.0% |    20.0%
      dog |     0.0% |   100.0% |     0.0% |     0.0%
 elephant |     0.0% |     0.0% |   100.0% |     0.0%
    horse |     0.0% |     0.0% |     0.0% |    84.0%
```

Identical to phase 1 to within stochastic noise: diagonals 100%
(self-patch is a no-op), off-diagonals 0% across 19 of 20 deterministic
target runs. The only "effect" is again `cow→horse` at 5/25 = 20%, all
on `attempt_588` (the same non-deterministic horse target identified in
D-35). The horse diagonal drops from 88% (phase 1) to 84% (phase 2a) —
sub-trial noise, not a signal.

### Logit-difference deltas (patched − baseline)

```
  src\tgt |      cow |      dog | elephant |    horse
  ---------------------------------------------------
      cow |    +0.00 |    +0.03 |    -0.04 |    -0.06
      dog |    +0.00 |    +0.00 |    -0.10 |    -0.09
 elephant |    -0.04 |    -0.11 |    +0.00 |    -0.07
    horse |    +0.08 |    +0.08 |    -0.12 |    +0.00
```

Diagonals are exactly 0 (correct: self-patch is identity). Off-diagonal
deltas are all within ±0.12 logits.

### Magnitude check

The natural baseline `logit[gt_class] - logit[next_best_class]` margins
across the 20 target runs span:

| target class | mean margin | range |
|---|---:|---|
| cow | +5.0 | +2.0 to +10.3 |
| elephant | +3.3 | +1.8 to +5.6 |
| horse | +4.1 | -1.0 (att 588) to +10.0 |
| dog | (metric unreliable, see caveat) | — |

So a typical baseline gt-vs-other separation is **3 to 10 logits**.
The patch moves logits by **0.05 to 0.12 logits** — i.e. **1-3% of the
natural gap**. The signs are also not uniformly toward src; e.g.
`cow→horse` is *negative* (the patch nudges *away* from cow), which is
consistent with random fluctuation, not a partial causal effect.

Replacing the entire L27-L48 band with src's representation moves the
reveal logits by less than 3% of the natural class separation. This is
a clean null on the continuous metric, not just on argmax.

### Methodological caveat — dog logit-diff is unreliable

For dog targets the baseline margin `logit[" Dog"] - max_other_class`
is **negative** (mean -1.9), even though the parsed reveal is `dog` 100%
of the time. This means dog's revealed answer doesn't begin with the
` Dog` token at the first generation step — likely the model says
something like "I was thinking of a dog" so step 0 is `I` and the
animal-name token sits at step 4-5. The first-step logit-diff metric
therefore captures noise on the dog row, not the dog signal. Treat dog
deltas as unreliable; cow/elephant/horse rows are the reliable
measurements.

This is fixable in phase 2b by either (a) generating a few tokens
ahead and finding the position where the animal token first becomes
argmax, or (b) using log-prob of the entire `' <Animal>'` substring
across positions. Not a blocker for the qualitative null conclusion,
which already holds on the cow/elephant/horse rows.

## Interpretation

Both the L29 null (D-35) and now the L27-L48 band null taken together
rule out the "narrow late-layer locus is causal" picture in either of
its forms:

- It is *not* the case that the late-layer signal is in some single
  layer that L29 missed (the band covers everything from L27 to L48).
- It is *not* the case that redundancy across the late-layer band is
  hiding a real causal locus that single-layer patching couldn't reach
  (the band patches all 22 layers simultaneously and still nothing
  moves).

So the probe-decodable signal at L29-L48 (M3 turn-4 LR LOO 0.79) is
genuinely **off the reveal-token causal path** at the pre-answer
position. The class identity is *legible* in those layers without
*driving* the reveal that follows.

This leaves two structural hypotheses, in increasing scope:

1. **Earlier-layer locus.** The class info is encoded in earlier layers
   (L1-L26) at the same pre-answer position, and the late layers
   carry it forward as an inert echo. A complementary band patch on
   L1-L26 — or, more decisively, on L1-L48 — would test this.
2. **Other-position locus.** The pre-answer position is not the
   bottleneck at all. The reveal token is computed via attention to
   *other* token positions in the dialogue (e.g., the secret-choice
   commit point in turn 0, or the cumulative yes/no answer pattern
   across turns 1-4). Single-position patching would null at every
   layer in this case. A position-band patch would be needed to test it.

The cheapest single experiment that distinguishes these is **all-layer
single-position patch (L1-L48)**. If null, the position itself is not
the bottleneck → hypothesis (2). If positive, narrow back to find the
sufficient layer-band → hypothesis (1).

This is the same "broaden then narrow" recipe, taken one step further
along the layer axis.

## Decision consequence — phase 2b

1. **Run the all-layers single-position patch** (`--layers 1,2,...,48`)
   as the next experiment. ~2 min walltime on the same gpu_h slice. No
   new activation captures needed; existing `turn_04_activations.pt`
   covers all layers.
2. **Improve the logit-diff metric for dog** by capturing logits at
   several generation steps and finding the position where any
   animal-name token first becomes argmax-favored. Defer to whichever
   phase next produces a positive signal.
3. **If 2b is also null**, the conclusion graduates: no single-position
   patch on the turn-4 pre-answer token can move the reveal, regardless
   of layer. Phase 2c then expands to a *position* band (last K
   pre-answer tokens, or the full turn-4 user message), which requires
   re-collecting source activations across positions.

The scientific narrative starts to crystallize: the M3 probe-decodable
signal is real but the model's mechanism for committing to a class at
reveal time does not flow through that single-position late-layer
representation. This is itself a positive result about the architecture
of self-chosen commitment, even with a chain of negative patching
results — and it matters for the "early decision vs late
crystallization" question we want to ask at 27B+.

## Artifacts

- `runs/m4_patch_turn4_12b_default_L27-48band.json` — full per-trial
  records + per-cell summary with both flip-rate and logit-diff metrics
  (400 patched + 20 baselines).
- `scripts/patch_turn4.py` at `0a4913c` — extended for layer band and
  logit capture.
- Reference: Heimersheim & Nanda 2024, *How to use and interpret
  activation patching* (arXiv 2404.15255).
