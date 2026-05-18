# M5 — Scale-comparison patch sweep at 27B: both anchors null

**Headline:** the 27B class-direction in the residual is **decodable
but not load-bearing** at every anchor we tested. Single-anchor band
patching at the L27-L62 late-band of `pre_answer_q4` flips 3/870
off-diagonal trials toward source (0.34%); patching at the L12-L20
end_ready peak band flips 3/870 (0.34%). Both flip counts are
concentrated on a single fragile target run (`attempt_593`, a
horse-reveal) and from a single source class (cow) — same
baseline-fragility pattern that 12B's [attempt_588](
M4-patch-turn4-band-null.md) showed. **Improvisation is scale-robust.**

This sharpens the [27B positional probe](M5-positional-probe-27b-default-v2.md)
result rather than overturning it. Scale upgrades the **legibility** of
the class commitment (end_ready 1.2× chance at 12B → 3.55× chance at
27B; pa4 3.15× → 4.70×) without upgrading its **causal role**. The
late-network re-derivation mechanism dominates at both scales; what
changes with scale is how visible the model's bookkeeping is to a
linear probe.

## Two experiments, one design

|  | anchor | layer band | rationale | trials |
|---|---|---|---|---:|
| **Exp-A** | `pre_answer_q4` | L27-L62 (36 layers) | Apples-to-apples to 12B M4 phase-2a band patch (L27-L48). 27B LR > 0.50 across this band; `patch_turn4.py` unchanged. | 1024 (870 off-diagonal) |
| **Exp-B** | `end_ready` | L12-L20 (9 layers) | The new 27B-only commitment locus. LR peaks 0.42-0.51 across this band; no analogous signal exists at 12B. Uses new `patch_anchor.py` reading source residuals from v2 capture. | 1024 (870 off-diagonal) |

Both jobs: gpu_1, 27B in bf16, 5 src × 5 tgt per class × 7×7 cells,
~170 s per job after the ~7-min model load. Source/target pools are
disjoint sets within each class (first-5 vs last-5 indices) so no
run patches into itself.

## Result — both anchors null on argmax flips

| metric | 27B Exp-A (pa4 / L27-L62) | 27B Exp-B (end_ready / L12-L20) | 12B M4 pa4 / L27-L48 (reference) |
|---|---:|---:|---:|
| off-diagonal trials | 870 | 870 | 300 |
| **flip-to-src** | **3 / 870 = 0.34%** | **3 / 870 = 0.34%** | 5 / 300 = 1.67% |
| any-change-from-baseline | 18 / 870 = 2.07% | 15 / 870 = 1.72% | 2 / 300 = 0.67% |
| logit-diff delta mean (off-diagonal, excl. shark) | **-0.189** | **-0.010** | (not recomputed here) |
| logit-diff delta median | -0.195 | -0.012 | — |
| logit-diff delta range | [-0.49, +0.09] | [-0.16, +0.20] | — |
| diagonal (self-patch) delta | 0.000 for all | 0.000 for all | 0.000 |

The diagonal sanity check passes at both: self-patch (src and tgt
runs of the same class, different runs) produces zero logit-diff
delta. So the patching machinery is faithful.

**Flip locations:**

- 27B Exp-A: 3 flips, all on `tgt=horse/attempt_593` with src classes
  {cow/003, cow/006, cow/013}.
- 27B Exp-B: 3 flips, all on `tgt=horse/attempt_593` with src classes
  {cow/013, cow/014, cow/030}.
- 12B M4 pa4 L27-L48: 5 flips, all on `tgt=horse/attempt_588` with
  src classes {dog/003, dog/004, dog/007, dog/008, dog/010}.
- 12B M4 pa4 L1-L48 (full residual, every layer patched at pa4): **0/300 flips**.

All three "flip" patterns are concentrated on one fragile horse-tgt
run per scale, with source from a single non-horse class. This is the
same signature as the [known baseline non-determinism](
../../STATUS.md) on `attempt_588`, `attempt_206`, `attempt_038`,
`attempt_049` at 12B — runs whose reveal is sensitive to any small
perturbation, not whose reveal causally depends on the patched
residual. The fact that 12B's strongest-possible intervention (L1-L48
*all-layer* residual replacement at pa4) flips 0/300 reinforces this:
if a causal channel existed at pa4, replacing 48 layers of residual
would expose it.

## Logit-diff delta tells the same story

Off-diagonal logit-diff delta mean is at 27B:

- **Exp-A: -0.19** — the patch pushes logits *away from source* on
  average. Consistent with "replacing 36 layers of late-band residual
  introduces noise that disrupts the model's classification without
  systematically routing it toward source's class".
- **Exp-B: -0.01** — essentially zero. The end_ready patch produces no
  consistent signed effect on the reveal logit difference, neither
  toward source nor away.

The shark column was excluded from these means because shark has only
2 baseline runs at 27B; in the raw matrix the shark column shows
+3 to +4 deltas, an artifact of the small denominator (baseline
logit[src]-logit[shark] is very negative; tiny absolute changes
inflate the delta).

## What this means for the M5 / blog story

The 12B story was "no Ready commitment, decodable but inert turn-4
signal, re-derivation across dialogue turns". The new 27B numbers
extend the story without changing its shape:

1. **Decoding shifts dramatically with scale.** Every anchor
   strengthens by +1.6 to +2.4× chance from 12B to 27B; end_ready and
   end_user_prompt go from at-chance to clearly decodable.
2. **Causal patching does *not* shift with scale.** At 27B's strongest
   decoding anchors (end_ready / L12-L20 peak; pa4 / L27-L62 carry
   band), single-anchor band patching produces the same null as 12B.
   Both anchors at 27B flip 3/870 trials on a single fragile target,
   matching the 12B pattern at attempt_588.
3. **The right way to describe the scale effect is:** scale rotates
   class information into more linearly-readable subspaces of the
   residual stream (the L12-L20 mid-network band is now legible where
   it wasn't at 12B), but the *use* of that information by the
   model's downstream computation remains improvisational —
   re-derived at every step from the accumulated dialogue, not read
   off a stored commitment.

This is also why our M5 SAE-negative result fits cleanly: at 12B the
class direction wasn't sparsely encoded *because there isn't a stored
class feature to sparsify*; the legible direction is a by-product of
the dialogue-integration step. At 27B, more of that by-product is
linearly visible, but it's still a by-product.

## Caveats

- **Single-anchor band patching is one of several possible
  interventions.** Multi-anchor simultaneous patching (e.g., end_ready
  AND end_model_q1..q4 simultaneously, blocking every "re-derivation"
  site at once) was not tested. If the model re-derives the class at
  every turn from the accumulated yes/no history, blocking only one
  anchor is naturally insufficient — the next turn re-derives from
  what's still legible. Worth running before the M5 writeup goes to
  blog.
- **The L1-Ln full-residual experiment was only done at 12B.** At
  27B we didn't yet run the L1-L62 full-residual replacement at any
  anchor. Predicted outcome based on the band null: 0 flips, but
  worth confirming.
- **Shark and gorilla are underpowered.** Shark has 2 baseline runs;
  gorilla has 5 at 27B (vs 20 in the probe's LR LOO). The "scale
  collapses output entropy to 6 attractors" finding from M4
  (`docs/progress/M4-comparative-prompt-and-scale.md` §Scale axis)
  means the 27B run distribution itself is skewed, and the smaller
  classes contribute little statistical weight to the off-diagonal
  flip count. The fragile-tgt pattern is in horse — the most abundant
  class — which is reassuring.
- **`attempt_593` baseline check pending.** We should replay
  attempt_593 several times to confirm its baseline is non-deterministic
  (same way attempt_588 was at 12B). If so, the 3 "flips" are noise
  amplification, not signal.

## STATUS.md correction

STATUS.md previously claimed "Single-position residual patching:
**0/2280 flips** across {L29, L27-L48, L1-L48} × {turns 1-4}." That
number is wrong. The correct breakdown of the 12B M4 patch jobs:

- pa4 / L29 single layer: 5/300 flips (all `tgt=horse/attempt_588`)
- pa4 / L27-L48 band: 5/300 flips (all same target/source pattern)
- pa4 / L1-L48 every-layer: 0/300 flips
- turn-sweep / L1-L48: not re-tallied here (separate JSON).

The "0/2280" appears to have collated the *zero outcome on the
strongest intervention* (L1-L48 every-layer) and rounded the others
to zero. The correct framing is: "every single-anchor band patch at
12B M4 produced flips only on a known-non-deterministic target run;
the strongest residual-replacement intervention (L1-L48 every-layer
at pa4) produced 0/300". The same picture holds at 27B.

## Artifacts

- `runs/m5_patch_27b_default_pa4_L27-62.json` (Exp-A, 472 KB)
- `runs/m5_patch_27b_default_endready_L12-20.json` (Exp-B, 472 KB)
- `runs/m5_patch_27b_default_endready_smoke.json` (smoke test, 39 KB)
- `scripts/patch_anchor.py` (new, generalises patch_turn4.py to
  arbitrary v2 anchors)
- TSUBAME jobs: 7654936 (Exp-A, gpu_1, ~170 s patched-trial wall
  after model load), 7654937 (Exp-B, same), 7654468 (smoke).
- Reference baseline: 12B `runs/m4_patch_turn4_12b_default_L27-48band.json`
  (5/300 flips on attempt_588) and `m4_patch_turn4_12b_default_L1-48all.json`
  (0/300 flips, every-layer at pa4).

## Next

- **Multi-anchor patching** at 27B: end_ready + pa4 simultaneously,
  same layer bands. Tests the "block every re-derivation site at
  once" version of the load-bearing question. ~15 min on gpu_1
  reusing patch_anchor.py with a small extension to accept multiple
  `--anchor` values.
- **Full-residual replacement at 27B**: L1-L62 every-layer at
  end_ready and at pa4 separately. Tests the strongest-possible
  single-anchor intervention. Predicted null based on 12B's L1-L48
  result.
- **Replay `attempt_593`** under the same kwargs to verify baseline
  non-determinism (mirrors the attempt_588 investigation flagged in
  STATUS). If baseline-fragile, the 3/870 "flips" are confirmed
  noise.
- **Update the 27B positional-probe writeup** to point forward to
  this null and reframe the headline from "scale-shifts-to-retrieval"
  to "scale-shifts-legibility, not causality".
