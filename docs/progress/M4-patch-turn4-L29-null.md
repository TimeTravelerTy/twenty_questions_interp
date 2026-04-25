# M4 phase 1 — Single-layer L29 turn-4 pre-answer patch is null

**Run date:** 2026-04-25 by Claude (Opus 4.7).
**Source collection:** `runs/diag/selfchosen_ready_20bank_12b_turn4scale_20260421/`
(M3 default-prompt n=80 scale-up; balanced 4 classes — `cow`, `dog`,
`elephant`, `horse` — 20 runs/class).
**Patch script:** `scripts/patch_turn4.py` at commit `9b33a77`.
**Job:** `tq_m4_patch_turn4_12b_20260424.sh` (gpu_h MIG slice).
**Machine-readable report:** `runs/m4_patch_turn4_12b_default_L29.json`.

## Scope

First M4 causal patching experiment. Replace the target run's residual
stream output at **layer 29, turn-4 pre-answer token position** with the
source run's saved L29 residual (from `turn_04_activations.pt`), then
generate the reveal turn greedily and parse the canonical animal name.

- Single layer (L29), single position (last pre-answer token).
- 5 src runs × 5 tgt runs per class × 4×4 = 16 cells = 400 patched
  trials, plus 20 no-patch baselines.
- L29 chosen from the M3 turn-4 scale-up: LR LOO 0.79 @ L31 / NC 0.66 @
  L29 on the same n=80 collection (3.2× chance).

## Result — null

Flip-to-source rate matrix (row = src class, col = tgt class):

```
  src\tgt |      cow |      dog | elephant |    horse
  ----------------------------------------------------
      cow |   100.0% |     0.0% |     0.0% |    20.0%
      dog |     0.0% |   100.0% |     0.0% |     0.0%
 elephant |     0.0% |     0.0% |   100.0% |     0.0%
    horse |     0.0% |     0.0% |     0.0% |    88.0%
```

Diagonals are at ~100% (self-patch is a no-op, so reveals match the
unpatched baseline). Off-diagonals are at 0% with a single exception:
`cow→horse` shows a 20% flip-to-cow rate.

That 20% is **all on a single target run, `attempt_588`**, whose no-patch
baseline already returns `cow` instead of its on-disk reveal of `horse`.
The other 4 horse targets are completely unmoved by any source. The
horse diagonal is also 88% rather than 100% from the same `attempt_588`
instability — at greedy `do_sample=False`, this run is non-deterministic
across forward-pass replays. So the only "effect" the patch produces is
on a target whose reveal is already non-deterministic without any
intervention.

**Conclusion:** at L29, single-position, the patch does not flip reveals
on any of the 19 deterministic targets across any of the 16 source/target
pairs. The probe-decodable signal at L29 (LR 0.79) does not translate
into causal control of the reveal at this granularity.

## Interpretation

The literature on activation patching has documented this kind of
dissociation extensively. **Decodability ≠ causality** (Heimersheim &
Nanda 2024, "How to use and interpret activation patching", arXiv
2404.15255; "Causality ≠ Decodability, and Vice Versa" arXiv
2510.09794): a probe finding information *present* in the residual
stream tells you that information *exists* there, not that the model
*uses* it for the downstream behavior we are scoring.

Two interpretations of the null at L29 are both interesting:

1. **Intervention too narrow (redundancy across layers/positions).**
   L29 carries class info at the pre-answer position, but the secret is
   also encoded redundantly across nearby positions and/or layers.
   Replacing one residual is washed out by the rest of the residual
   stream. Heimersheim & Nanda explicitly flag this as a common failure
   mode and recommend starting at *low* granularity (broad
   layer/position bands) before refining.
2. **Decodable ≠ causal (off-causal-path representation).** L29 holds a
   class signature, but the reveal token's logits are driven by other
   layers (e.g., very-late layers ~L48 directly write to the unembedding
   without funneling through L29) and L29 simply isn't on the causal
   path of the reveal. A layer sweep could in principle distinguish
   this, but if interpretation (1) is correct, every single-layer
   single-position sweep will null at every layer too, so a sweep alone
   is non-diagnostic.

## Methodological caveat: argmax is a coarse instrument

The current metric is greedy-decoded canonical match. On a 4-class
panel against a strong attractor (cow/horse are cosine-close in name
space; both are mid-frequency 4-letter animals), a patch could be
moving the reveal logit 30% of the way toward src without flipping the
argmax, and we'd never see it. Heimersheim & Nanda 2024 strongly
recommend **logit difference** as the primary patching metric:
continuous, linear in residual-stream contributions, and sensitive to
partial effects.

`patch_turn4.py` has been extended to capture first-step generation
logits and report `logit[src_first_tok] - logit[tgt_first_tok]` per
trial alongside the categorical flip rate (commit forthcoming, see
phase 2a). The L29 null reported here is on argmax flip-rate only;
revisiting the same patches under logit-diff is the cheapest sensitivity
check available.

## Decision consequence — phase 2a

Skip a bare layer sweep at single-position. It only distinguishes
interpretation (2) from (1) if the answer is (2); a layer-sweep null
gives no extra evidence for (1) over (2).

Instead, broaden along the residual stream as the next experiment, in
line with the published "low-granularity-first" recommendation:

1. **Patch a layer band** L27–L48 simultaneously at the same single
   pre-answer position. This forces the entire late-layer band to be
   src's representation. If even *that* doesn't flip reveals, both
   interpretation (1)-by-layer and interpretation (2) for the
   late-layer band are ruled out and the probe-decodable signal is
   genuinely off the reveal-token causal path. If it *does* flip, the
   locus is real but redundantly encoded across layers, and we narrow
   back down via a layer sub-sweep to find a minimal sufficient site.
2. **Logit-diff metric** added in parallel. Even if the argmax doesn't
   flip on phase 2a, a positive `logit[src] - logit[tgt]` shift relative
   to baseline is informative.

Phase 2b (deferred, conditional): if 2a is also null, re-collect source
activations across a window of turn-4 positions (not just the
pre-answer token) and patch a position-band as well as a layer-band.

## Side issue — `attempt_588` non-determinism at greedy

At `do_sample=False, dtype=bfloat16`, the saved on-disk reveal for
`attempt_588` is `horse`, but baseline replay produces `cow` 5/5 times.
Same prompt, same model, same generation kwargs, different output. This
is not a patching artifact — it's a generation-stack reproducibility
issue (KV cache, attention impl, or BLAS nondeterminism in bfloat16).
Worth a short investigation before relying on horse trials in any future
patch round; non-blocking for phase 2a since the band patch is
diagonal-symmetric and 4 of 5 horse targets are stable.

## Artifacts

- `runs/m4_patch_turn4_12b_default_L29.json` — full per-trial records
  + per-cell summary (400 patched trials + 20 baselines).
- `scripts/patch_turn4.py` — script (extended for phase 2a).
- Source activations: `runs/diag/selfchosen_ready_20bank_12b_turn4scale_20260421/attempt_*/turn_04_activations.pt`.
- Reference: Heimersheim & Nanda 2024, *How to use and interpret
  activation patching* (arXiv 2404.15255).
