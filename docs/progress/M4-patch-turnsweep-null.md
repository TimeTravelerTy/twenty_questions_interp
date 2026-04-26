# M4 phase 2c-i — Per-turn pre-answer all-layer patch sweep is null at every turn

**Run date:** 2026-04-26 by Claude (Opus 4.7).
**Source collection:** `runs/diag/selfchosen_ready_20bank_12b_turn4scale_20260421/`
(M3 default-prompt n=80 scale-up; balanced 4 classes — `cow`, `dog`,
`elephant`, `horse` — 20 runs/class).
**Patch script:** `scripts/patch_turn4.py` at commit `b27d39e`
(extended with `--turn N` flag, 1..4).
**Job:** `tq_m4_patch_turnsweep_12b_20260425.sh` (gpu_h MIG slice,
job `7260593`; queued ~80 min, ran ~7 min).
**Outputs:**
- `runs/m4_patch_turn1_12b_default_L1-48all.json`
- `runs/m4_patch_turn2_12b_default_L1-48all.json`
- `runs/m4_patch_turn3_12b_default_L1-48all.json`
- (turn 4 already at `runs/m4_patch_turn4_12b_default_L1-48all.json` from D-37)

## Scope

Cheapest informative follow-up to phase 2b (D-37). Phases 1/2a/2b
established that no layer scope ({L29, L27-L48, L1-L48}) at the
**turn-4** pre-answer position flips reveals. Phase 2c-i widens the
search across other turn pre-answer positions while keeping the layer
scope at maximum (L1-L48, all 48 decoder layers).

The collection already saved per-turn activations
(`turn_01_activations.pt` through `turn_04_activations.pt`) at every
attempt. Each is a `(49, 3840)` tensor capturing the residual stream
at *that* turn's pre-answer position, all layers. So per-turn patching
is free — just point at a different file. Identical 16-cell trial
design as before (5 src × 5 tgt per class, 400 trials + 20 baselines
per turn).

## Result — null at every turn on stable targets

### Per-turn flip-to-source matrices (argmax)

```
TURN 1 (L1-L48, n=400)            TURN 2 (L1-L48, n=400)
  src\tgt | cow  dog  ele  hor      src\tgt | cow  dog  ele  hor
      cow | 88%   0    0   20%          cow |100    0    0   12%
      dog |  0  100    0    0           dog |  0  100    0    0
 elephant |  0    0  100    0      elephant |  0    0  100    0
    horse |  0    0    0   88%         horse |  0    0    0   96%

TURN 3 (L1-L48, n=400)            TURN 4 (L1-L48, n=400; phase 2b)
  src\tgt | cow  dog  ele  hor      src\tgt | cow  dog  ele  hor
      cow |100    0    0   12%          cow | 92    0    0    0
      dog |  0  100    0    0           dog |  0  100    0    0
 elephant |  0    0  100    0      elephant |  0    0  100    0
    horse |  4    0    0   84%         horse |  0    0    0  100
```

Per-turn logit-diff delta ranges (patched − baseline,
`logit[src_first_tok] − logit[tgt_first_tok]`):

| turn | min delta | max delta | mean |abs delta| |
|---|---:|---:|---:|
| 1 | -0.16 | +0.19 | 0.07 |
| 2 | -0.10 | +0.15 | 0.07 |
| 3 | -0.19 | +0.12 | 0.07 |
| 4 | -0.46 | +0.23 | 0.16 |

Natural baseline `logit[gt] − logit[next_best]` margins span +2 to
+10 logits across cow/elephant/horse targets. So every turn's
maximum patched delta is <10% of the natural separation.

### Stable-vs-unstable target accounting

Across the entire 4-turn sweep, the only baseline-non-deterministic
target is **`attempt_588`** (horse class, baseline always returns
`cow` instead of `horse` at greedy `do_sample=False`). The other 19
target runs reproduce their on-disk reveals exactly under all four
turn baselines.

Counting non-tgt-preserving outputs on the 19 *stable* targets only
(i.e. excluding `attempt_588`):

| turn | non-tgt outputs on stable targets | flip-to-src on stable targets |
|---|---:|---:|
| 1 | 2 / 300 | **0** |
| 2 | 4 / 300 | **0** |
| 3 | 1 / 300 | 1 |
| 4 | 3 / 300 | **0** |

Total: **1 flip-to-src across 1200 patched trials on stable targets.**
That single trial is `horse → cow via src=attempt_023 into
tgt=attempt_581 output='horse'` at turn 3.

### Why even the 1 flip-to-src in turn 3 is not a real signal

`attempt_581` is the cow target whose perturbation-fallback is
`horse`: across the entire sweep, **9 out of 60 patched trials** into
`attempt_581` (15%) tip the reveal to `horse`, regardless of src
class:

- Turn 1: `dog→cow into 581` produces `horse` 1/5 times
- Turn 2: `dog→cow into 581` produces `horse` 2/5; `elephant→cow into 581` produces `horse` 2/5
- Turn 3: `horse→cow into 581` produces `horse` 1/5
- Turn 4: `dog→cow into 581` produces `horse` 2/5; `elephant→cow into 581` produces `horse` 1/5

The single Turn-3 case where this same perturbation-flip happens to
align with src (because src is horse and the fallback is horse) is
indistinguishable from the other 8 cases where src is dog or elephant.
It's the model's **prior attractor** showing through under residual
disruption, not an injected src class signal.

This is the same horse/cow attractor leakage documented in phase 2b
(D-37) and at less-obvious-prompt scale (D-34): cow + horse hold ~71%
of the model's prior mass, and any sufficiently strong residual-stream
disruption tips reveals toward those classes regardless of what src
actually contains.

**On the 19 stable targets, across 1200 patched trials sampling 4
turn pre-answer positions × 16 (src,tgt) class cells, there is not a
single genuine flip-to-source.**

## Interpretation — comprehensive single-position null

Phases 1, 2a, 2b, and 2c-i now span:

| phase | turn(s) | layers | flip-to-src on stable targets |
|---|---|---|---:|
| 1 | 4 | L29 | 0 / 380 |
| 2a | 4 | L27-L48 (22) | 0 / 380 |
| 2b | 4 | L1-L48 (48) | 0 / 380 |
| 2c-i | 1, 2, 3 | L1-L48 (48) | 0 / 1140 (Turn-3 case is attractor leakage) |
| **total** | **all** | **all** | **0 / 2280** |

**No single-position patch at any turn pre-answer position, at any
layer scope, flips reveals on stable targets.** The class-decodable
signal at turn-4 pre-answer L29-L48 (M3 LR LOO 0.79, ~3.2× chance) is
*legible* there but **decisively off the reveal-token causal path**.

The mechanism the model uses to commit to a class at reveal time does
not flow through any single pre-answer position residual, regardless
of which turn you choose. The reveal must compute via attention to
*other* positions in the dialogue prefix.

This matches the pattern documented in [Heimersheim & Nanda 2024](https://arxiv.org/abs/2404.15255)
and ["Causality ≠ Decodability"](https://arxiv.org/html/2510.09794):
probe-decodability and causal effect can dissociate sharply, and a
single-position single-feature patch that nulls is informative —
especially when scaled up the layer axis as we have done here.

## What positions could carry the reveal-causal signal?

Single-position patching has been ruled out for **the four pre-answer
positions** (one per turn). Other candidate positions are:

1. **Turn-0 Ready output position.** In the self-chosen condition,
   the model first emits `Ready` after silently picking an animal.
   The class commitment plausibly sits in the residual stream around
   that token. **Not yet tested.** Would require either re-running
   src forward passes through the up-to-Ready context to capture
   activations, or re-collecting activations at the Ready position.
2. **Question-text token positions** within each turn. These are the
   positions where turn-specific information enters the residual
   stream. Not yet tested.
3. **Earlier turn answer-token positions** (the `Yes`/`No` outputs).
   Not yet tested.
4. **Distributed across multiple positions.** If no single position is
   the bottleneck, the class is encoded redundantly and only a
   position-band patch can move reveals.

These all share a structural feature: they require either (a) live src
forward-pass capture during the patching experiment, or (b) re-running
the collection pipeline with a different position-capture target. Both
are moderate engineering steps relative to phase 2c-i, which used
existing saved tensors.

## Decision consequence — phase 2c-ii (position-band patch)

Single-position patching has produced a comprehensive null. The
remaining structural hypothesis is that the class commitment is
encoded **across multiple positions** rather than at any one of them.
The diagnostic test is a position-band patch: replace src's
activations across a *window* of positions in tgt's context, all
layers, and see if reveals flip.

Implementation work for phase 2c-ii:

1. **Live src activation capture.** Add a routine that, for each src
   run, builds the up-to-qN-preanswer context, does a forward pass
   with `output_hidden_states=True`, and grabs hidden-states across
   a position window. This replaces the current "load saved
   single-position activation file" step.
2. **Position window CLI.** Add `--position-window K` and patch
   the last K positions of the prefill at the chosen turn (positions
   `[pos-K+1, ..., pos]`). For chat-template alignment, the last 5
   tokens of any prefill ending with `add_generation_prompt=True` are
   typically the same scaffolding tokens (`<end_of_turn>`, `\n`,
   `<start_of_turn>`, `model`, `\n`), so K up to ~5 is alignment-safe
   without needing per-run position alignment logic. Larger K extends
   into question text, which differs per run and would require
   alignment work or restriction to src/tgt pairs with matching
   turn-N questions.
3. **First test.** K=5 at turn-4 pre-answer (i.e. patch the trailing
   5 scaffolding tokens at all 48 layers). Cheapest informative
   variant; the patch covers the full chat-template scaffolding band.
4. **Escalation if K=5 nulls.** Either (a) extend to K=20 with
   per-run question-text alignment, or (b) move the window to turn-0
   Ready output position (requires a different up-to-Ready context
   builder), or (c) patch multiple discontiguous positions (e.g.,
   turn-0 Ready + turn-1 pre-answer + turn-4 pre-answer all together).

This is a moderate refactor: the script needs a new "live src capture"
path that lives alongside the existing "load saved activations" path.
Defer until the user has a chance to weigh in on scope and whether the
position-band experiment is the right next move vs. something else
(e.g., switch focus to the Ready-output position specifically, or do
a positional probing sweep first to localize where the class signal
*originally* enters the residual stream).

## Side observations

- **`attempt_588` non-determinism is consistent across all 4 turns.**
  The same horse target's baseline returns `cow` deterministically
  *across all four* per-turn baseline runs, but on disk it was saved
  as `horse`. This is reproducible and replays consistently — so it's
  not transient nondeterminism in the patch script's forward pass; it's
  a persistent divergence from the original collection's forward pass.
  Likely cause: subtle KV-cache or attention-implementation difference
  between the original collection (which generated the secret choice +
  4 turns + reveal in one streaming forward pass) and the replay
  (which rebuilds the input from saved manifests and prefills
  fresh).
- **`attempt_581` is baseline-stable but perturbation-unstable.** Its
  baseline reveal is `cow` deterministically across all 4 turns, but
  under any heavy residual disruption it tips to `horse` 15% of the
  time. The other 4 cow targets do not show this — they remain `cow`
  under the same disruption. So `attempt_581` is closer to the
  cow/horse decision boundary than its peers in the model's
  representation space.
- **Self-patch noise scales with turn distance from reveal.** In
  turn 4, all-layer self-patch keeps the reveal at 100% (cow) or
  92% (cow→cow). In turn 1, all-layer self-patch destabilizes more
  (cow→cow at 88%, horse→horse at 88%). Hypothesis: the residual
  perturbation introduced at an earlier position has more downstream
  attention/MLP propagation before reaching the reveal-token decode,
  which gives it more chance to propagate into the cow/horse
  attractor.

## Artifacts

- `runs/m4_patch_turn{1,2,3}_12b_default_L1-48all.json` — three
  full per-trial records + per-cell summaries.
- `scripts/patch_turn4.py` at `b27d39e` with `--turn N` flag.
- Job script `jobs/tq_m4_patch_turnsweep_12b_20260425.sh` on TSUBAME.
- Reference: Heimersheim & Nanda 2024, *How to use and interpret
  activation patching* (arXiv 2404.15255); "Causality ≠ Decodability"
  (arXiv 2510.09794).
