# M5 — Scale-robust improvisation: the unified headline

**Headline:** across the 12B→27B scale step, the model's privately-chosen
class is **decodable in the residual but not load-bearing for the reveal.**
Scale rotates more class information into linearly-readable subspaces — it
upgrades *legibility* — without changing the *causal* mechanism. The model
re-derives ("improvises") its answer downstream of any cached commitment.
Six causal interventions, escalating in strength, are all null. The single
apparent exception is one boundary-degenerate target run that flips
content-agnostically and carries zero class signal.

This document unifies the three M5 strands — a decoding probe, an
SAE-negative, and a patch sweep — into one claim for the blog draft.

## The three strands

### 1. Decoding probe — scale upgrades legibility

(`docs/progress/M5-positional-probe-27b-default-v2.md`)

A linear probe (LR LOO, 7-way, chance 0.143) reads the self-chosen class
off the residual at 16 structural anchors. Every anchor strengthens at 27B
vs 12B by ~+1.6–2.4× chance. The qualitative jump is at the early anchors:

| anchor | 27B LR max @ L | × chance | 12B status |
|---|---|---:|---|
| end_user_prompt | 0.484 @ L19 | 3.39 | at chance |
| **end_ready** | **0.508 @ L16** | **3.55** | **at chance** |
| pre_answer_q1 | 0.574 @ L38 | 4.02 | weak |
| **pre_answer_q4** | **0.672 @ L38** | **4.70** | 3.15 |
| pre_reveal_gen | 0.820 @ L58 | 5.74 | strong |

At 27B the class is already legible at `end_ready` — before any question is
asked. 12B is at chance there. Read naively, this looks like "scale grants
explicit pre-commitment."

### 2. SAE-negative — the direction is residual-distributed

(`docs/progress/M5-sae-residual-misaligned-12b-L31-L41.md`)

Default Gemma Scope 2 SAEs at L31 and L41 fail to sparsify the class
direction the probe finds: residual LR 0.79 → sparse LR 0.21–0.33, and the
top-10 class-discriminative feature sets share **∅** across turns. The
self-chosen class direction is residual-distributed, not a sparse feature.
Whatever the probe reads, it is not a clean "I picked horse" latent.

### 3. Patch sweep — the legible direction is not load-bearing

The decoding result does **not** survive contact with causal patching. We
transplant a source run's residual into a target run at a structural anchor,
regenerate the reveal greedily across the intervening dialogue turns, and
measure (a) argmax flip of the reveal token and (b) `logit[src]-logit[tgt]`
shift vs. the unpatched baseline. Six interventions, escalating in strength:

| # | scale | intervention | off-diag flip-to-src* |
|---|---|---|---:|
| 1 | 12B | full-residual L1-L48 @ pa4 | 0 / 300 |
| 2 | 27B | band L27-L62 @ pre_answer_q4 | 0 / 843 |
| 3 | 27B | band L12-L20 @ end_ready | 0 / 843 |
| 4 | 27B | **multi-anchor** L12-L48 @ end_ready + end_model_q1–q4 | **0 / 843** |
| 5 | 27B | **full-residual L1-L62 @ pre_answer_q4** | **0 / 843** |
| 6 | 27B | **full-residual L1-L62 @ end_ready** | **0 / 843** |

\* off-diagonal trials, *excluding* the boundary-degenerate target
`attempt_593` (see below). Interventions 4–6 are new in this session;
1–3 are the prior M5/M4 sweep, re-tallied with the same exclusion.

Intervention 4 blocks every commitment / re-derivation site at once — the
initial commitment locus plus the post-answer state after each of the four
turns. Interventions 5–6 replace the *entire* residual stream at a single
anchor — the strongest possible single-site intervention. None transplant
the class. Every diagonal self-patch sanity check passes (zero logit-diff
delta). Off-diagonal logit-diff deltas (excl. shark) are negligible and
centred on zero or slightly *negative* — the patch, if anything, nudges the
reveal *back toward the target's own class*:

| intervention | logit-diff Δ mean | median | any-change |
|---|---:|---:|---:|
| 4 multi-anchor L12-48 | −0.027 | 0.000 | 20 / 870 |
| 5 pa4 full-residual L1-62 | −0.211 | −0.250 | 17 / 870 |
| 6 end_ready full-residual L1-62 | −0.016 | 0.000 | 13 / 870 |

## The `attempt_593` correction

STATUS previously reported the 27B band patches as "3/870 flips, all from
cow sources." That framing is a **metric artifact** and is now corrected.

The flip-to-*source* counter only registers a flip when the reveal lands on
the source class. `attempt_593` (a horse-reveal target) sits exactly on the
horse/cow decision boundary: *any* residual patch tips it into the cow
attractor. So a cow→593 patch counts ("flip to cow source"), but a
horse→593 or tiger→593 patch flipping it to cow does **not** — masking the
true picture.

The full-residual results make the degeneracy unambiguous. Under L1-L62
every-layer replacement — which substitutes the source run's *entire*
residual — `attempt_593` does this:

| intervention | attempt_593 patched outcomes (n=32, all source classes) |
|---|---|
| pa4 full-residual L1-62 | **16 cow / 16 horse** |
| end_ready full-residual L1-62 | 15 cow / 17 horse |
| multi-anchor L12-48 | 9 cow / 23 horse |

A ~50/50 horse/cow split *regardless of source class* — including
horse→horse self-patches — is the signature of a target on a knife-edge
between two attractors, not of class transfer. If the residual carried the
class causally, a full-residual replacement from a dog or tiger source
would push `attempt_593` toward dog or tiger; instead it only ever flickers
between its own two boundary attractors. `attempt_593` contributes **zero**
class-signal evidence and is excluded from the flip tallies above.

With it excluded: **0 off-diagonal class transfers across all six
interventions and all other target runs.** Unpatched baselines are rock
stable — 0/32 drift in every 27B job (every target's fresh-replay reveal
matches its on-disk reveal). The fragility of `attempt_593` is real but is
revealed *only* under perturbation; its unpatched baseline is stable horse.

This also retires the standalone "replay `attempt_593`" follow-up: the
horse→horse self-patch is a stronger fragility control than a baseline
replay would have been, and it is already in hand.

### Minor non-specific drift (not class transfer)

Two residual non-null cells, both benign:

- **Multi-anchor, shark sources:** 7 non-593 targets show drift, all from
  `shark` source runs, all flipping to *non-shark* classes
  (elephant/cow/horse/tiger). `shark` has only 2 source runs (the 27B run
  distribution collapses to ~6 attractors; see
  `M4-comparative-prompt-and-scale.md`). A degenerate shark residual
  patched at five anchors disturbs targets non-specifically — it does not
  transfer "shark."
- **pa4/end_ready full-residual:** 3 isolated single-trial flip-to-*other*
  events (e.g. `shark→gorilla-target → tiger`). Not flip-to-source.

Both are noise from the underpowered `shark` class, not evidence of a
causal class channel.

## Synthesis for the blog

**Scale shifts class-info legibility, not its causal role.** At 27B the
residual exposes the self-chosen class to a linear probe far earlier and
more strongly than at 12B (`end_ready` goes from at-chance to 3.55× chance).
But that legible direction is *epiphenomenal* for the reveal: replacing the
entire residual stream at `end_ready`, or at `pre_answer_q4`, or blocking
all five commitment/re-derivation sites at once, never transplants the
class. The model re-derives its answer from the dialogue's accumulated
yes/no constraints downstream of any cached state — the improvisation /
re-derivation mechanism established at 12B (M4). It is **scale-robust**.

The naive reading of strand 1 ("scale grants explicit pre-commitment") is
therefore wrong at the causal level. Scale makes the model's *bookkeeping*
more visible to a probe; it does not make that bookkeeping load-bearing.

The three strands cohere: (1) the class is increasingly decodable with
scale, (2) but not as a sparse feature — it is a distributed residual
direction, and (3) not a causal one — patching it does nothing. A decodable,
distributed, epiphenomenal direction is exactly what you expect if the model
is *improvising* the reveal from the dialogue rather than reading out a
stored commitment.

### Why "kept target under patch" is the prediction, not a puzzle

Under greedy decoding (temperature 0), the *only* per-run randomness
that distinguishes a horse-run from a tiger-run is the **permutation of
the 20-candidate list** baked into the user prompt
(`manifest.permutation`). Same model, same prompt template, different
list ordering → different secret commitment. Question selection and
yes/no answers downstream are deterministic consequences of that
initial commitment, not independent variation. So the class identity
is encoded in the *prompt itself* (which 20 animals, in what order),
not just in the model's later residuals.

A patch at one anchor (e.g. end_ready) replaces the residual at *one
token position* in a few layers with source's residual. Everything
else stays target's: the entire prompt's KV cache (with target's
permutation), all later positions, and all unpatched layers at the
patched position. When the model continues generating, attention at
every later position can re-attend to the original target prompt and
re-derive what the secret should be. The patched residual at end_ready
propagates only via that single token's K/V entry — easily drowned
out by the rest of the context.

So "kept target under patch" isn't surprising; it's the prediction of
the improvisation thesis. The interesting question M5 asked was
whether the *legibly decodable* class direction at end_ready (LR
0.508 @ L16) actually drives behavior, given that the prompt-level
class identity is already fully present. The answer keeps coming
back: no — that direction is a readout of the model's reasoning, not
a hidden commitment that would survive being overwritten with another
class's residual.

## Artifacts

New this session:
- `runs/m5_patch_27b_default_multi_endready_endmodelq1-4_L12-48.json`
- `runs/m5_patch_27b_default_pa4_L1-62full.json`
- `runs/m5_patch_27b_default_endready_L1-62full.json`
- `scripts/patch_anchor.py` — extended: `--anchor` accepts a comma-separated
  list for simultaneous multi-anchor patching at one layer band.
- TSUBAME jobs 7737407 (multi-anchor), 7737408 (pa4 full), 7737409
  (end_ready full); smoke 7737394. All gpu_1, ~3 min compute after load.

Prior M5 artifacts: `M5-patch-27b-end_ready-pa4.md`,
`M5-positional-probe-27b-default-v2.md`,
`M5-sae-residual-misaligned-12b-L31-L41.md`,
`M5-attribute-bundle-12b-default-v2.md`.

## Next

M5 is blog-ready. The chapter is: one probe story (legibility rises with
scale), one SAE-negative (no sparse class feature), one patch story
(scale-robust improvisation — six null interventions). Write the blog draft
unifying M4 (improvisation established) and M5 (improvisation is
scale-robust and the decodable direction is epiphenomenal).
