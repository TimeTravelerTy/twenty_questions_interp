# M5 Phase A — SAE basis does not sparsify the self-chosen class direction at 12B (L31 or L41)

**Headline:** at 12B, the class-information direction M3 finds in the
residual stream (LR LOO 0.79 at `pre_answer_q4` / L31) does **not**
decompose into sparse Gemma Scope 2 features at either default fixed-depth
resid_post SAE we tested (L31 or L41), at any of the four
`pre_answer_qN` anchors. Sparse-feature LR LOO sits at chance (0.21–0.33,
chance = 0.25); the top-10 class-discriminative features at one turn share
**zero** members with any other turn at L31 (and 0 features survive
intersection across all four turns at L41 either). The class commitment at
12B is held as a small linear component of a dense residual, not as one or
a stable handful of interpretable SAE features.

Decision gate fired on both layers tested — M5 Phase A halts at default
Gemma Scope 2 layers. M5 pivots to (a) this writeup and (b) the 27B scale
comparison ([project_scale_question.md](../../../.claude/projects/-Users-tyronewhite-side-projects-twenty-questions-interp/memory/project_scale_question.md)).

## The question

M3 found a clean linear class direction at `pre_answer_q4` / L31 — 4-way
LR LOO **0.79** (chance 0.25), broad L27–L48 band. M4 ruled out causal
patchability of that direction at the residual level (0/2280 reveal flips
across 12 single-position patches). Two complementary readings remained:

1. **Improvisation.** The model re-derives the class at every step from
   accumulated yes/no history. The probe-decodable direction is a
   *legible by-product* of the dialogue-integration step (L27 → L48), not
   a stored commitment.
2. **Retrieval, just-not-here.** A stored commitment exists but lives in
   a small set of interpretable features that single-position residual
   patching diffused across.

Phase A asks: do M3's linear-probe class directions decompose into sparse
SAE features at the M3-band layers? If yes, retrieval is back on the
table and the next step is single-feature steering. If no, the
improvisation reading hardens and the scientific weight of the M4 patching
null grows: not only is the residual causally inert, the sparse basis
also fails to surface a class-carrier.

## Setup

- **Source residuals.** v2 16-anchor capture
  `runs/positional_residuals/12b_default_n80_v2/` (600 self-chosen runs,
  Gemma 3 12B-IT, default prompt, 4-question panel).
- **Subset.** 4 well-realized reveals `{cow, dog, elephant, horse}`,
  balanced 20/class → n=80.
- **Anchors.** Four per-turn pre-answer positions `pre_answer_q1..q4` —
  including the M3 LR-0.79 peak at `pre_answer_q4`.
- **SAEs.** Gemma Scope 2 12b-it, `resid_post`, JumpReLU, width 64k,
  L0=medium (60).
  - L31 (capture-index 32): nearest fixed-depth SAE to the M3-measured
    peak.
  - L41 (capture-index 42): late integration band; second cut.
  - Reconstruction quality on the n=80 × 4 anchors: **L31 FVU 1.56%**,
    **L41 FVU 2.30%** — both well inside Gemma Scope 2's published
    bounds, so SAE artifacts are loaded correctly.
- **Pipeline.** Encode each (run, anchor, layer) residual through the SAE,
  keep top-k=64 firings per record. Aggregate across the 80 runs per
  anchor. Fit a sparse LR LOO over active features (active = fired in any
  run for the anchor).
- **Reference.** M3's residual-stream 4-way LR LOO at
  `pre_answer_q4` / L31 = **0.79** (n=80, same subset).

## Result 1 — sparse LR LOO is at chance at every (layer, anchor)

| anchor          | L31 sparse LR | L31 active feats | L41 sparse LR | L41 active feats |
|---|---:|---:|---:|---:|
| pre_answer_q1   | 0.325 |  89 | 0.287 |  54 |
| pre_answer_q2   | 0.325 |  78 | 0.287 |  62 |
| pre_answer_q3   | 0.212 |  90 | 0.275 |  97 |
| pre_answer_q4   | 0.312 |  80 | 0.325 |  51 |

Chance is 0.25. The L31 numbers are uniformly ≤0.33; the L41 numbers
≤0.33. The residual probe at the same (anchor, layer) reaches 0.79. So
the SAE encoding is **shedding the class signal** that the linear probe
sees — by a margin of ≥0.46 at the M3 peak position.

Per-class breakdown shows the chance behavior is uniform across classes
(no single class is "easy"); e.g. at L31 `pre_answer_q4` the per-class
accuracies are [cow=0.35, dog=0.40, elephant=0.25, horse=0.25] — the
small lift on dog is consistent with [the dog-early-emergence pattern
from the attribute-bundle probe][attr], not with a sparse class circuit.

[attr]: M5-attribute-bundle-12b-default-v2.md

## Result 2 — top class-discriminative features are turn-specific (zero cross-turn carry)

For each anchor we rank features by a Fisher-style discriminative score
across the 4 classes, then take the top-10. Cross-turn intersection of
those top-10 sets:

**L31 top-10 pairwise intersection (top_n=10):**

|  | q1 | q2 | q3 | q4 |
|---|---:|---:|---:|---:|
| q1 | 10 | 1 | 0 | 0 |
| q2 | 1 | 10 | 1 | 0 |
| q3 | 0 | 1 | 10 | 1 |
| q4 | 0 | 0 | 1 | 10 |

**L41 top-10 pairwise intersection:**

|  | q1 | q2 | q3 | q4 |
|---|---:|---:|---:|---:|
| q1 | 10 | 0 | 0 | 0 |
| q2 | 0 | 10 | 1 | 1 |
| q3 | 0 | 1 | 10 | 1 |
| q4 | 0 | 1 | 1 | 10 |

**All-four-turn intersection: ∅ at both layers.** Only adjacent turns
ever share even a single feature, and never more than one. There is no
"the class-X feature" that persists across `pre_answer_q1 → q4`. Whatever
class-discriminative SAE activations the model exhibits at each turn are
locally re-derived at that turn — they do not carry a stable class label
forward in the feature basis.

## Why this is interesting (not just a method failure)

Three reasons the negative is a load-bearing finding rather than a SAE
tooling complaint:

1. **It sharpens the M4 improvisation claim from "causally inert" to
   "structurally distributed".** M4 said: single-position residual
   patching doesn't move reveals. M5 Phase A says: even when you encode
   the same residual through a learned sparse basis, the class signal is
   not concentrated on a small set of features — and the small sets that
   *are* class-discriminative don't carry from one turn to the next.
   Both readings — causal and structural — point to the same picture:
   class identity at 12B is held as a small linear component of a dense
   residual that the dialogue-integration step keeps re-deriving, not as
   a stored commitment in an interpretable substrate.

2. **It is consistent with the AxBench pessimism, but on a sharper test.**
   AxBench's headline is that supervised dense readouts (DiffMeans, LR)
   beat SAE features for concept detection. Here we have a much sharper
   gap: residual LR 0.79 → sparse LR 0.31, on a 4-way categorical that
   the residual makes linearly separable. Not "SAEs are slightly worse"
   — SAEs are *at chance*.

3. **The cross-turn ∅ intersection rules out one mechanistic story.**
   If the late-network L27→L48 step were "look up the class-X feature
   and write it forward", we would expect some stability of the
   class-discriminative top-k across `pre_answer_q1..q4` — at minimum a
   subset that survives intersection. Zero survivors at both L31 and
   L41 makes "feature lookup + carry" implausible as the local mechanism
   at the M3-peak band.

## What this does **not** show

These are real and the writeup should not be over-claimed past them.

- **Only `resid_post` is tested.** Gemma Scope 2 ships `mlp_out` and
  `attn_out` SAEs at the same layers. They use different captures
  (submodule-level hooks before the add-to-residual), which we don't
  currently have. The negative is for the residual stream *between
  blocks*, not for the read-and-write steps that produce it. If a
  class signal lives inside a single MLP's output before it gets
  diffused into the residual, this experiment cannot see it.
  Deferred to a potential Phase A1b (not gated on this writeup).
- **Only L31 and L41 are tested.** Gemma 3 12B has 48 decoder blocks;
  Gemma Scope 2 publishes fixed-depth SAEs at 25/50/65/85% (L12 / L24 /
  L31 / L41) and a less-curated every-layer set. The L27–L48 M3 band
  contains L31 and L41 — but a class-carrier feature could live at
  L37 (between them, no fixed-depth SAE) or earlier. The negative is
  about *the two SAE layers in the M3 band that are pre-trained and
  available*, not about every layer.
- **Only `medium` L0 width is tested.** L0 controls sparsity tightness.
  A `low`-L0 SAE (sparser, fewer firings) might isolate a class signal
  the medium-L0 distributes; conversely a `high`-L0 SAE (denser, more
  firings) might surface a class direction the medium misses. The
  medium width is Google's recommended default for non-circuit-level
  interpretability; we tested only that.
- **Possible Gemma Scope 2 IT-variant distribution mismatch.** The SAEs
  were released Dec 2025 against the IT-variant model; they are newer
  than the Gemma Scope 1 base-model artifacts. FVU 1.6%/2.3% is healthy,
  but training-distribution mismatch against the 20-questions dialogue
  format is harder to detect from FVU alone.
- **n=80 is small for sparse classifiers.** 80 examples × ~50–90 active
  features pushes the LOO classifier into a regime where regularization
  starvation could mask a weak true signal. Residual LR at 0.79 on the
  same n=80 shows the *dense* basis isn't regularization-starved at
  this n, but a *sparse* basis with comparable-or-fewer informative
  features could be.

## Predicted findings reconciliation (preregistered in the plan)

The Phase A plan made specific calls. Reconciling:

- ❌ "A3.1 class-discriminative features at L30+: expect 5–15 features
  per class … Many more → distributed; many fewer → unexpectedly sharp
  commitment." → Got ~50–100 *active* features per anchor, but the
  *class-discriminative* set doesn't reproduce across turns. Closer to
  "distributed and turn-specific" than to either preregistered alternative.
- ❌ "A4 reproducibility check: sparse classifier matches residual probe
  LR LOO within ±0.05." → Got a 0.46-point gap at the M3 peak. AxBench
  negative confirmed at extreme magnitude.
- ❌ "A3.2 turn-progressive features: expect them to cluster in `mlp_out`
  at mid-to-late layers." → Not tested at L31/L41 because the cross-turn
  intersection is empty, meaning there is no candidate set of
  turn-progressive features in the resid_post basis to even chase into
  mlp_out yet.
- The L30 generic-prior-dominance prediction (horse asymmetry) and B1
  single-feature steering are now moot: Phase B is gated on Phase A
  surfacing ≥3 mechanism-shaped candidates. It did not.

## Caveats addressed

- **Tautology at `pre_reveal_gen`.** Not tested in this writeup — we
  stayed strictly inside the four `pre_answer_qN` anchors, which are
  upstream of the reveal-generation logits. No tautology risk.
- **Class-balance robustness.** n=80 is balanced 20/class. Per-class
  LR accuracies are visible in the per-anchor JSON; no class is
  systematically masking the mean.

## Artifacts

- **Firings (sparse top-k=64 per record):**
  - `runs/m5_sae_firings_12b_default_resid_post_L31_pre_answer_q1q2q3q4.pt` (2400 records)
  - `runs/m5_sae_firings_12b_default_resid_post_L41_pre_answer_q1q2q3q4.pt` (2400 records)
- **Per-anchor sparse LR + top features:**
  - `runs/m5_sae_analysis_12b_default_resid_post_L31_pre_answer_q[1234]_balanced.json`
  - `runs/m5_sae_analysis_12b_default_resid_post_L41_pre_answer_q[1234]_balanced.json`
- **Turn-progressive aggregator (intersection tables above):**
  - `runs/m5_sae_turn_progressive_L31_balanced.json`
  - `runs/m5_sae_turn_progressive_L41_balanced.json`
- **TSUBAME jobs:**
  - L31 firings: `tq_m5_sae_firings_12b_default_L31_pa1234_20260509.sh` (job 7342174 retrofit, succeeded earlier; current artifact from job 7342179)
  - L41 firings: `tq_m5_sae_firings_12b_default_L41_pa1234_20260509.sh` (job 7342179, FVU 2.30%)
  - Per-anchor + aggregator: `tq_m5_analyze_sae_L31_pa1234_balanced_20260509.sh`,
    `tq_m5_analyze_sae_L41_pa1234_balanced_20260509.sh` (jobs 7342183, 7342191)

## What's next

Per `~/.claude/plans/check-the-latest-status-bright-horizon.md`, Phase A
halts at default Gemma Scope 2 layers and M5 pivots to the 27B scale
comparison. The discriminating observable at 27B is **end_ready LR LOO**:
at 12B it sits at chance (improvisation); if 27B clears chance, scale
grants explicit pre-commitment, and the follow-up patch test asks
whether that pre-commitment is causally load-bearing or just legible.

Deferred (not blocking):

- **A1b** `mlp_out` / `attn_out` SAEs at L31/L41. Would need a fresh
  capture pass with submodule hooks; might surface a class-carrier
  inside the read-and-write steps that the inter-block residual loses.
- **Layer scan at less-curated SAEs.** Gemma Scope 2 publishes
  every-layer reduced-coverage SAEs. Would not change the structural
  finding (no cross-turn intersection at the layers tested) but could
  narrow where the class direction is most reachable in sparse-feature
  space.
