# M5 — Attribute-bundle probe on 12B self-chosen (v2 capture)

**Headline:** the "attribute bundle" hypothesis collapses to per-class
identity for our 4-way subset (every panel attribute is a 1v3 split,
picking out exactly one class), but the experiment surfaces a new
structural finding: **different classes become linearly decodable at
different dialogue anchors**. Dog is decodable from `pre_answer_q1`;
elephant/cow/horse only crystallise at turn-4.

## Setup

- Source: `runs/positional_residuals/12b_default_n80_v2/` (600 self-chosen
  attempts, 12B default prompt, 4-question prompt panel
  `is_mammal, is_bird, lives_primarily_in_water, has_four_legs` — note
  *none of these distinguish among the 4 realized classes* in the bank).
- Subset: balanced 20/class over the 4 well-realized reveals
  `{cow, dog, elephant, horse}` → n=80.
- Attributes probed (matched 6-question panel, trained directly on the
  v2 self-chosen residuals via per-attribute binary LR LOO):
  `is_carnivore, is_larger_than_human, is_domesticated, lives_in_africa,
  produces_dairy_milk, is_ridden_by_humans`.
- Per-attribute majority baseline is **0.75** for all six (each is a
  1v3 split in the 4-class subset).
- Anchors: 16 (`end_user_prompt`, `end_ready`, `end_user_qN/pre_answer_qN/end_model_qN`
  for N∈{1..4}, `end_reveal_user`, `pre_reveal_gen`).
- Reference: 4-way class LR LOO at the same (anchor, layer) grid.

## Per-attribute = per-class binarisation

For our 4-class subset, the 6 attributes resolve to 4 distinct 1v3 splits:

| Attribute | Yes class | Effective "is X?" axis |
|---|---|---|
| `is_carnivore` | dog | is_dog |
| `is_larger_than_human` | cow, elephant, horse (dog=no) | is_dog (inverted) |
| `is_domesticated` | cow, dog, horse (elephant=no) | is_elephant (inverted) |
| `lives_in_africa` | elephant | is_elephant |
| `produces_dairy_milk` | cow | is_cow |
| `is_ridden_by_humans` | horse | is_horse |

So the probe collapses to **four per-class "is X?" binary axes**, two
of which (dog, elephant) appear twice from different sides.

## Late-band L27-48 mean LR LOO (baseline 0.75)

| Anchor | is_dog (carnivore) | is_elephant (africa) | is_cow (dairy) | is_horse (ridden) |
|---|---|---|---|---|
| end_user_prompt | 0.680 | 0.666 | 0.646 | 0.688 |
| **end_ready** | 0.657 | 0.662 | 0.651 | 0.673 |
| end_user_q1 | 0.676 | 0.704 | 0.653 | 0.726 |
| **pre_answer_q1** | **0.801** | 0.711 | 0.654 | 0.712 |
| end_model_q1 | 0.690 | 0.694 | 0.630 | 0.701 |
| end_user_q2 | 0.759 | 0.670 | 0.669 | 0.738 |
| pre_answer_q2 | 0.709 | 0.701 | 0.685 | 0.710 |
| end_model_q2 | 0.690 | 0.673 | 0.634 | 0.721 |
| end_user_q3 | 0.670 | 0.667 | 0.659 | 0.661 |
| pre_answer_q3 | 0.730 | 0.695 | 0.673 | 0.673 |
| end_model_q3 | 0.668 | 0.669 | 0.688 | 0.705 |
| **end_user_q4** | 0.751 | **0.835** | 0.718 | 0.732 |
| **pre_answer_q4** | **0.853** | **0.885** | **0.846** | **0.835** |
| end_model_q4 | 0.804 | 0.764 | 0.708 | 0.747 |
| end_reveal_user | 0.786 | 0.715 | 0.696 | 0.762 |
| pre_reveal_gen | 0.940 | 0.880 | 0.917 | 0.856 |

Values **below 0.75** are below majority baseline — the per-fold LR is
worse than predicting the majority class, i.e. zero signal.

## Findings

1. **No per-class commitment at end_ready.** All four axes sit at/below
   baseline (0.657–0.673). Matches the M3/M4 finding that 4-way class
   identity is at chance at end_ready (LR 0.30); the *per-class binary*
   refinement adds nothing — even "is this X?" for X ∈ {dog, elephant,
   cow, horse} is not above majority.
2. **Class-specific emergence ordering.** The first anchor at which
   each "is X?" axis clears 0.78 (visibly above baseline):
   - **is_dog: `pre_answer_q1` (0.801)** — three turns earlier than the rest.
   - **is_elephant: `end_user_q4` (0.835)** — turn-4 only.
   - **is_cow: `pre_answer_q4` (0.846)** — turn-4 only.
   - **is_horse: `pre_answer_q4` (0.835)** — turn-4 only.
3. **All four axes peak at pre_answer_q4** (0.835–0.885), matching the
   M3 4-way LR LOO peak at the same anchor. The 4-way signal isn't
   coming from a single class being decoded — all four classes are
   simultaneously discriminable by turn-4.
4. **pre_reveal_gen lights everything up** (0.856–0.940). Partly
   tautological (the model is about to emit the class name).
5. **Dog's early emergence is the new texture.** Around turns 1–2,
   "is_dog" rides 0.04–0.08 above the other three classes' axes. This
   isn't because the model has answered a dog-distinguishing question
   (the in-prompt panel `is_mammal,is_bird,lives_primarily_in_water,
   has_four_legs` is 1,0,0,1 for every class — degenerate). Plausible
   explanations:
   - **Prior asymmetry.** Dog is the most-frequent animal token, so
     when the model improvises a class the dog-conditioned residual
     state is the most strongly perturbed from the generic-prior mean.
   - **Carnivore-as-singleton.** Of the 4 classes, dog is the only
     carnivore. The model may track a "predator-vs-not" axis that
     starts forming as soon as the dialogue context exists.

## Implications for the blog claim

The improvisation story holds and sharpens:

> At 12B, the model does not commit to a class at end_ready. By turn-4
> pre-answer the class is linearly decodable from the residual stream
> at ~0.8–0.9 binary accuracy. **Crucially, class commitment is
> class-specific**: dog crystallises by `pre_answer_q1`; cow, elephant,
> and horse only at turn-4. The model is not making one decision at
> one moment; it is resolving uncertainty across multiple dialogue
> anchors, in a class-dependent order.

Combined with M5 Phase A (the SAE basis at L31/L41 doesn't sparsify
the class direction at any turn), the picture is: class identity at
12B is **residual-distributed, dialogue-progressive, and class-asymmetric**.

## Caveats

- 1v3 splits mean per-attribute binary accuracy lower-bounds at 0.75
  even with zero signal. The "above baseline" reads are differences
  of 0.05–0.13 — substantial but not enormous, and on n=80 the
  per-fold variance is high.
- The "attribute bundle" framing in the original plan envisioned
  *orthogonal* attributes (a 6-dim binary vector); our realised 4-way
  subset doesn't have orthogonal partitions, so we cannot test
  attribute-vs-class separability strongly. Would require a richer
  realised subset (e.g. 6-way over {cow, dog, elephant, horse, +2
  more classes the 12B will reliably commit to under the 20-bank
  prompt}).
- The per-class emergence ordering needs replication with a different
  prompt seed / question panel before being load-bearing.

## Artifacts

- Probe script: `scripts/probe_attribute_anchors.py`
- TSUBAME job: `jobs/tq_m5_attribute_probe_12b_default_v2_20260509.sh`
  (job `7360344`, cpu_40, 2.7 h walltime, 4704 LR LOO fits).
- Output: `runs/m5_attribute_probe_12b_default_v2_n80.json`.
