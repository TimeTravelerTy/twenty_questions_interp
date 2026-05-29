# Exp 2 — 27B SAE feature analysis at `end_ready` (probe peak)

**Headline:** at the one scale+position where the linear class probe newly
lights up (27B `end_ready`, dense LR LOO **0.525** @ resid_post[16], 3.15×
chance), the Gemma Scope 2 SAE **does not sparsify the class direction**.
Balanced sparse LR LOO is **0.300** (1.80× chance) — the SAE retains only
~57% of the dense class accuracy. And the SAE features that *do* fire at
`end_ready` are **generic chat-template / web-text format features**
("names of people", "XML tags and identifiers", "HTML tags and URLs",
"punctuation and separators", "code files and definitions") — **zero**
animal-class features and **zero** attribute (size/mammal/predator)
features. The legible class signal lives in dense, low-variance residual
structure the interpretable sparse basis does not isolate; what the SAE
isolates at Ready is the *prompt scaffolding*, not the chosen animal.

This replicates the 12B SAE-negative result (residual 0.79 → sparse
0.21–0.33) at the scale where the linear class signal is strong, closing
the most glaring gap in the M5 legibility story: 12B `end_ready` was at
chance, so its SAE-negative was uninformative; 27B `end_ready` has a real
3.15× signal, and the SAE still misses it.

## Setup

- **Model / SAE:** `google/gemma-3-27b-it`; Gemma Scope 2 SAE
  `google/gemma-scope-2-27b-it`, `resid_post/layer_16_width_65k_l0_medium`
  (25%-depth flagship; JumpReLU; d_in 5376, d_sae 65536, l0=53). Width/L0
  match the 12B SAE (`gemma-scope-2-12b-it` layer_31 width_65k_l0_medium)
  for an apples-to-apples sparsification comparison.
- **Layer indexing:** probe peak "L16" = capture-index 16 = resid_post[15].
  The flagship SAE `layer_16` is trained on resid_post[16] = **capture-index
  17** (one block past the peak, inside the L12–L18 decodable plateau
  ~0.44–0.51). Firings job fed `--capture-index 17 --block-id 16`.
- **Inputs:** 600 self-chosen runs, v2 16-anchor positional residuals
  `runs/positional_residuals/27b_default_n80_v2/` (job 7566164).
- **Realized classes (7):** cow, dog, elephant, gorilla, horse, shark,
  tiger. shark (n=2) is degenerate; dropped for the balanced 6-class
  comparison (chance 1/6 = 0.167).
- **Reconstruction sanity:** **FVU ≈ 0.0048** — the SAE reconstructs the
  `end_ready` residual near-perfectly, so the class info *is* present in the
  reconstruction; the question is whether it is in the *interpretable sparse
  code*. It is not.
- **Jobs:** firings 7796102 (cpu_4, 425 s, 3000 firings @ end_ready +
  pre_answer_q1..q4); dense baseline 7796138 (cpu_4). Prong-B labels pulled
  from Neuronpedia `gemma-3-27b` / `16-gemmascope-2-res-65k`.

## Prong A — does the legible class direction sparsify? **No.**

Matched 6-class balanced (20/class, drop shark), chance 0.167:

| representation | LR LOO @ resid_post[16] (cap-idx 17) | × chance |
|---|---:|---:|
| **dense residual** | **0.525** | 3.15× |
| dense residual @ cap-idx 16 (probe peak) | 0.508 | 3.05× |
| **sparse SAE (65k/medium)** | **0.300** | 1.80× |

- Only **45 of 65,536** SAE features are active at `end_ready` across all
  600 runs.
- Every top feature fires at rate **1.00 in every class**, with per-class
  mean activations differing by **<1%** (top ANOVA feature 8836: per-class
  means 27.07/26.82/27.17/27.43/26.97/27.02; F=9.2, effect 0.61 on a ~27
  magnitude). These are class-*invariant* "Ready-state" features.
- Unbalanced 7-class sparse LR LOO = 0.455, but that is just the majority
  baseline (horse 0.67 / tiger 0.53; cow 0.06, dog 0, elephant 0,
  gorilla 0.17, shark 0) — the sparse code carries essentially no
  minority-class signal.

The SAE keeps ~57% of the dense class accuracy (0.300/0.525) and loses the
rest. Cf. 12B (residual 0.79 → sparse 0.21–0.33, ~60–73% lost): 27B retains
a bit more, but the class direction remains substantially **residual-
distributed, not sparsely encoded**, at the scale where it is most legible.

## Prong B — what *does* fire at Ready? **Format features, not class.**

Top features by mean activation at `end_ready` (identical to the
firing-frequency ranking, since all 45 active features fire at 100%).
NPmax = Neuronpedia `maxActApprox` (l0-match validation: our observed
activations sit ≤ NPmax for essentially all features → Neuronpedia's
`16-gemmascope-2-res-65k` == our l0_medium encode):

| feat | mean act | NPmax | Neuronpedia label |
|---:|---:|---:|---|
| 2455 | 255.1 | 397.3 | first on |
| 1643 | 199.6 | 375.3 | names of people |
| 183 | 194.9 | 467.2 | special characters and diacritics |
| 3852 | 182.4 | 163.4 | code files and definitions |
| 174 | 177.4 | 584.6 | sequences of lines |
| 2158 | 138.8 | 198.3 | European or Latin-sounding word endings |
| 4293 | 101.9 | 336.0 | hundred and percentages |
| 2239 | 95.2 | 359.1 | links in text |
| 1115 | 75.2 | 604.2 | XML tags and identifiers |
| 5746 | 68.4 | 307.5 | HTML tags and URLs |
| 6348 | 86.5 | 130.3 | conjunctions and punctuation |
| 1763 | 78.9 | 378.7 | Punctuation and separators |

The top *class-discriminative* feature (by ANOVA F) is feat 8836 =
"loop termination conditions" — also a code/format feature, with
near-identical per-class means. **No** feature in either ranking is about
animals, size, mammals, or predators.

Interpretation: the interpretable content the SAE exposes at `end_ready` is
the **chat-template scaffolding** (`<start_of_turn>model … Ready
<end_of_turn>`, special tokens, formatting) — exactly what you'd expect to
dominate the high-magnitude residual at a structural boundary. The faint
but real linear class signal (dense 0.525) is *not* one of these features;
it is smeared across their low-variance magnitudes and/or the inactive tail.

## Why this matters for the claim

- **Legibility chapter, sharpened.** At 27B `end_ready` there is no sparse
  "I'm thinking of a horse" feature. The class direction is legible to a
  *linear probe over the dense residual* but not isolated by an
  *interpretable sparse code*. Legibility (to a probe) ≠ sparse
  representation (an interpretable feature). This is the cleanest statement
  of "legible but not load-bearing" on the representational side, and it
  pairs with the causal side (all M5/M5b patch + steering nulls).
- **Bears on Exp 1 (attribute probing).** The SAE shows **no attribute
  features** firing at Ready either, so the "the probe is really reading
  attributes (size/mammal/predator)" hypothesis gets *no support* from the
  sparse code at this position. It does not refute a *linear* attribute
  direction (attributes could be dense-decodable without a dedicated SAE
  feature) — so Exp 1 is still worth running, but it should be framed as
  "is there a linear attribute axis," not "is there an attribute feature."

## Caveats / follow-ups

- Top-by-activation features are the highest-magnitude ones, which at a
  structural position are format-dominated by construction. But prong A
  uses the *full* active set (all 45 features incl. low-activation ones)
  and still only reaches 0.300 — so the null is not an artifact of looking
  only at the loud features.
- Single SAE width (65k, matches 12B). A 262k-width SAE could in principle
  resolve a finer class feature; fast follow if we want to push it, but the
  dense-vs-sparse gap + format-only labels already make the point.
- pre_answer_q1..q4 firings were captured in the same pass (`end_ready`'s
  probe peak is here; the pre_answer peaks are at other layers, L38/L61, so
  those firings are context, not a matched test).

**Artifacts:**
`runs/m5_sae_firings_27b_default_resid_post_L16_endready_pa1234.pt`,
`runs/m5_sae_analyze_27b_L16_end_ready{,_bal6}.json`,
`runs/m5_sae_firings_..._L16_endready_pa1234.labels_end_ready.json`,
`runs/m5_dense_probe_27b_end_ready_L16_17_drop_shark.json`.
Scripts: `scripts/sae_feature_firings.py`,
`scripts/analyze_sae_features.py` (+`--drop-class`),
`scripts/probe_positional_anchors.py` (+`--drop-class`),
`scripts/neuronpedia_label_features.py` (new).
