# M2 — Ready-state decoder smoke test

**Closed:** 2026-04-19 by Claude (Opus 4.7).
**Model:** `google/gemma-3-1b-it` on Mac CPU, float32.
**Runs:** 160 calibration (20 candidates × 8 seeds) + 40 self-chosen.
**Per-run capture:** residual stream at the token position immediately before
`Ready` is generated, all 27 layer outputs (embedding + 26 transformer blocks),
hidden size 1152.

## Exit criteria

| Criterion | Threshold | Result | Status |
|---|---|---|---|
| `Ready` parses cleanly | n/a | 200/200 | ✅ |
| Reveal parses (self-chosen) | ≥ 80% | 39/40 (97.5%) | ✅ |
| Calibration nearest-centroid LOO at some layer | > 20% | max 0.12 (layer 16–17) | ❌ |
| Calibration logreg LOO at some layer | n/a (new metric) | **max 0.38 (layer 15)** | ✅ |
| Any binary attribute decoder at some layer | > 70% | **0.89 `lives_in_africa` (L17); 0.84 `is_bird`; 0.78 `is_carnivore`; 0.74 `is_mammal`** | ✅ |

Infrastructure is proven. The stricter NC-LOO criterion missed — see the
NC-vs-LR gap finding below; at M2 sample sizes this is not diagnostic of
representation geometry. Attribute-decoder wins should also be read against
the majority baseline in `runs/m2_report.json` (some attributes are only
modestly above majority); treat the 0.89 / 0.84 headline numbers as upper
bounds, not as evidence that the bundle framing is right. M2 closed on
balance as infra; the scientific interpretation is deferred to M3.

## Full layer-sweep table

| layer | NC LOO | LR LOO | is_mammal | is_bird | is_carnivore | lives_in_africa | SC-agree |
|---|---|---|---|---|---|---|---|
| 0 | 0.05 | 0.00 | 0.00 | 0.80 | 0.65 | 0.80 | 0.41 |
| 1 | 0.03 | 0.06 | 0.44 | 0.63 | 0.62 | 0.67 | 0.05 |
| 2 | 0.07 | 0.05 | 0.47 | 0.69 | 0.58 | 0.66 | 0.00 |
| 3 | 0.07 | 0.04 | 0.44 | 0.66 | 0.59 | 0.67 | 0.00 |
| 4 | 0.07 | 0.06 | 0.40 | 0.64 | 0.61 | 0.67 | 0.00 |
| 5 | 0.08 | 0.07 | 0.46 | 0.62 | 0.62 | 0.62 | 0.00 |
| 6 | 0.06 | 0.06 | 0.43 | 0.59 | 0.63 | 0.61 | 0.05 |
| 7 | 0.07 | 0.06 | 0.47 | 0.57 | 0.56 | 0.66 | 0.00 |
| 8 | 0.06 | 0.05 | 0.46 | 0.59 | 0.57 | 0.68 | 0.00 |
| 9 | 0.07 | 0.05 | 0.44 | 0.64 | 0.64 | 0.68 | 0.00 |
| 10 | 0.03 | 0.06 | 0.46 | 0.62 | 0.59 | 0.63 | 0.03 |
| 11 | 0.05 | 0.07 | 0.46 | 0.61 | 0.52 | 0.61 | 0.05 |
| 12 | 0.05 | 0.06 | 0.46 | 0.66 | 0.54 | 0.68 | 0.03 |
| 13 | 0.07 | 0.32 | 0.74 | 0.81 | 0.76 | 0.85 | 0.00 |
| 14 | 0.07 | 0.34 | 0.72 | 0.84 | 0.78 | 0.86 | 0.03 |
| 15 | 0.10 | **0.38** | 0.71 | 0.83 | 0.76 | 0.88 | 0.03 |
| 16 | 0.12 | 0.34 | 0.72 | 0.78 | 0.76 | 0.88 | 0.03 |
| 17 | 0.12 | 0.34 | 0.68 | 0.78 | 0.76 | **0.89** | 0.03 |
| 18 | 0.07 | 0.29 | 0.71 | 0.80 | 0.74 | 0.86 | 0.03 |
| 19 | 0.06 | 0.25 | 0.71 | 0.77 | 0.71 | 0.86 | 0.03 |
| 20 | 0.07 | 0.24 | 0.71 | 0.76 | 0.71 | 0.84 | 0.03 |
| 21 | 0.09 | 0.23 | 0.69 | 0.77 | 0.76 | 0.88 | 0.03 |
| 22 | 0.08 | 0.22 | 0.72 | 0.81 | 0.74 | 0.89 | 0.03 |
| 23 | 0.08 | 0.23 | 0.72 | 0.78 | 0.76 | 0.88 | 0.03 |
| 24 | 0.08 | 0.17 | 0.69 | 0.77 | 0.71 | 0.87 | 0.03 |
| 25 | 0.08 | 0.21 | 0.71 | 0.76 | 0.71 | 0.86 | 0.03 |
| 26 | 0.11 | 0.17 | 0.67 | 0.72 | 0.73 | 0.83 | 0.03 |

Raw JSON at `runs/m2_report.json`.

## Findings

### 1. Sharp mid-layer transition at layer 13

Every decoder metric jumps discontinuously between layer 12 and 13. LR LOO
goes 0.06 → 0.32. Attribute decoders jump from the 50–60% band to the 70–89%
band. This is the residual stream going from "near-token features" to
"category/semantic features" — consistent with the standard mid-to-late
transformer story. Peak candidate-identity signal sits around layers 14–17.
Peak attribute signal is slightly later (17–22) and stays high through L26.

### 2. NC-vs-LR gap — suggestive but not yet diagnostic

Nearest-centroid tops out at 0.12 (vs. chance 0.05); multinomial logreg on
the same activations hits 0.38. Two things could produce this 3× gap:
(a) the secret's representation at Ready isn't a single direction per
candidate — it's a richer manifold that cosine-to-mean can't capture;
(b) NC is simply underpowered at 8 samples/class and will close the gap
with more data. We cannot distinguish these at M2 sample sizes. M3 at
~100 runs/class is the test: if NC stays below LR there, (a) gains weight;
if NC catches up, the "single concept direction" story is back in play.
**Do not lean on the attribute-bundle framing for the headline yet.**

### 3. Zero nearest-centroid transfer calibration→self-chosen (ignoring L0)

SC-agree is 0.00–0.05 everywhere except L0 (0.41, but L0 is the raw
embedding; its "signal" is most likely surface-lexical leakage from the
tiger-heavy self-chosen distribution — see finding 4). **Important caveat:
this is measured only with nearest-centroid.** LR and per-attribute transfer
have not yet been measured, so "zero transfer across readouts" is not
established — only "zero NC transfer" is. Given NC underperformed on the
calibration task itself, it's a weak choice as the sole transfer metric.
M3 should measure transfer with every readout (NC, LR, binary attributes)
before concluding that calibration→self-chosen transfer fails.

Still, the NC result is consistent with D-01's stance that calibration is
infra, not the scientific result. The planned M3 fix — question turns that
force the Ready-state commitment — is independently motivated.

### 4. Heavy prior bias under self-chosen — tiger 40%

| animal | count |
|---|---|
| tiger | 16 |
| eagle | 5 |
| owl | 4 |
| elephant | 4 |
| crocodile | 3 |
| bat, cat, bee, dolphin, shark, gorilla | 1–2 each |
| (reveal failed to parse) | 1 |

Gemma 3 1B picks tiger 40% of the time when asked to choose "one animal"
from the list, despite displayed-order randomization per run. If the real
study uses self-chosen data with this bias, per-class variance will be
awful (many tigers, barely any horse/kangaroo/salmon). **At M3/M4 we must
either (a) scale to Gemma 3 4B/12B and re-measure the bias, (b) inject a
"be diverse across runs" nudge, or (c) accept the bias and oversample
under-represented candidates.** Log this as an open question for M3.

## Watch-outs for the next agent

- Two bugs were hit and fixed in this session: `apply_chat_template` returns
  `BatchEncoding` in newer transformers (extracted `.input_ids`), and sklearn
  `LogisticRegression` dropped the `multi_class` kwarg (removed). Look for
  similar silent-API-drift issues when `uv` upgrades `transformers` or
  `scikit-learn`.
- `runs/` is gitignored; the 200 .pt files (~50 MB total) are ephemeral.
  `runs/m2_report.json` is the durable decode artifact.
- Tiger-bias in self-chosen is load-bearing for M3 design. Do not let the
  next-agent spec 40 self-chosen runs and expect >2 per class.
- Decoder-vs-reveal agreement is the least-noisy transfer signal we have now;
  layer 0's 0.41 is an artifact, not a result.

## Next concrete step

M3: move to `google/gemma-3-4b-it` on TSUBAME A100 (the `tsubame-ssh` skill
handles remote runs). Full calibration at ~100 runs per candidate (2000
total) and introduce question turns so the Ready-state commitment is
*forced*, not optional. Then revisit whether NC vs LR gap persists at 4B
and whether the tiger bias shrinks.
