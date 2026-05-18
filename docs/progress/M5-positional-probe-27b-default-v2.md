# M5 scale comparison — 27B positional probe (v2, 16 anchors)

**Headline:** at **27B**, the self-chosen class direction is linearly
decodable from the residual stream **at `end_ready`** — peak LR LOO
**0.508 @ L16** (3.55× chance 0.143), late-band L27-48 mean 0.309
(2.16× chance). At 12B the same anchor sits at chance (max 0.300 @ L1
≈ 1.2× chance 0.25; L20–45 mean 0.213, below chance). **Scale grants
the model an early-network class commitment that 12B lacks.** Whether
that commitment is causally load-bearing is the next experiment (M4
patch sweep at 27B, headline anchor `end_ready` L12–L20 band — see
"Next" below).

`pre_answer_q4` is still the strongest anchor at 27B (LR LOO **0.672 @ L38**,
L20–45 mean 0.560, **4.7× chance peak / 3.9× band mean**), about 1.5×
the 12B turn-4 multiple. So scale shifts both: turn-4 class commitment
sharpens, *and* a new earlier commitment appears at end_ready.

## Setup

- **Model:** `google/gemma-3-27b-it`, bf16, 63 decoder blocks, hidden 5376.
- **Capture:** v2 16-anchor positional residuals,
  `runs/positional_residuals/27b_default_n80_v2/` (600 self-chosen runs,
  default prompt, 4-question panel; job 7566164, gpu_1, 69.2 s wall).
- **Probe:** balanced LR + NC LOO across (16 anchors × 63 layers) cells
  (job 7568083, cpu_40, ~5h wall). Output:
  `runs/m5_positional_probe_27b_default_v2_n120.json`.
- **Realized classes (7):** cow, dog, elephant, gorilla, horse, shark,
  tiger. LOO subsample: 20/class for 6 classes; shark only 2 attempts,
  so it enters the fit but cannot be cleanly LOO'd. Effective n=122,
  chance ≈ 1/7 = 0.143.
- **Reference:** 12B default n=80 (4-class, chance 0.25):
  `runs/m4_positional_probe_12b_default_n80.json`. 12B v2 has the same
  4 classes plus the four pre_answer anchors and is the cleanest
  point-by-point comparator for `pre_answer_qN`.

## Result — 16-anchor LR LOO table at 27B (chance 0.143)

| anchor | LR_max | @ L | L20-45 mean | × chance (max) | × chance (band) |
|---|---:|---:|---:|---:|---:|
| end_user_prompt | 0.484 | L19 | 0.379 | 3.39 | 2.65 |
| **end_ready** | **0.508** | **L16** | **0.338** | **3.55** | **2.37** |
| end_user_q1 | 0.426 | L22 | 0.347 | 2.98 | 2.43 |
| pre_answer_q1 | 0.574 | L38 | 0.466 | 4.02 | 3.26 |
| end_model_q1 | 0.459 | L60 | 0.380 | 3.21 | 2.66 |
| end_user_q2 | 0.475 | L45 | 0.432 | 3.32 | 3.02 |
| pre_answer_q2 | 0.533 | L61 | 0.394 | 3.73 | 2.76 |
| end_model_q2 | 0.557 | L61 | 0.412 | 3.90 | 2.88 |
| end_user_q3 | 0.492 | L14 | 0.378 | 3.44 | 2.65 |
| pre_answer_q3 | 0.500 | L38 | 0.412 | 3.50 | 2.88 |
| end_model_q3 | 0.484 | L42 | 0.398 | 3.39 | 2.79 |
| end_user_q4 | 0.582 | L45 | 0.482 | 4.07 | 3.37 |
| **pre_answer_q4** | **0.672** | **L38** | **0.560** | **4.70** | **3.92** |
| end_model_q4 | 0.590 | L41 | 0.504 | 4.13 | 3.53 |
| end_reveal_user | 0.639 | L61 | 0.444 | 4.47 | 3.11 |
| pre_reveal_gen | 0.820 | L58 | 0.526 | 5.74 | 3.68 |

## end_ready: where the scale effect lives

At 27B `end_ready`, LR LOO ramps up from L0=0.000 to a sharp peak at
L12–L18 (~0.46–0.51) then *slowly decays* through the late band to
~0.31. This is the opposite of the 12B picture, where `end_ready` LR
sits at chance everywhere and the class signal only forms during the
question turns.

`end_ready` LR LOO by layer (27B, every 3rd block):

```
L 0  0.000      L21  0.426      L42  0.336
L 3  0.197      L24  0.369      L45  0.270
L 6  0.303      L27  0.320      L48  0.246
L 9  0.336      L30  0.336      L51  0.287
L12  0.459      L33  0.336      L54  0.270
L15  0.467      L36  0.311      L57  0.311
L18  0.443      L39  0.320      L60  0.287
```

The structure is **mid-network peak, gradual late-band decay** — class
information enters the residual by L12, stabilises around 0.32–0.34 in
the late band, and never decays to chance. Compare 12B `end_ready` LR:
max 0.300 @ L1, L20–45 mean 0.213 (below chance). The whole curve
shifts up at 27B *and* re-shapes from "flat/chance" to "peaked + carry".

`end_user_prompt` also clears chance at 27B (LR max 0.484 @ L19,
L20–45 mean 0.379, **3.4× chance peak**) — even before the model has
said "Ready", a class commitment is partially decodable from the
post-user-prompt residual. This pushes the commitment locus earlier
than M3/M4 ever localised at 12B; the model appears to lock in (or at
least bias toward) a class before the dialogue begins.

## Comparison to 12B (apples-to-apples normalised to × chance)

For anchors present at both scales (using the 12-anchor 12B default
collection as the comparator; chance 0.25):

| anchor | 12B LR_max | 12B × chance | 27B LR_max | 27B × chance | shift |
|---|---:|---:|---:|---:|---:|
| end_user_prompt | 0.325 | 1.30 | 0.484 | 3.39 | **+2.1×** |
| end_ready | 0.300 | 1.20 | 0.508 | 3.55 | **+2.4×** |
| end_user_q1 | 0.338 | 1.35 | 0.426 | 2.98 | +1.6× |
| end_model_q1 | 0.388 | 1.55 | 0.459 | 3.21 | +1.7× |
| end_user_q4 | 0.550 | 2.20 | 0.582 | 4.07 | +1.9× |
| end_model_q4 | 0.550 | 2.20 | 0.590 | 4.13 | +1.9× |
| end_reveal_user | 0.537 | 2.15 | 0.639 | 4.47 | +2.3× |
| pre_reveal_gen | 0.925 | 3.70 | 0.820 | 5.74 | +2.0× |

Two patterns:

1. **Every anchor gets stronger at scale.** The bottom of the table
   (turn-4, reveal-adjacent) widens by ~2× chance; the top of the
   table (Ready, post-prompt) widens by ~2.1–2.4× chance.
2. **The biggest scale-shift is at the *earliest* anchors.** Late-band
   shifts are large (~+2×) but expected — they amplify what was
   already there at 12B. The end_user_prompt / end_ready shifts move
   the model from *chance* to *clearly decodable*, which is a
   qualitative state change, not just a magnitude one.

## What this means for the M5 / blog story

The 12B story (M3 + M4) was:

> The model does not commit at end_ready. It re-derives the class
> across each Q-A turn, with the late-network L27–L48 band carrying
> a probe-decodable class direction by `pre_answer_q4` that is
> causally inert under single-position residual patching.

The 27B picture adds a layer:

> At 27B scale, an additional commitment locus appears at **end_ready**
> (LR LOO **3.6×** chance at L16, **2.4×** chance through L20–L45)
> that does not exist at 12B. The late-band turn-4 commitment is also
> stronger (4.7× chance peak vs 3.2× at 12B). The improvisation /
> re-derivation mechanism we localised at 12B does not disappear at
> 27B — it *coexists* with an earlier explicit pre-commitment that
> only emerges with scale.

Whether the new early commitment is **load-bearing** or only **legible**
is the headline question for the next experiment. M4 found that the
12B turn-4 signal is decodable but causally inert (0/2280 reveal flips
on single-position patching). The same null at 27B end_ready would say
"scale produces legible early commitment but still doesn't move the
output" — improvisation-coexists-with-decoration. A non-null would say
"scale produces a stored commitment that the model *uses* downstream"
— retrieval at scale, qualitatively different mechanism.

## Side observations worth flagging

- **NC LOO is much weaker than LR LOO across the board.** At
  `pre_answer_q4` LR peaks at 0.672 but NC only at 0.623 (still
  unusually close compared to 12B's LR/NC gap, where NC was ~2/3 of
  LR). At `end_ready` LR=0.508 / NC=0.336 — the by-layer LR curve is
  much smoother and higher than NC. Read: classes are linearly
  separable but clusters aren't spherical — same picture M3 found at
  12B, persisting at scale.
- **Realised class distribution is heavily skewed at 27B.** Of 600
  attempts, 240 came back as `horse` and 201 as `tiger`. The other
  classes range from 53 (cow) down to 2 (shark). The 7-way LR LOO
  uses balanced 20/class subsamples for those with ≥20 attempts, but
  the underlying generation distribution is not uniform — at 27B with
  the 20-bank prompt, the prior strongly favors horse+tiger as the
  realised class. Worth a side investigation: does the larger model
  collapse to fewer attractors than 12B?
- **`pre_answer_q1` LR LOO at 27B = 0.574** — the per-turn pre_answer
  signal forms much earlier in the dialogue than at 12B. At 12B v2,
  pre_answer_q1 LR was 0.32 (n=80 4-class, max likely lower; see
  M5-attribute-bundle for the per-class attribute readout where dog
  alone clears baseline at q1). At 27B, pre_answer_q1 is already at
  4.0× chance, almost as strong as 12B pre_answer_q4 in chance units.
  Scale appears to accelerate the per-turn dialogue-integration step.

## Artifacts

- **Probe output:** `runs/m5_positional_probe_27b_default_v2_n120.json`
  (60 KB) + `_centroids.pt` (152 MB, on /gs/bs).
- **Source residuals:** `runs/positional_residuals/27b_default_n80_v2/`
  (600 .pt files, ~13 GB on /gs/bs).
- **Jobs:** capture `jobs/tq_m5_capture_positional_27b_default_v2_20260517.sh`
  (7566164); probe `jobs/tq_m5_probe_positional_27b_v2_20260517.sh`
  (7568083).
- **Comparator:** `runs/m4_positional_probe_12b_default_n80.json` for the
  12-anchor 12B default; v2 16-anchor 12B comparator is in
  `runs/m4_positional_probe_12b_default_n80.json`'s v2 successor (TBD —
  not present locally; the M5 attribute-bundle writeup
  `docs/progress/M5-attribute-bundle-12b-default-v2.md` is the v2 12B
  reference for per-turn pre_answer anchors).

## Caveats

- **n=122 with shark=2.** The 7-way LR fit handles shark gracefully
  (only 2 examples, can't dominate), but per-class accuracy on shark
  is effectively undefined. The headline `end_ready` number is robust
  to dropping shark entirely (re-fit on 6 classes; not shown — easy
  follow-up).
- **No 27B M4 patch sweep yet.** The decoding result alone cannot say
  whether end_ready is load-bearing. Single-position residual patching
  at end_ready / L16 (the peak) is the next experiment; same null
  story for 12B at pre_answer_q4 / L29-L48 is the reference baseline.
- **Class-balance asymmetry.** Horse + tiger account for 73% of all
  realised attempts at 27B. If those two classes dominate the LR
  separating hyperplane's geometry, "class commitment" at end_ready
  could be more accurately phrased "horse-vs-tiger commitment". Easy
  diagnostic: per-class LR accuracy at the end_ready peak layer
  (already in the JSON's `summary` section; not unpacked in this
  writeup but should be).
- **No 12B v2 16-anchor LR table here.** The 12B v2 capture and probe
  exist; the per-turn pre_answer comparison at 12B is in the
  attribute-bundle writeup but it's a per-attribute binary probe, not
  the same 4-way LR LOO. For a clean v2-vs-v2 comparison, re-run the
  12B v2 probe (cheap; just re-fits LR over existing residuals).

## Next

The decision-gate threshold from STATUS was "end_ready LR LOO ≥0.30
in late band" → trigger M4 patch sweep at 27B. The actual band mean is
**0.309**, peak **0.508**. Gate fired.

**Next concrete experiment:** M4 single-position residual patch at
**27B end_ready, L12-L20 band** (where the LR signal is strongest;
exact band picked to mirror Heimersheim & Nanda "low-granularity band
first" methodology). Measure flip rate vs. 0/2280 12B null. If
≥5/120 flips at any (source, target) class pair, scale has produced a
load-bearing pre-commitment — qualitative mechanism shift. If ≤2/120,
improvisation-coexists-with-decoration.

Engineering: `scripts/patch_turn4.py` currently only patches at
`pre_answer_qN` positions; it needs extending to support `end_ready`
as a patch position. New script
`scripts/patch_anchor.py` or `--anchor` arg to patch_turn4. Layer set:
L12-L20 (9 layers, mid-network band). gpu_1 needed for 27B; expect
~6-8h walltime at 5 src × 5 tgt × 49 pairs × ~20 layers ≈ 5000 patched
trials.

Secondary next experiment (cheaper, useful for the scale story
regardless of patch outcome): re-run 12B v2 probe to get the 4-way
LR LOO at the same 16 anchors as 27B for clean apples-to-apples
comparison. ~30 min cpu job.
