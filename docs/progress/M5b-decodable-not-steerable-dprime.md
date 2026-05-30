# Decodable but not steerable — d-prime reconciliation (Appendix A)

**Headline:** the apparent paradox (class is decodable at `end_ready` LR ~0.51
yet the difference-of-means norm there is only ~5, too small to steer) resolves
cleanly: **decodability tracks the scale-free SNR (d-prime), steerability tracks
magnitude relative to the residual (leverage).** Across the 6 realized classes
(drop shark, 20/class, mean over the 15 class pairs):

| position (layer) | mean ‖x‖ | raw ‖μₐ−μᵦ‖ | leverage (raw/‖x‖) | d-prime (raw/σ_within) |
|---|---:|---:|---:|---:|
| end_ready (L16) | 3,418 | 5.29 (med 4.43) | 0.00155 | 0.93 (med 0.66) |
| pre_answer_q1 (L38) | 48,631 | 759 (med 497) | 0.0156 | 1.37 (med 1.41) |
| pre_answer_q4 (L38) | 49,215 | 1,568 (med 1481) | 0.0319 | 2.49 (med 2.30) |

**Reading:**
- Raw between-class norm spans ~300× (5 → 1568), but scale-free **d-prime spans
  only ~2.7×** (0.93 → 2.49). The probe reads d-prime, so `end_ready` is in the
  same order of magnitude as the question anchors → it decodes.
- ~14× of the raw gap is just overall activation magnitude (‖x‖ grows
  3,418 → 48,600 from L16 to L38). The remainder is that the class direction is
  a genuinely smaller *fraction* of the residual at Ready: steering **leverage**
  is 0.0016 vs 0.016–0.032, i.e. ~10–20× lower than at the question anchors.
- So `end_ready` is **readable** (comparable d-prime) but **not pushable** (tiny
  leverage), the textbook regime for a faint, low-variance linear direction.
  This is the quantitative backing for the steering null at L16 (0–1% flip) vs
  35–70% at the question anchors, and for "decodability ≠ causality."

**Method:** `scripts/fisher_separation.py`, job 7801671 (cpu_4). For each class
pair, project residuals onto the unit mean-difference direction; d-prime =
‖μₐ−μᵦ‖ / pooled within-class std of those projections. Inputs:
`runs/positional_residuals/27b_default_n80_v2`. (First attempt 7801667 OOM'd:
`tensor[...].numpy()` returned a view that pinned every 21MB/file tensor; fixed
by copying the slice.)

Used as **Appendix A** of the blog. Related: [[reference_gemma_scope2_27b]],
`docs/progress/M5-sae-27b-endready-L16.md`,
`docs/progress/M5b-steering-position-bisection-27b.md`.
