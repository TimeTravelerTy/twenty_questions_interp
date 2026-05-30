# STATUS

> **First file any agent reads.** The `Next concrete step` is always actionable
> without reading anything else. Update the `Last updated` line on every session.

**Current milestone:** **Blog drafting in progress (M4+M5+M5b+Exp2). Two new supporting analyses landed.** (1) **SAE features at the question anchors** (job 7797717, Gemma Scope 2 `layer_40_width_65k_l0_medium`, capture-index 41): at `end_ready` only formatting features fire, but at `pre_answer_q1` (same layer) the loud features become yes/no-answering features PLUS "mammals" / "animal classification and biology"; per-turn they track the fixed question content — Q1 mammal→"mammals", Q4 four-legs→"how it walks"/"anatomical parts", and the two "No" questions (bird?, water?) are dominated by negation features. The 4 questions are FIXED and the realized mammals answer them identically (Yes/No/No/Yes; only shark n=2 differs), so the transcript barely distinguishes the picked animals — the distinction lives only in the residual. Heatmap fig `docs/blog_figures/fig3_features_by_turn.png`. (2) **d-prime reconciliation (Appendix A)**: decodable-but-tiny-norm at `end_ready` resolves as decodability∝SNR vs steerability∝leverage. d-prime 0.93 (Ready L16) / 1.37 (q1 L38) / 2.49 (q4 L38) spans ~2.7× while raw ‖μₐ−μᵦ‖ spans ~300× (5/759/1568); ~14× of that is activation scale (‖x‖ 3.4k→48.6k), rest is genuinely lower leverage at Ready (0.0016 vs 0.016–0.032). Writeup `docs/progress/M5b-decodable-not-steerable-dprime.md`, job 7801671. Blog data figures committed in `docs/blog_figures/` (fig1 decodability, fig2 steering, fig3 feature-by-turn) via `scripts/blog_figures.py`. **Next: finish stitching the blog; optional Exp 1 (linear attribute axis at 27B).**

**Prior milestone:** **Exp 2 done — 27B SAE at `end_ready` (probe peak) does NOT sparsify the class direction; the features that fire there are chat-template/format features, not class or attribute.** At resid_post[16] (Gemma Scope 2 `layer_16_width_65k_l0_medium`, FVU 0.0048), matched 6-class balanced LR LOO (chance 0.167): **dense residual 0.525 (3.15× chance) vs sparse SAE 0.300 (1.80× chance)** — the SAE keeps only ~57% of the dense class accuracy. Only 45/65,536 features active at `end_ready`, all firing at rate 1.00 across every class with <1% per-class mean variation. Prong B (Neuronpedia `16-gemmascope-2-res-65k`): top-activating features are all generic format/web-text ("names of people", "XML tags and identifiers", "HTML tags and URLs", "punctuation and separators", "code files and definitions"); the top class-discriminative feature (ANOVA) is "loop termination conditions". **Zero animal-class features, zero attribute features.** This replicates the 12B SAE-negative at the scale where the linear signal is strong (12B `end_ready` was at chance, so its SAE-negative was uninformative). Cleanest representational statement of "legible to a linear probe ≠ sparse interpretable feature". Bears on Exp 1: no SAE attribute features fire at Ready, so the "probe reads attributes" idea gets no sparse-code support — Exp 1 should test a *linear* attribute axis, not an attribute feature. Writeup: `docs/progress/M5-sae-27b-endready-L16.md`. Jobs 7796102 (firings), 7796138 (dense baseline). **Next: Exp 1 (linear attribute probing at 27B) OR fold Exp 2 into the blog's legibility chapter.**

**Prior milestone:** **M5b fully closed — position-bisection shows setup-time class info is readable but not load-bearing, while pre-answer/readout directions are causal. Ready to draft the blog.** End_ready L16 rollout steering (job 7783019) is decisively null: 0/192 flip→src at α=1, 1/192 at α=3, 2/192 at α=10, with only 3/2304 answer slots flipped across all alphas and clean baselines. The old concern that teacher-forced Q/A masked causality is handled: this was `--answer-rollout` + `scope=from_anchor`, so the model regenerated answers under steering. The surprise is real but coherent: end_ready median contrastive direction norm is only ~3.9, versus ~518 at q1 and ~1354 at q4. New bisection jobs at `pre_answer_q1..q4` show the direction becomes causal once the dialogue trajectory begins: q1/L38 α=3 flips 35.4% to source, q2/L61 α=10 flips 61.5%, q3/L38 α=10 flips 56.8%, q4/L38 α=3 flips 70.3%, and pre_reveal_gen/L58 α=3 flips 80.2%. Answer-consistency analysis shows reveal steering is not identical to rewriting the whole hidden game state: only 22% of answer slots are source/target-diagnostic, and many steered reveals remain target-compatible in the regenerated answers. Final interpretation: the model does not store a stable, load-bearing setup-time commitment in the residual; it re-derives/assembles class state during the trajectory, and that state becomes increasingly causal as the reveal approaches. New writeup: `docs/progress/M5b-steering-position-bisection-27b.md`; updated `docs/progress/M5b-steering-27b-endready-L16.md`. **Next: draft the M4+M5+M5b blog post. No more M5b GPU experiments needed.**

**Prior milestone:** **M5b: L58 steering broadly positive but tautological — it's an unembed perturbation, not evidence of hidden commitment. Combined with L16 steering null + M5 patch nulls, the story closes coherently.** Job 7782821 (CAA steering at pre_reveal_gen L58, the strongest probe peak with LR 0.820/5.74× chance, scope=from_anchor): aggregate flip→src 80% at α=3, 75% at α=10. But this is the position+layer whose residual passes through one final layer norm + unembed to produce the reveal logits — adding `α·(μ_src−μ_tgt)` there is essentially "tilt the unembed input toward the source-class direction." Per-source-class detail: horse and tiger directions are clean (100% flip→src at α=3); cow/dog/elephant/gorilla are 93% with small horse leakage; **shark is anomalous** — flips to "salmon" (a 20-bank class not in our realized set) at α=3, the [steering off-manifold critique](https://arxiv.org/html/2604.09839v1) fired in miniature. The interpretation: the L58 contrastive direction *is* the unembed direction (probe at LR 0.820 confirms it), so a positive there proves only that the readout layer carries class info — not that early-position class directions are load-bearing. Combined with L16 null + M5 full-residual L1-L62 at end_ready null: the model recomputes class identity at each later position via attention over the prompt, the probe direction is a *summary* of that computation, and only at the readout layer can a direct injection trivially affect output. M5b coherently closed.

**Prior milestone (also M5b):** **M5b extended (2026-05-28): late-band multi-anchor patch + CAA-style steering both substantially null.** Two new jobs added to the M5b chain. (1) Job 7782327 — multi-anchor rollout at L35-L45 (probe peak band) — gives 1002/1024 (97.9%) kept-target, 6/870 off-diag flips, 6/3480 answer-flip slots. Marginally more movement than L14-L18 but still null. The new movement is class-pair-specific: elephant→shark/attempt_431 and tiger→shark/attempt_431 both flip the reveal to **tiger** with the same `[True, False, False, True]` regenerated-answer fingerprint — large-mammal sources unlock a "tiger attractor" in one specific fragile target, not a general patch effect. (2) Job 7782428 — CAA steering at end_ready L16 (probe peak), `α ∈ {1, 3, 10}`, mode=add — gives 93-96% kept-target across α, 0.5-1.0% flip-to-source, and the only textbook-clean cell is **tiger→gorilla** (0/5, 1/5, 2/5 across α monotonic; both large mammals — semantically-meaningful sub-axis). Drift-to-other rises with α (3.1%→5.7%), consistent with α=10 starting to push residuals off-distribution rather than activating a clean class attractor. **Take:** the probe-decoded class direction is overwhelmingly epiphenomenal — even direct injection at the probe peak with α up to 10 fails to redirect the reveal. Structurally different from refusal-direction findings (Arditi et al.), consistent with our hypothesis that class commitment in 20-Questions is held in the prompt and re-derived via attention, not in a single subspace the probe captures. New writeup: `docs/progress/M5b-steering-27b-endready-L16.md`. Late-band patch added to `M5b-patch-answer-rollout-27b-endready.md`.

**Prior milestone (also M5b):** **M5b CLOSED — both answer-rollout and multi-anchor escalation confirm the M5 null.** Two jobs at L14-L18 with `--answer-rollout`: single-anchor end_ready (7779768) and multi-anchor end_ready + end_model_q1..q4 (7779896). Results are virtually identical: 1007/1024 (98.3%) kept-target in both, off-diag flip-to-source 2/870 vs 3/870, off-diag answer-flips identical at 3/3480 slots (0.086%), baseline non-determinism 0/32. The multi-anchor extension adds exactly one extra reveal flip (cow → horse/attempt_593, with 0 answer flips — the known boundary-fragile target re-triggering, not a patch-driven rerouting). The tiger/attempt_009 → shark/attempt_431 hit from the single-anchor run persists with the same regenerated-answer pattern in multi-anchor — same trial, same mechanism, not amplified. Skipped full-residual L1-L62 as off-manifold per Heimersheim & Nanda. Verdict: across two protocol upgrades (rollout fixes the teacher-forced-(Q,A) confound, multi-anchor extends across every model-side position) the linearly-decodable class direction at end_ready (LR 0.508 @ L16) is epiphenomenal for the model's downstream behavior, full stop. M5+M5b joined as the patch story for the blog. Writeup: `docs/progress/M5b-patch-answer-rollout-27b-endready.md`.

**Prior milestone:** **M5 scale comparison — DONE. Scale shifts class-info *legibility*, not its causal role. Improvisation is scale-robust.** At 27B the residual carries decodable class information at end_ready (LR 0.508 @ L16, 3.55× chance) where 12B was at chance — but the legible direction is **epiphenomenal** for the reveal. Six escalating causal interventions are all null: 12B full-residual L1-L48 @ pa4; 27B band patches @ pa4 (L27-L62) and @ end_ready (L12-L20); 27B multi-anchor L12-L48 @ end_ready + end_model_q1–q4; 27B full-residual L1-L62 @ pa4; 27B full-residual L1-L62 @ end_ready. Each produces **0 off-diagonal class transfers** across all target runs except the single boundary-degenerate target `attempt_593`. STATUS's prior "3/870 flips, all from cow sources" was a metric artifact — `attempt_593` flips content-agnostically: under L1-L62 full-residual replacement it splits ~50/50 horse/cow regardless of source class (incl. horse→horse self-patches), so it carries zero class signal and is excluded. Unpatched baselines are stable (0/32 drift in every 27B job). The improvisation/re-derivation mechanism dominates at both scales; scale rotates more class information into linearly-readable subspaces without changing how the model uses it downstream. Headline writeup: `docs/progress/M5-scale-improvisation-headline.md`. Strand writeups: `M5-patch-27b-end_ready-pa4.md`, `M5-positional-probe-27b-default-v2.md`, `M5-sae-residual-misaligned-12b-L31-L41.md`.

Per-anchor 27B headlines (LR LOO; chance 0.143):

| anchor | LR max | @ L | L20–45 mean | × chance peak |
|---|---:|---:|---:|---:|
| end_user_prompt | 0.484 | L19 | 0.379 | 3.39 |
| **end_ready** | **0.508** | **L16** | **0.338** | **3.55** |
| pre_answer_q1 | 0.574 | L38 | 0.466 | 4.02 |
| **pre_answer_q4** | **0.672** | **L38** | **0.560** | **4.70** |
| pre_reveal_gen | 0.820 | L58 | 0.526 | 5.74 |

Every anchor strengthens by ~+1.6–2.4× chance relative to 12B; the **biggest qualitative shift is at end_user_prompt + end_ready**, which go from at-chance (12B) to clearly decodable (27B). Late-band turn-4 also sharpens (12B 3.15× → 27B 4.70× chance at pre_answer_q4 peak).

**Prior milestones (still locked):**
- *M5 SAE-negative writeup* (2026-05-17): `docs/progress/M5-sae-residual-misaligned-12b-L31-L41.md`. L31 + L41 default Gemma Scope 2 SAEs fail to sparsify the M3 class direction at every pre_answer_qN anchor; residual LR 0.79 → sparse LR 0.21–0.33; all-four-turn top-10 class-discriminative intersection = ∅ at both layers.
- *M5 attribute-bundle 12B v2*: `docs/progress/M5-attribute-bundle-12b-default-v2.md`. Dog clears pre_answer_q1 (0.801, baseline 0.75); cow/elephant/horse only clear baseline at turn-4 (0.835–0.885).

Side-by-side, balanced 20/class LR LOO (chance 0.25; M3 residual probe at L31/q4 = 0.79):

| anchor | L31 sparse LR | L31 active feats | L41 sparse LR | L41 active feats |
|---|---|---|---|---|
| pre_answer_q1 | 0.325 | 89 | 0.287 | 54 |
| pre_answer_q2 | 0.325 | 78 | 0.287 | 62 |
| pre_answer_q3 | 0.212 | 90 | 0.275 | 97 |
| pre_answer_q4 | 0.312 | 80 | 0.325 | 51 |

Top-10 cross-turn intersection is **∅** at both layers (only adjacent turns ever share even one feature). Read: at default Gemma Scope 2 layers, the self-chosen class direction the M3 linear probe finds is **residual-distributed, not sparsely encoded**, and what little signal the SAE picks up is turn/position-specific rather than class-specific.

Phase A first cut at L31/q4 (commit 52cd9ce): A4 fails by 0.48. Phase A3.2 turn-progressive at L31 (commit 8b1acf0) and L41 second cut (this commit): both fail at every turn.

**Next: write the blog draft.** M5 is closed — all three strands (decoding
probe, SAE-negative, patch sweep) are done and the headline writeup
unifies them. The blog chapter is: M4 establishes improvisation; M5 shows
it is scale-robust *and* that the decodable class direction is
epiphenomenal. Draft `docs/progress/` → blog post unifying M4 + M5.

No further M5 experiments are needed. Deferred scope unchanged (see below).

**Done this session:**
- Extended `scripts/patch_anchor.py`: `--anchor` accepts a comma-separated
  list for simultaneous multi-anchor patching (single-anchor behavior
  unchanged). Smoke 7737394 validated the multi-anchor path.
- Submitted + completed 3 new 27B patch jobs on gpu_1: multi-anchor L12-48
  @ end_ready+end_model_q1–q4 (7737407), full-residual L1-62 @ pa4
  (7737408), full-residual L1-62 @ end_ready (7737409). All null:
  0/843 off-diagonal flip-to-source excluding `attempt_593`.
- Resolved the `attempt_593` replay analytically from existing JSONs — the
  horse→horse self-patch flip is a stronger fragility control than a
  baseline replay; no compute spent.
- Headline writeup `docs/progress/M5-scale-improvisation-headline.md`.

**Prior session (M5 strands 1–2 + first patch sweep):**
- 27B v2 capture + probe (job 7568083): end_ready LR 0.508 @ L16, pa4 LR 0.672 @ L38.
- 12B v2 16-anchor probe (job 7654195): end_ready 1.10× chance, late-band below chance.
- `scripts/patch_anchor.py` created (generic v2-anchor patcher).
- 27B band patches: Exp-A 7654936 (pa4 L27-62), Exp-B 7654937 (end_ready L12-20).
- Writeups: `M5-positional-probe-27b-default-v2.md`, `M5-patch-27b-end_ready-pa4.md`, `M5-sae-residual-misaligned-12b-L31-L41.md`.

Plan: `~/.claude/plans/check-the-latest-status-bright-horizon.md`. Scope deferred (unchanged): Phase B1 steering (no candidate features), B2/B3 (M5b), `mlp_out`/`attn_out` SAEs at L31/L41 (would tell us about per-component sparsity but doesn't change the residual-stream story).
**Last agent:** Claude
**Last updated:** 2026-05-30 (Blog drafting + two supporting analyses. SAE-at-question-anchors (job 7797717, L40): formatting-only at Ready → "mammals"/"animal biology" + yes/no features at Q1; per-turn features track the fixed questions (mammal→mammals, four-legs→"how it walks", No-questions→negation). Realized mammals answer the 4 fixed questions identically (Y/N/N/Y), so transcript barely distinguishes picked animals. d-prime reconciliation (job 7801671): decodability∝SNR (d′ 0.93/1.37/2.49) vs steerability∝leverage (0.0016/0.016/0.032); raw norm gap ~300× is ~14× activation scale + lower leverage. Blog figures in docs/blog_figures/ via scripts/blog_figures.py. New scripts: neuronpedia_label_features.py, fisher_separation.py; +--drop-class on analyze_sae_features/probe_positional_anchors. Writeups: M5-sae-27b-endready-L16.md, M5b-decodable-not-steerable-dprime.md. Next: finish blog stitch.)
**Prior agent:** Claude
**Prior updated:** 2026-05-29 (Exp 2 done: 27B SAE at end_ready L16 does not sparsify the class direction. Dense 0.525 vs sparse 0.300 at matched 6-class balanced (chance 0.167); only 45 active features, all class-invariant; Neuronpedia labels are all format/web-text, zero class/attribute features. Jobs 7796102 firings + 7796138 dense baseline. Added `scripts/neuronpedia_label_features.py`; `--drop-class` to analyze_sae_features.py + probe_positional_anchors.py. Writeup `docs/progress/M5-sae-27b-endready-L16.md`. Validated Neuronpedia `16-gemmascope-2-res-65k` == l0_medium encode via maxActApprox. Note: Gemma Scope 2 = SAE *release* for Gemma 3, not a separate model. Next: Exp 1 linear attribute probing at 27B, or draft the blog.)
**Prior agent:** Codex
**Prior updated:** 2026-05-28 (M5b final bisection complete. Fixed `steer_class_direction.py` rollout `from_anchor` behavior for late anchors; submitted q4/q3/q2/q1 steering jobs 7783472/7783603/7783776/7783903. Results: q1 α=3 35.4% flip→src, q2 α=10 61.5%, q3 α=10 56.8%, q4 α=3 70.3%; end_ready L16 remains 0–1% despite same rollout protocol; pre_reveal L58 remains 80.2% at α=3. Added `scripts/analyze_steering_answer_consistency.py`; answer consistency shows many reveal flips do not fully rewrite regenerated answer policy. New writeup `docs/progress/M5b-steering-position-bisection-27b.md`; updated `M5b-steering-27b-endready-L16.md`. M5b is no longer structurally open. Next concrete work is the M4+M5+M5b blog draft.)
**Prior agent:** Claude
**Prior updated:** 2026-05-28 (M5b: L58 steering result (7782821) is broadly positive but tautological — adding the contrastive direction at the position whose residual *is* the unembed input trivially perturbs the readout. Per-source pattern (horse/tiger 100% flip, shark→salmon at α≥3) confirms off-manifold steering caveat. CPU probe analysis 7782820 stuck in cpu_80 queue; resubmitted as 7782923 with cpu_4. M5b is now structurally complete: 6 patch protocols all null at early/mid positions, steering null at L16, steering "positive" at L58 only in the trivial unembed-perturbation sense. Story: model recomputes class via attention at each position, probe directions are summaries not substrates. Ready for the M4+M5+M5b blog draft once 7782923 lands the per-class probe analysis for the writeup.)
**Prior agent:** Claude
**Prior updated:** 2026-05-28 (M5b extended with late-band multi-anchor patch (7782327, L35-L45, null with class-pair-specific shark/attempt_431 "tiger attractor") and CAA-style probe-direction steering (7782428, end_ready L16 α∈{1,3,10}, weakly causal only for tiger→gorilla with monotonic α-dependence). Steering decisive against the probe direction being broadly load-bearing — 0.5-1.0% flip-to-source across α with drift rising into off-distribution territory. New script `scripts/steer_class_direction.py` (CAA add/ablate modes). Open follow-ups noted: ablate-mode test, layer sweep at L38/L58, larger α to find breakdown point.)
**Prior agent:** Claude
**Prior updated:** 2026-05-28 (M5b CLOSED. Added multi-anchor escalation job 7779896 (end_ready + end_model_q1..q4, same L14-L18 band, same `--answer-rollout`). Result virtually identical to single-anchor: 1007/1024 kept target, 3/870 off-diag flips (one extra is into attempt_593 fragility), same 3/3480 answer-flips, same persistent tiger/attempt_009 → shark/attempt_431 hit. Fix along the way: `_find_anchors_relaxed` was returning None on any missing anchor, which would have killed every multi-anchor rollout trial on step 1 (end_model_qi anchors don't exist until A_i is generated); changed to return the present-anchor subset. Caught preemptively before the job ran. Skipped full-residual L1-L62 as off-manifold per Heimersheim & Nanda. M5+M5b is the final patch story for the blog. List-permutation mechanism note added to M5 headline writeup explaining why "kept target under patch" is the prediction.)
**Prior agent:** Claude
**Prior updated:** 2026-05-28 (M5b done. Job 7779618 returned empty because `_find_anchors` requires 11 EOTs and bails on partial rollout contexts — fixed via `_find_anchors_relaxed` in `patch_anchor.py` (EOT-index lookup tolerant of fewer turns). Resubmitted as 7779768; 1024 trials in 616s wall (10min compute + 6min model load). Headline: 98.3% kept target, 2/870 off-diagonal flips, 3/3480 answer slots flipped, baselines fully deterministic. End_ready L14-L18 patch confirmed null on answer-rollout protocol, ruling out the teacher-forced-(Q,A) confound and strengthening the M5 "improvisation is scale-robust" claim. One isolated tiger/attempt_009 → shark/attempt_431 trial flipped both axes — class-specific, not the broad fragility attempt_593 showed.)
**Prior agent:** Claude
**Prior updated:** 2026-05-21 (M5 closed. The three pending follow-ups are done: multi-anchor patch (27B, 5 sites, L12-48 — job 7737407), full-residual L1-62 at pa4 (7737408) and at end_ready (7737409). All three null: 0/843 off-diagonal class transfers excluding the boundary-degenerate target attempt_593. The attempt_593 "replay" was resolved analytically — under L1-62 full-residual replacement it splits ~50/50 horse/cow regardless of source class, incl. horse→horse self-patches, so it carries zero class signal; STATUS's prior "3/870 flips, all from cow sources" was a flip-to-source metric artifact and is corrected in the headline writeup. patch_anchor.py extended for multi-anchor (`--anchor` now comma-separated). Headline writeup `docs/progress/M5-scale-improvisation-headline.md` unifies all three M5 strands. M5 chapter is blog-ready; next is the M4+M5 blog draft.)

**North star:** *Calibration is infra; the scientific claim is self-chosen only.*
Do not headline calibration-only results.

---

## Next concrete step

**Actual current next step (2026-05-28): draft the M4+M5+M5b blog
post.** Use these source notes:

- `docs/progress/M4-*.md` for the 12B improvisation/null-patching story.
- `docs/progress/M5-scale-improvisation-headline.md` for scale-robustness.
- `docs/progress/M5b-patch-answer-rollout-27b-endready.md` for rollout patch nulls.
- `docs/progress/M5b-steering-27b-endready-L16.md` for L16 null + L58 positive.
- `docs/progress/M5b-steering-position-bisection-27b.md` for final q1-q4 bisection and answer-consistency metrics.

No more M5b GPU experiments are needed before drafting. Optional future
work (ablation, CCM mediation, larger alpha breakdown) is refinement,
not a blocker.

Prior notes, newest first:
**`docs/progress/M3-12b-selfchosen-turn4scale.md`** (n=80 scale-up on job `7232075`; turn-4 locked),
`docs/progress/M3-12b-selfchosen-turns.md` (turn-1..4 sweep on job `7230807` kept runs),
`docs/progress/M3-12b-selfchosen-direct.md` (job `7230807`; Ready direct-fit LOO local),
`docs/progress/M3-12b-selfchosen-transfer.md` (job `7230657`; plus retrospective 20-bank slice),
`docs/progress/M3-12b-pilot-readouts.md` (job `7226576`; readouts local),
`docs/progress/M3-selfchosen-20bank.md` (jobs `7223018`, `7226501`, `7226502`, `7226538`, `7226546`, `7226547`),
`docs/progress/M3-selfchosen-ready-T07.md` (job `7219788`),
`docs/progress/M3-selfchosen-ready-smoke.md` (job `7218660`),
`docs/progress/M3-h-persistence.md` (job `7218322`),
`docs/progress/M3-binding-bank-audit.md`,
`docs/progress/M3-3cond-binding-smoke.md` (job `7218265`),
`docs/progress/M3-4cond-binding-smoke.md`, `docs/progress/M3-4b-smoke-diagnostics.md`.

**Result so far (12B calibration is good; turn-4 self-chosen is the first useful probe position):**

1. **20-bank prompt beats the 4-candidate panel.** Greedy 4B no longer
   collapses to only salmon/frog. Over 200 attempts: dolphin 104 / penguin 85 /
   shark 5 / crocodile 2 / horse 2 / cow 1 / salmon 1; 7 classes appear and 5
   reach quota `n=2`.
2. **4B Ready geometry is still weak.** On the realized 5-way subset
   (`dolphin,horse,penguin,crocodile,shark`), self-chosen best post-L13 NC is
   **10.0%** (chance 20%) and post-L13 within-between contrast is **+4.70e-06**.
3. **Matched persistence is strong.** State A and B both hit **100% NC at L21**,
   cross A→B is **90%**, and State B contrast is **+4.87e-04**. Self-chosen is
   about **104× weaker than matched persistence State B** on contrast.
4. **Matched calibration still fails at 4B.** `name_paraphrase` scores only
   **34/40 = 85.0%** on the same 5-way subset, so it is not ready to standardize.
5. **12B improves self-chosen behavior, but the old 4-question panel was broken.**
   Greedy 12B over 100 attempts realizes 4 classes and reaches **37.5% NC** at
   L15, but the legacy panel is non-diagnostic there because every class shares
   the same `1,0,0,1` fingerprint.
6. **A six-question matched panel fixes that and clears the gate.** On
   `is_carnivore,is_larger_than_human,is_domesticated,lives_in_africa,produces_dairy_milk,is_ridden_by_humans`,
   matched persistence stays strong (`7226546`: 45/48 = 93.8%, NC-A 100% at L30)
   and matched calibration with **`name_paraphrase`** passes (`7226547`:
   **47/48 = 97.9%**).
7. **The narrowed 4-way self-chosen prompt is a dead end at 12B.** Re-running
   self-chosen directly on `{elephant,cow,dog,horse}` (job `7230657`) realizes
   only `cow` and `horse` over 160 attempts; `elephant` and `dog` never appear.
8. **Calibration -> self-chosen Ready transfer still fails at 12B.** On the
   balanced kept 4-way run, agreement vs reveal at `L6/L17/L27/L48` is
   NC `0.35/0.00/0.00/0.10` and LR `0.00/0.00/0.00/0.10`. On the earlier
   natural 20-bank 12B self-chosen slice (2 x `{elephant,cow,dog,horse}`),
   transfer is only `0.125-0.25`, i.e. chance.
9. **Self-chosen Ready direct-fit is also weak (decisive).** Job `7230807`,
   300 attempts on the 20-bank prompt, 40 kept runs balanced 10/class.
   LOO over all 49 layers: NC mean 0.23 / max 0.45 @ L14; LR mean 0.27 /
   max 0.45 @ L4 (chance 0.25). Best layers disagree between NC and LR,
   consistent with noise + a weak underlying signal. Calibration on the
   same 4 classes saturates at 1.00 from L6 (LR) / L27 (NC). So at 12B,
   Ready-state self-chosen geometry itself is the bottleneck, **not**
   calibration->self-chosen transfer.
10. **Turn-4 pre-answer is materially stronger than Ready.** On the same 40
    kept self-chosen runs, turn 4 pre-answer reaches NC mean **0.40** /
    max **0.625 @ L44** and LR mean **0.40** / max **0.60 @ L42**. The
    signal forms a broad late-layer band (L27-48 means: NC **0.549**,
    LR **0.539**). Turn 1 is moderate, turns 2-3 are weak. So the right
    lesson is not "later is always better"; it is that **turn 4 pre-answer**
    is the first clearly probe-usable self-chosen position we have found.
11. **This turn-4 signal is a latent-state result, not public-history leakage.**
    On the realized kept subset `{elephant,cow,dog,horse}`, the 4-question
    panel is degenerate (`1,0,0,1` for every class). So the turn-4 decode
    cannot be coming from publicly distinguishing yes/no history.
12. **Scale-up (n=80, 20/class) crystallizes the turn-4 signal decisively.**
    Job `7232075`, same 20-bank prompt, same 4-question panel. On turn 4:
    LR LOO **0.787 @ L31** / NC **0.662 @ L29** (chance 0.25). Broad
    coherent L27-48 band: **LR mean 0.731**, NC mean 0.558. Compared to
    the n=40 pilot, LR jumps +0.19 in both mean and max while NC barely
    moves — consistent with linearly separable geometry that was only
    regularization-starved at n=40. Turn 1 L27-48 LR mean 0.431 remains
    above chance but well below turn 4. The STATUS ~70% threshold for
    locking this probe position is cleared.

Bottom line: at **12B**, the self-chosen class code is decodable at
**turn-4 pre-answer, late layers (L26-L48, peaks near L29-L31)**. LR
0.79 at L31 is ~3.2x chance and the signal is coherent across depth. That
position is locked for M4.

**Pilot done (2026-04-21):** 100-run 12B `name_paraphrase` calibration on
`{elephant,cow,dog,horse}` with the six-question panel (job `7226576`). Local
Ready readouts across all 49 layers:

- **LR LOO saturates at 1.00 by L6**, stays saturated to L48.
- **NC LOO climbs 0.25→0.66 @ L7→0.93 @ L16→1.00 from L27.**
- Six binary attribute decoders all 1.00 from L7 (trivial at 4 classes × 25).

The LR ≫ NC gap at L6–L26 is still the main calibration-side structural signal:
candidate identity is linearly available from ~1/4 depth, but class clusters are
not spherical until much deeper. Transfer, however, is now the decisive result:
see `docs/progress/M3-12b-selfchosen-transfer.md`.

**Next concrete step — write the M5 negative + spin up 27B capture.**

L41 second cut returned the same hard negative as L31. Decision gate
fired: SAE-retrieval at default Gemma Scope 2 12b-it layers is dead.
The two parallel tracks:

**(a) Negative writeup.** New file
`docs/progress/M5-sae-residual-misaligned-12b-L31-L41.md`. Skeleton:
- The question (Phase A: do M3's linear-probe class directions decompose
  into sparse SAE features?).
- The result (no, at L31 OR L41, at any pre_answer_qN anchor).
- The mechanics (turn/position-specific top-features, zero cross-turn
  intersection, sparse LR LOO ~ chance vs residual LR 0.79).
- Why this is interesting, not just a method failure: it sharpens M4's
  "improvisation, not retrieval" claim — the model's class commitment
  is held *only* as a small linear component of a dense residual, not
  as one (or even a stable handful of) interpretable features.
- Caveats (only resid_post tested; only 2 layers from a 49-block
  model; only `medium` L0 width; possible Gemma Scope 2 SAE
  training-distribution mismatch).
- Tables: from `runs/m5_sae_turn_progressive_L31_balanced.json` and
  `runs/m5_sae_turn_progressive_L41_balanced.json`.

**(b) 27B scale capture.** Reuse the M3 positional capture pipeline at
27B. Per project_scale_question.md, the discriminating observable is
**end_ready LR LOO**: at 12B it sits at ~chance (model improvises);
if 27B clears chance, scale grants explicit pre-commitment, and the
follow-up patch test tells us if it's load-bearing or just legible.

Pipeline:
1. Run `capture_positional_residuals.py` against Gemma 3 27B with the
   six-question matched panel + 20-bank prompt. Walltime ≈ 4-6× the
   12B job (n=80 needs an A100 80GB, possibly multi-GPU). Aim for
   a 4-class balanced (20/class) collection at the same anchors
   the 12B v2 capture used.
2. M3-style positional probe: balanced LR LOO at end_ready, turn-1..4
   boundaries, turn-4 pre_answer.
3. If end_ready LR LOO > 0.40 (≥1.6× chance): run the M4 patching
   sweep at end_ready on a flip-vs-null target.
4. If end_ready stays at chance: improvisation is scale-robust; that's
   the headline result.

Skip A4b (pulls toward calibration, which the north star says we
don't headline) and `mlp_out`/`attn_out` SAEs at 12B (same residual
target, won't change the negative).

Pipeline:

1. **A5: Re-capture with 16 anchors.** `capture_positional_residuals.py`
   now adds four `pre_answer_qN` anchors (4 tokens past `end_user_qN`,
   the `\n` after `<start_of_turn>model` — M3's turn-4 LR-0.79 peak
   position). qsub against the existing 600-run default 12B collection
   on TSUBAME. Output: `runs/positional_residuals/12b_default_n80_v2/`
   (~6 MB/run × 600 = 3.6 GB). Cheap (~minutes walltime).
2. **A1–A2: Load Gemma Scope 2 12b-it SAEs and encode captured residuals.**
   New script `scripts/sae_feature_firings.py`. Cherry-pick start: L30 /
   L40 / L48 × {resid_post, mlp_out, attn_out} = 9 SAE loads, width 64k,
   medium L0 (30–60). Persist top-k=64 per (run, anchor, layer) as
   sparse `(feature_idx, activation)` pairs. Sanity check: SAE
   reconstruction MSE within Gemma Scope 2's published bounds on a
   50-run held-out subset.
3. **A3: Three feature-axis analyses** (class-discriminative,
   turn-progressive, yes/no-conditional) + **A4 reproducibility check**
   (sparse classifier vs. residual probe LR LOO at L30 `pre_answer_q4`)
   + **A4b calibration-attribute pickup** (do M3 calibration's six
   binary-attribute decoders correspond to single SAE features, or to
   distributed combinations?).
4. **B1 (gated on A producing ≥3 candidate features):** single-feature
   ablate-to-zero / set-to-target-class-mean steering at `pre_answer_q4`
   L30 and L40. Headline test: any flip rate r > 0/80 vs. the residual
   patching null. Reuses `patch_turn4.py:349` hooking pattern.

Predicted findings (preregistration-style) in the plan file. Headline
write-up: `docs/progress/M5-sae-features-12b-default-n80.md`.

**Side investigations still pending (non-blocking):**
- `attempt_588`/`206`/`038`/`049` baseline non-determinism (~4%).
- Phase 2c-iii centroids file (deferred indefinitely; only needed
  for residual-level steering which D-40 made lower priority).

**Side investigations still pending (non-blocking):**
- `attempt_588` (and 3 newly-flagged in D-40: `attempt_206`,
  `attempt_038`, `attempt_049`) baseline non-determinism across
  replays. ~4% noise floor.
- Phase 2c-iii centroids file deferred (won't be needed unless we
  pursue residual-level steering, which D-40 makes lower priority).
  Bug fix for `counts` ref already pushed (`bada4fc`) for next time.

**Methodological follow-up (deferred to whichever phase produces a
positive signal):** the first-step logit-diff metric is unreliable for
classes whose reveal doesn't begin with the animal-name token (dog at
phase 2a). When a positive signal emerges, upgrade the metric to scan
several generation steps and locate where the animal-name token first
becomes argmax-favored.

**Side investigation (non-blocking):** `attempt_588` is non-deterministic
at `do_sample=False`. On-disk reveal `horse`, replayed reveal `cow`. Same
prompt, same kwargs. Suspect KV-cache or attention-impl drift in
bfloat16. Worth ~30min of investigation before relying on horse trials
in subsequent patching work.

**Do not:**
- repeat 4-way narrowed self-chosen prompts (collapse is established)
- probe "State A/B" in the self-chosen condition — the concept does not
  apply there
- spend more cycles on Ready-state self-chosen decoding as the main branch
- sweep more positions on the 80-run dataset — turn-4 L26-48 is locked
- run a bare single-position layer sweep as the next M4 experiment.
  Heimersheim & Nanda 2024 recommend low-granularity (band) interventions
  *first*, then refine. A bare layer sweep nulls under both the
  redundancy and off-path interpretations and is non-diagnostic.

**Open threads on the backlog:**
- **Bigger-model ladder (D-05):** 4B's salmon attractor may be model-specific.
  At 12B+ the distribution could sharpen differently or flatten; re-run both
  `diagnose_persistence.py` and `diagnose_selfchosen_ready.py` when we scale.
- **Bank improvement:** disputed cells in the 20×30 table
  (`docs/progress/M3-binding-bank-audit.md`) touched `frog.has_four_legs`,
  which showed up in both self-chosen smokes' 53% correctness number. Not
  blocking the 20-candidate run, still worth a broader audit pass.

Scientific ground truth remains self-chosen. Keep publishing only self-chosen
results as the headline.

Full plan is at `~/.claude/plans/here-is-a-project-calm-hummingbird.md` and
`docs/PLAN.md` (scientific).

---

## Milestone tracker

- [x] **M0 — Repo bootstrap.** Skeleton, pyproject, docs seeded. See commit history.
- [x] **M1 — Data artifacts + feasible-set utility.** 20 animals × 30 questions × 0/1
      table; pairwise-distinguishability floor relaxed to 2 (D-14); 13 tests pass.
- [x] **M2 — Ready-state decoder smoke test** (Gemma 3 1B, local CPU). 160 calibration
      + 40 self-chosen runs; LR LOO 0.38 @ L15, attribute decoder 0.89 @ L17. See
      `docs/progress/M2-ready-smoke-test.md`.
- [ ] **M3 — TSUBAME + Gemma 3 4B, full calibration dataset (~2k).** With question turns.
- [ ] **M4 — Self-chosen full study + causal patching.**
- [ ] **M5 — Transcoder / SAE feature case studies.**
- [ ] **M6 — Blog post draft.**

---

## Open questions

- HF license approval for Gemma 3 1B is pending. Without it, M2 scales only to
  open models (Qwen 2.5 0.5B, Llama-3.2 if granted, etc.). The plan keeps
  Gemma 3 as the target (D-05); using a non-Gemma model for M2's scientific
  exit criterion would break downstream SAE/transcoder reuse at M5.
- Exact middle-layer target for Gemma 3 1B (26 layers → probably layer 13–18).
  Defer: M2 captures *all* layers, picked empirically from the sweep.
- Question-regime choice for M2 dialogues (ambiguity-first vs. disambiguation-first).
  Defer to M3; M2 is Ready-only, no questions yet.
- Whether to include a 21st "I don't know / none of the above" option to catch
  non-commitment. Deferred; current bet is the model will pick one under the prompt.

---

## Known risks

- **Probe transfer from calibration to self-chosen may fail.** The core scientific
  risk. Mitigation: build the feasible-set control (`S_t`) into the first cut so
  we can ask whether the signal exceeds what public dialogue trivially reveals.
- **Instruction-following failures on Gemma 3 1B.** The 1B may not reliably emit
  `Ready` alone, may leak the secret, may refuse self-choice. Track parse-success
  rate in M2 as an explicit metric.
- **Gemma Scope / circuit-tracer maturity.** SAE artifacts exist for Gemma 3 but
  circuit-tracer's NNsight backend is flagged experimental. Plan to do readouts
  and patching *without* circuit-tracer first; add it only for the case studies.
- **Introspection unreliability.** Treat the end-of-game reveal only as sanity
  check, never as ground truth for `ẑ_0`.

---

## Handoff protocol (reminder)

1. Read this file first.
2. Do the `Next concrete step`.
3. On milestone close, write `docs/progress/M<n>-<slug>.md`.
4. Append non-obvious choices to `docs/DECISIONS.md` with today's date.
5. Update this file's `Current milestone`, `Last agent`, `Last updated`, and
   `Next concrete step`. Commit with a milestone-prefixed message (`M1: ...`).
