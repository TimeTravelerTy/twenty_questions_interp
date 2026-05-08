# STATUS

> **First file any agent reads.** The `Next concrete step` is always actionable
> without reading anything else. Update the `Last updated` line on every session.

**Current milestone:** **M5 Phase A first cut in (L31, pre_answer_q4, resid_post 65k medium-L0)** — A1 sanity passes (FVU 1.68%) and a striking sparsity pattern is visible (88 unique features fire across all 600 runs; per-run L0 ~60), with the dominant class-discriminative axis being **horse-asymmetric**, matching the M4 lens generic-prior result. **A4 AxBench check fails** at this single (anchor, layer): sparse classifier balanced 20/class LR LOO = **0.3125** vs M3 residual probe LR LOO **0.79** at the same position (gap = 0.48, far outside the ±0.05 reproducibility threshold). The residual's 4-way class direction is *not* aligned with the SAE's sparse-feature basis at L31. Plan: `~/.claude/plans/check-the-latest-status-bright-horizon.md`. Scope: Phase A + B1 only; B2/B3 deferred to M5b.
**Last agent:** Claude
**Last updated:** 2026-05-09 (M5 Phase A first cut shipped. Captures: `runs/positional_residuals/12b_default_n80_v2/` (600 × 16 anchors). Firings: `runs/m5_sae_firings_12b_default_resid_post_L31_pre_answer_q4.pt`. Analysis: `runs/m5_sae_analysis_..._balanced.json` (LR LOO 0.3125 / 600-pool 0.54). Loaded Gemma Scope 2 12b-it directly from HF (sae-lens 6.30.1's pretrained dir does not yet wire it in); JumpReLU SAE math implemented in 5 lines of torch via `JumpReLUSAEModule` in `scripts/sae_feature_firings.py`. Capture script switched to legacy torch.save format on Lustre (zip-archive corruption was actually masquerading EDQUOT). HF cache moved to `/gs/bs/tga-sip_arase/tyrone/.cache/huggingface` per user direction (40 PB bulk fs vs constrained `/gs/fs`).)

**North star:** *Calibration is infra; the scientific claim is self-chosen only.*
Do not headline calibration-only results.

---

## Next concrete step

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

**Next concrete step — M5 Phase A second cut at L41, plus turn-progressive
encoding.** L31 first cut shows the AxBench warning: SAE features
recapitulate a horse-asymmetric pattern (matching M4 lens) but **fail
A4 reproducibility** (sparse LR LOO 0.31 vs residual LR LOO 0.79 at the
balanced 20/class condition). Two follow-ups before deciding whether
Phase B1 is worth running:

1. **L41 encoding + analysis** (Gemma Scope 2's deepest fixed-depth
   layer at 85% depth — closer to the L40-48 "monotonically growing"
   band M4 found, possibly less generic-prior contaminated than L31).
   Same script, swap `--hf-subfolder resid_post/layer_41_width_65k_l0_medium`
   and `--capture-index 42 --block-id 41`.
2. **Turn-progressive encoding** at L31 across all four pre_answer_qN
   anchors. The 600-run capture already has them; encode once per
   anchor and aggregate. This answers Phase A3.2 directly: do features
   form monotonically across q1 → q4? Independent of A4's classifier
   result.

If L41 also fails A4 ≥ 0.05 of residual probe, Phase B1 is unlikely to
land a positive — recommend pivoting to A4b (calibration-attribute
pickup, via M3 calibration runs) as the cleanest decomposition story
that doesn't depend on the imbalanced 4-way class direction being
sparse-feature-aligned.

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
