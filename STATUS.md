# STATUS

> **First file any agent reads.** The `Next concrete step` is always actionable
> without reading anything else. Update the `Last updated` line on every session.

**Current milestone:** M3 — TSUBAME + Gemma 3 12B, transfer verdict in; self-chosen direct-fit next.
**Last agent:** Codex
**Last updated:** 2026-04-21 (12B calibration -> self-chosen Ready transfer tested two ways. New 4-way self-chosen run on TSUBAME, job `7230657`, collapses to `{cow,horse}` only; balanced kept transfer at `L6/L17/L27/L48` is NC `0.35/0.00/0.00/0.10` and LR `0.00/0.00/0.00/0.10`. Retrospective transfer on the earlier 20-bank 12B slice is `0.125-0.25`, i.e. chance. Follow-up direct-fit collection on the natural 20-bank prompt has already been submitted as job `7230807` (`runs/diag/selfchosen_ready_20bank_12b_directfit_20260421/`). See `docs/progress/M3-12b-selfchosen-transfer.md`.)

**North star:** *Calibration is infra; the scientific claim is self-chosen only.*
Do not headline calibration-only results.

---

## Next concrete step

Prior notes, newest first:
**`docs/progress/M3-12b-selfchosen-transfer.md`** (job `7230657`; plus retrospective 20-bank slice),
`docs/progress/M3-12b-pilot-readouts.md` (job `7226576`; readouts local),
`docs/progress/M3-selfchosen-20bank.md` (jobs `7223018`, `7226501`, `7226502`, `7226538`, `7226546`, `7226547`),
`docs/progress/M3-selfchosen-ready-T07.md` (job `7219788`),
`docs/progress/M3-selfchosen-ready-smoke.md` (job `7218660`),
`docs/progress/M3-h-persistence.md` (job `7218322`),
`docs/progress/M3-binding-bank-audit.md`,
`docs/progress/M3-3cond-binding-smoke.md` (job `7218265`),
`docs/progress/M3-4cond-binding-smoke.md`, `docs/progress/M3-4b-smoke-diagnostics.md`.

**Result so far (12B calibration is good; transfer still fails):**

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

Bottom line: 4B self-chosen improved, but is still not probe-ready. At **12B**,
`name_paraphrase` is good enough for calibration infrastructure, but
calibration-trained Ready probes still do **not** transfer to self-chosen.

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

**Next concrete step:** use the running 20-bank direct-fit collection to fit
probes directly at self-chosen Ready on 12B.

1. Wait for / pull `runs/diag/selfchosen_ready_20bank_12b_directfit_20260421/`
   from job `7230807`. Do **not** start another narrowed 4-way self-chosen run;
   that collapse is now established.
2. Use the partial-analysis output to identify the realized classes that reach
   quota under the 20-bank prompt. Treat those self-chosen Ready activations as
   the actual probe-training data.
3. Fit NC and LR directly on self-chosen reveal labels (LOO on the realized
   subset first; train/test only if the sample gets large enough).
4. Compare self-chosen-direct readouts against the failed calibration ->
   self-chosen transfer numbers in `docs/progress/M3-12b-selfchosen-transfer.md`.
   If direct-fit is strong, lock M4 around self-chosen-trained probes and causal
   patching. If direct-fit is weak too, the bottleneck is self-chosen geometry
   itself, not transfer.
5. Write `docs/progress/M3-12b-selfchosen-direct.md` and append the decision in
   `docs/DECISIONS.md`.

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
