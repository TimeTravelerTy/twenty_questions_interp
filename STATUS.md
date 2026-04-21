# STATUS

> **First file any agent reads.** The `Next concrete step` is always actionable
> without reading anything else. Update the `Last updated` line on every session.

**Current milestone:** M3 — TSUBAME + Gemma 3 12B, pilot calibration + first readouts.
**Last agent:** Claude
**Last updated:** 2026-04-21 (12B `name_paraphrase` pilot collected on TSUBAME in job `7226576`: 100/100 runs, 25×{elephant,cow,dog,horse}. First Ready-state readouts trained locally via `scripts/decode_ready.py` on the six-question discriminative panel: LR LOO saturates at 1.00 by L6, NC LOO climbs 0.25→1.00 by L27, attribute decoders 1.00 from L7. See `docs/progress/M3-12b-pilot-readouts.md`. Also fixed a latent NC class-set bug that only surfaces on subset pilots; committed with the report.)

**North star:** *Calibration is infra; the scientific claim is self-chosen only.*
Do not publish calibration-only results as the headline.

---

## Next concrete step

Prior notes in order of recency:
**`docs/progress/M3-12b-pilot-readouts.md`** (job `7226576`; readouts local),
`docs/progress/M3-selfchosen-20bank.md` (jobs `7223018`, `7226501`, `7226502`, `7226538`, `7226546`, `7226547`),
`docs/progress/M3-selfchosen-ready-T07.md` (job `7219788`),
`docs/progress/M3-selfchosen-ready-smoke.md` (job `7218660`),
`docs/progress/M3-h-persistence.md` (job `7218322`),
`docs/progress/M3-binding-bank-audit.md`,
`docs/progress/M3-3cond-binding-smoke.md` (job `7218265`),
`docs/progress/M3-4cond-binding-smoke.md`, `docs/progress/M3-4b-smoke-diagnostics.md`.

**Result so far (12B calibration readouts green; next is self-chosen transfer):**

1. **The 20-bank prompt is a real improvement over the 4-candidate panel.**
   Greedy 4B no longer collapses to just salmon/frog. Over 200 attempts the
   reveal histogram is dolphin 104 / penguin 85 / shark 5 / crocodile 2 /
   horse 2 / cow 1 / salmon 1, so 7 classes appear and 5 reach quota `n=2`.
2. **But Ready-state geometry is still very weak at 4B.** On the balanced
   realized 5-way subset (`dolphin,horse,penguin,crocodile,shark`),
   self-chosen best post-L13 NC is only **10.0%** (chance 20%) and
   post-L13 within-between contrast is **+4.70e-06**.
3. **Matched persistence on the same 5 classes remains strong.** State A and
   State B both hit **100% NC at L21**, cross A→B is **90%**, and State B
   contrast is **+4.87e-04**. Self-chosen is therefore still about **104×
   weaker than matched persistence State B** on contrast.
4. **Matched calibration still fails the gate.** The best single schema,
   `name_paraphrase`, scores only **34/40 = 85.0%** primary correctness on
   the same realized 5-way subset. That is not good enough to standardize.
5. **12B improves raw self-chosen behavior but exposes a question-panel bug.**
   Greedy 12B over 100 attempts realizes 4 classes with quota
   (`elephant,cow,dog,horse`) and reaches **37.5% NC** at L15 (chance 25%),
   but the legacy 4-question panel is non-diagnostic on that subset:
   every realized class has the same answer fingerprint `1,0,0,1`. So the
   apparent 100% correctness there is not a meaningful calibration signal.
6. **A discriminative 12B matched control panel fixes that and clears the gate.**
   On the six-question panel
   `is_carnivore,is_larger_than_human,is_domesticated,lives_in_africa,produces_dairy_milk,is_ridden_by_humans`,
   matched persistence stays strong (`7226546`: 45/48 = 93.8%, NC-A 100% at L30)
   and matched calibration with **`name_paraphrase`** finally passes the
   semantic gate (`7226547`: **47/48 = 97.9%**).

Conclusion: 4B self-chosen is better with the 20-bank prompt, but 4B is still
not in a probe-ready regime. At **12B**, calibration is no longer the blocker:
`name_paraphrase` is good enough to use as probe-training infrastructure.

**Pilot done (2026-04-21):** 100-run 12B `name_paraphrase` calibration collected
on `{elephant,cow,dog,horse}` with the six-question discriminative panel
(job `7226576`). Readouts trained locally at Ready across all 49 layers:

- **LR LOO saturates at 1.00 by L6**, stays saturated to L48.
- **NC LOO climbs 0.25→0.66 @ L7→0.93 @ L16→1.00 from L27.**
- Six binary attribute decoders all 1.00 from L7 (trivial at 4 classes × 25).

The LR ≫ NC gap at L6–L26 is the non-trivial structural signal: candidate
identity is linearly available in the residual stream from ~1/4 depth,
but the per-class clusters are not spherical until much deeper. Full
analysis: `docs/progress/M3-12b-pilot-readouts.md`; per-layer table:
`docs/progress/M3-12b-pilot-readouts-detail.md`.

**Next concrete step:** the scientific test this whole arc has been
building toward — probe transfer from calibration Ready to self-chosen
Ready at 12B on the same 4-way subset.

1. Collect a 12B self-chosen Ready run on `{elephant,cow,dog,horse}` (or
   the realized 12B subset if it broadens on re-run) using the same
   six-question panel and `name_paraphrase`-compatible self-chosen prompt.
   Use `scripts/diagnose_selfchosen_ready.py` with an explicit candidate
   list and `--question-ids is_carnivore,is_larger_than_human,
   is_domesticated,lives_in_africa,produces_dairy_milk,is_ridden_by_humans`.
   Target ~10 runs per realized class.
2. Fit NC and LR at Ready on the 100-run calibration pilot (layers of
   interest: L6 (LR's earliest perfect layer), L17 (NC's turn-on layer),
   L27 (NC saturation), L48 (last)).
3. Evaluate those probes on the self-chosen runs. Headline metric: probe
   agreement with the self-chosen reveal label. This is the direct test
   of whether calibration-position probes transfer to the self-chosen
   position at 12B — i.e. whether H-rotation (D-21) at 4B was a
   small-model artifact or persists at 12B.
4. If transfer is weak (say <70% at every layer), the probe-training
   location conclusion from D-23 reinforces: we would need to fit probes
   at self-chosen directly, which requires enough self-chosen runs per
   class, which loops back to collecting a full 12B self-chosen dataset.
5. Write `docs/progress/M3-12b-selfchosen-transfer.md` and update STATUS
   + DECISIONS.md (expected: adds a D-25 entry summarizing 12B transfer).

**Open threads kept deliberately on the backlog, not closed:**
- **Bigger-model ladder (D-05):** 4B's salmon attractor may be model-specific.
  At 12B+ the distribution could sharpen differently or flatten — re-run
  both `diagnose_persistence.py` and `diagnose_selfchosen_ready.py` when
  we scale.
- **Bank improvement:** disputed cells in the 20×30 table
  (`docs/progress/M3-binding-bank-audit.md`) touched `frog.has_four_legs`,
  which showed up in both self-chosen smokes' 53% correctness number.
  Not blocking the 20-candidate run, still worth a broader audit pass.

Ground truth for scientific claims remains self-chosen. Keep publishing only
self-chosen results as the headline.

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
  risk. Mitigation: build the feasible-set control (`S_t`) into the first cut so we
  can always ask "is the signal above what the public dialogue trivially reveals?"
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
