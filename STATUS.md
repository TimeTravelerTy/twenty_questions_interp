# STATUS

> **First file any agent reads.** The `Next concrete step` is always actionable
> without reading anything else. Update the `Last updated` line on every session.

**Current milestone:** M3 — TSUBAME + Gemma 3 4B, full calibration dataset.
**Last agent:** Codex (GPT-5)
**Last updated:** 2026-04-19 (4-condition binding harness added locally; TSUBAME run pending)

**North star:** *Calibration is infra; the scientific claim is self-chosen only.*
Do not publish calibration-only results as the headline.

---

## Next concrete step

M2 closed on balance (see `docs/progress/M2-ready-smoke-test.md`). The 4B
remote smoke is done and written up as a diagnostics note:
**`docs/progress/M3-4b-smoke-diagnostics.md`**. Read that before the step
below; it frames the calibration-binding question as a research finding
(index-based commitment does not reliably instantiate a specific-entity
Ready state at 4B), not just an infra hiccup.

**Next concrete step:** sync and run `scripts/diagnose_index_binding.py` on
TSUBAME, then use the result to decide whether to reverse D-06 before scaling.

1. Build a small remote smoke on Gemma 3 4B covering:
   - **4 candidates** spanning categories: `tiger, eagle, frog, salmon`.
     (The eagle `is_mammal` failure may be bird-in-mammal-heavy-list
     specific; we need to see if it's category-general.)
   - **5 questions**: `is_mammal, is_bird, can_fly, can_swim, has_feathers`
     (or similar). More signal per run.
   - **2 seeds** per (candidate × condition), so 16 runs per condition.
   - **4 conditions**:
     (a) current D-06 index-based prompt,
     (b) index + explicit position reminder per turn,
     (c) name-based ("Your secret animal is X"),
     (d) name-based paraphrase ("You have chosen X as your secret").
     Persist per-turn activations for all four so we can cheaply fit
     NC/LR at Ready between variants.
2. Score each condition on two gates, not just parse success:
   - **answer correctness** vs. the bank (must be ≥ 95% on (c)/(d) or we
     stop and investigate);
   - **within-secret Ready-state cosine** (does the same secret across
     seeds/positions collapse to a tight cluster? index vs name will be
     the interesting comparison).
3. Write the result into a *new* DECISIONS.md entry. If (c)/(d) pass and
   (a)/(b) don't, reverse D-06 explicitly with a new numbered entry and
   note the token-confound mitigation (paraphrase the binding; randomise
   positions; later, check the last token before `Ready` isn't trivially
   the animal's literal first BPE piece).
4. Only after that passes, scale calibration to ~100 runs per candidate
   (2000 total), keeping per-run permutation + seed logging.
5. Re-measure the NC-vs-LR gap at 4B with more samples per class. If it
   persists with ~100 runs/class (where NC should be well-estimated), the
   attribute-bundle hypothesis gains weight; if NC catches up, single-
   direction is back on the table. Also measure LR/attribute transfer to
   self-chosen, not just NC.
6. Decide tiger-bias mitigation (see M2 progress note finding 4):
   (a) trust 4B to be more diverse, (b) add a "be diverse" nudge, or
   (c) oversample under-represented candidates in self-chosen.

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
