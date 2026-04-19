# STATUS

> **First file any agent reads.** The `Next concrete step` is always actionable
> without reading anything else. Update the `Last updated` line on every session.

**Current milestone:** M3 — TSUBAME + Gemma 3 4B, full calibration dataset.
**Last agent:** Codex (GPT-5)
**Last updated:** 2026-04-19 (4-condition 4B smoke run on TSUBAME; index failed, name improved but missed gate)

**North star:** *Calibration is infra; the scientific claim is self-chosen only.*
Do not publish calibration-only results as the headline.

---

## Next concrete step

The earlier M3 diagnostics note is at
**`docs/progress/M3-4b-smoke-diagnostics.md`**. The completed 4-condition
follow-up is at **`docs/progress/M3-4cond-binding-smoke.md`**.

**Result so far:** the 4-condition 4B run confirms that index-based
calibration remains unusable (`50%`, `40%`), while name-based prompts are much
better (`90%`, `90%`) but still miss the `>=95%` gate required to reverse
D-06. Plain within-secret cosine is too saturated to use alone; the useful
signal is the within-vs-between Ready-state contrast.

**Next concrete step:** run one final small 4B binding follow-up before
choosing the replacement for D-06.

1. Keep the same **4 candidates** and **2 seeds** as the completed smoke:
   `tiger, eagle, frog, salmon`.
2. Use a **non-ambiguous primary score set**:
   `is_mammal, is_bird, can_fly, has_feathers`.
   Report `can_swim` separately as a secondary sanity check; do not let the
   known `tiger/can_swim` edge case dominate the decision.
3. Compare three non-index options:
   - current best name-based prompt,
   - one slightly stronger name-based prompt that explicitly says to answer
     only about the named animal, not the average animal in the list,
   - the earlier verbalized-index control as a reserve option.
4. Score each option on:
   - **answer correctness** on the primary score set (must hit `>=95%`);
   - **Ready-state separation** using within-vs-between cosine contrast or a
     tiny NC/LR check, not plain within-secret cosine alone.
5. If one option clears the gate, add the D-06 reversal/replacement explicitly
   in `docs/DECISIONS.md` and then scale calibration to ~100 runs per
   candidate. If not, stop and investigate prompt semantics before scaling.
6. Only after the calibration regime is fixed, resume the original M3 scale-up:
   re-measure the NC-vs-LR gap at 4B and then decide tiger-bias mitigation for
   self-chosen.

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
