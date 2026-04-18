# STATUS

> **First file any agent reads.** The `Next concrete step` is always actionable
> without reading anything else. Update the `Last updated` line on every session.

**Current milestone:** M3 — TSUBAME + Gemma 3 4B, full calibration dataset.
**Last agent:** Claude (Opus 4.7)
**Last updated:** 2026-04-19 (after M2 close)

**North star:** *Calibration is infra; the scientific claim is self-chosen only.*
Do not publish calibration-only results as the headline.

---

## Next concrete step

M2 closed on balance (see `docs/progress/M2-ready-smoke-test.md`). Infra works;
attribute decoders pass; NC LOO missed (0.12 vs 0.20 threshold) but LR LOO
hit 0.38 — diagnostic, pushes toward the "answer-sufficient attribute bundle"
framing over "single concept direction." Zero calibration→self-chosen transfer
confirms calibration-as-infra stance.

**M3 — scale to Gemma 3 4B on TSUBAME.** Use the `tsubame-ssh` skill for remote
runs on an A100. Targets:

1. Bump model to `google/gemma-3-4b-it`. Verify `capture_ready_state` works
   unchanged (`output_hidden_states=True` is model-agnostic).
2. Full calibration dataset: ~100 runs per candidate (2000 total). Keep per-run
   permutation + seed logging.
3. **Introduce question turns.** M2 was Ready-only; the Ready-state
   commitment must be *forced*, not optional. See PLAN.md §6 for the
   dialogue loop. This changes what's captured at each turn from
   `h^{(ℓ)}_{r,0}` (pre-Ready) to `h^{(ℓ)}_{r,t}` (pre-answer at turn t).
4. Re-measure the NC-vs-LR gap at 4B. If it persists, the attribute-bundle
   hypothesis is strengthened; if NC catches up, single-direction back on
   the table.
5. Decide tiger-bias mitigation (see M2 progress note finding 4):
   (a) trust 4B to be more diverse, (b) add a "be diverse" nudge, or
   (c) oversample under-represented candidates in self-chosen.

Ground truth for scientific claims remains self-chosen. Keep publishing only
self-chosen results as the headline.

Full plan is at `~/.claude/plans/here-is-a-project-calm-hummingbird.md` and
`docs/PLAN.md` (scientific).

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
