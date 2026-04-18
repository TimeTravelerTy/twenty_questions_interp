# STATUS

> **First file any agent reads.** The `Next concrete step` is always actionable
> without reading anything else. Update the `Last updated` line on every session.

**Current milestone:** M2 — Ready-state decoder smoke test on Gemma 3 1B (local CPU).
**Last agent:** Claude (Opus 4.7)
**Last updated:** 2026-04-18 (after M1 close)

**North star:** *Calibration is infra; the scientific claim is self-chosen only.*
Do not publish calibration-only results as the headline.

---

## Next concrete step

Write `src/twenty_q/prompts.py` (calibration + self-chosen templates;
calibration is index-based per D-06), `src/twenty_q/hooks.py` (NNsight
residual-stream capture at the token position immediately before `Ready`,
**all layers** per D-08), and `src/twenty_q/dialogue.py` (drives a single
Ready-capture run end-to-end). Target Gemma 3 1B (`google/gemma-3-1b-it`)
on local CPU. Gemma 3 is gated on HF — user needs `HF_TOKEN` in `.env`.

After scaffolding: `scripts/run_calibration.py --n-per-candidate 8`
(160 runs) and `scripts/run_selfchosen_smoke.py --n 40` produce manifests.
Then `scripts/decode_ready.py` runs the layer sweep + nearest-centroid /
logistic-regression / attribute decoders and writes the M2 progress note.

Full plan is at `~/.claude/plans/here-is-a-project-calm-hummingbird.md` and
`docs/PLAN.md` (scientific).

---

## Milestone tracker

- [x] **M0 — Repo bootstrap.** Skeleton, pyproject, docs seeded. See commit history.
- [x] **M1 — Data artifacts + feasible-set utility.** 20 animals × 30 questions × 0/1
      table; pairwise-distinguishability floor relaxed to 2 (D-14); 13 tests pass.
- [ ] **M2 — Ready-state decoder smoke test** (Gemma 3 1B, local CPU). 160 calibration
      runs + 40 self-chosen smoke.
- [ ] **M3 — TSUBAME + Gemma 3 4B, full calibration dataset (~2–4k).**
- [ ] **M4 — Self-chosen full study + causal patching.**
- [ ] **M5 — Transcoder / SAE feature case studies.**
- [ ] **M6 — Blog post draft.**

---

## Open questions

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
