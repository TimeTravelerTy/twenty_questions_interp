# STATUS

> **First file any agent reads.** The `Next concrete step` is always actionable
> without reading anything else. Update the `Last updated` line on every session.

**Current milestone:** M3 — TSUBAME + Gemma 3 4B, full calibration dataset.
**Last agent:** Claude (Opus 4.7)
**Last updated:** 2026-04-19 (H-persistence test done; replaced by H-rotation: entity is still NC-decodable at turn 2, but geometry rotates and amplitude collapses ~25x)

**North star:** *Calibration is infra; the scientific claim is self-chosen only.*
Do not publish calibration-only results as the headline.

---

## Next concrete step

Prior notes in order of recency: **`docs/progress/M3-h-persistence.md`**
(job `7218322`), `docs/progress/M3-binding-bank-audit.md`,
`docs/progress/M3-3cond-binding-smoke.md` (job `7218265`),
`docs/progress/M3-4cond-binding-smoke.md`, `docs/progress/M3-4b-smoke-diagnostics.md`.

**Result so far (H-persistence diagnostic, done):** on `verbalized_index`,
all 8 runs verbalize the correct name. At middle layers the entity is
**100% NC-decodable at both state A (post-verbalization) and state B
(pre-Ready)** — but:

- **Cross A→B at layer 21 = 37.5%** (at-chance for most middle layers).
  Centroids fit at A do not classify B.
- **Within-vs-between cosine contrast at B is ~25× weaker than at A**
  (5.17e-04 vs 1.28e-02, post-L13). The entity signal is still present but
  nearly drowned in Ready-prep components.
- Primary correctness still 71.9% with the exact same systematic errors
  (eagle → mammal=Yes in 2/2, frog → mammal=Yes in 2/2, etc.).

The original H-persistence hypothesis ("entity does not persist") is
**refuted in its strong form**. Replaced by **H-rotation**: entity identity
persists across the chat-turn boundary, but the geometry rotates and the
amplitude collapses. Attribute readouts trained at A would not transfer to
B — which is exactly what explains the answer drift.

**Next concrete step:** small self-chosen 4B smoke on the same primary
question set to decide which regime self-chosen resembles.

1. Use the existing `scripts/run_selfchosen_smoke.py` (or a thin
   persistence-style variant of it) on `google/gemma-3-4b-it`.
2. Use the same 4 candidates and the primary question set
   (`is_mammal, is_bird, lives_primarily_in_water, has_four_legs`).
3. Capture the Ready-state activations per run. Compute within-vs-between
   contrast and NC LOO across the 4 "implicit" secrets the model chose.
4. Compare: does self-chosen Ready look like State A (clean,
   high-contrast, probe-ready) or State B (decodable identity but
   collapsed amplitude)?

If self-chosen looks like A: the readout pipeline is unblocked; pick a
layer, scale the calibration, move to M3 full. If self-chosen looks like
B: the blog story has to center on "decodable-but-rotated latent," and
we need to think harder about where in the dialogue to anchor probes —
possibly before we bother with the full calibration sweep.

**Open threads kept deliberately on the backlog, not closed:**
- **Bigger-model ladder (D-05):** competence at 12B+ will likely change
  the H-rotation picture; re-run `diagnose_persistence.py` when we scale.
- **Bank improvement:** the 20×30 answer table still has documented
  disputed cells (`docs/progress/M3-binding-bank-audit.md`); a broader
  audit is due but not blocking the current research branch.

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
