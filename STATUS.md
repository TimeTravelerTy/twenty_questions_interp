# STATUS

> **First file any agent reads.** The `Next concrete step` is always actionable
> without reading anything else. Update the `Last updated` line on every session.

**Current milestone:** M3 — TSUBAME + Gemma 3 4B, full calibration dataset.
**Last agent:** Claude (Opus 4.7)
**Last updated:** 2026-04-19 (3-condition 4B binding follow-up run; all three missed the 95% gate; pivoting from prompt sweeps to representation diagnostics)

**North star:** *Calibration is infra; the scientific claim is self-chosen only.*
Do not publish calibration-only results as the headline.

---

## Next concrete step

Prior notes: `docs/progress/M3-4b-smoke-diagnostics.md`,
`docs/progress/M3-4cond-binding-smoke.md`, and the new
**`docs/progress/M3-3cond-binding-smoke.md`** (job `7218265`).

**Result so far:** none of the three non-index prompts tried in the
follow-up cleared the ≥95% primary gate.
- `name_paraphrase` 84.4%, `name_strict` 84.4%, `verbalized_index` 71.9%.
- `verbalized_index` verbalized the correct name in 8/8 runs but still
  drifted to candidate-list priors by Ready; `eagle.is_mammal=Yes` and
  `eagle.can_swim=Yes` happened in every single eagle run across all
  conditions.
- Plain within-cos remains near 1.0; within-vs-between contrast is ~10x
  larger than on the 4cond run but does not track correctness.

Decision per D-19: **do not reverse D-06, do not scale**. Pivoting from
prompt sweeps to representation diagnostics. Hypothesis tightened from
H-binding to **H-persistence**: at 4B the instantiated-entity representation
does not reliably persist across a chat-turn boundary even when name
retrieval succeeds.

**Next concrete step:** two cheap probes, in this order.

1. **Bank audit under D-14** against the systematic failures. Candidates
   for review: `tiger/can_swim` (known), `eagle/can_swim` (all 6 eagle runs
   call this true), `frog/has_four_legs`, `frog/lives_primarily_in_water`.
   Add or remove cells per the rule in D-14 and re-score the existing
   `runs/diag/binding_smoke_5_20260419/` JSON offline. If the revised bank
   lifts name_paraphrase above 95%, reopen D-06 with the updated numbers
   before doing any new remote work.
2. **Mechanistic test of H-persistence.** Re-run `verbalized_index` once
   on the same 4 candidates × 2 seeds, but capture two Ready states: one at
   the end of turn 1 (model has just verbalized the name), one at the end
   of turn 2 (normal lock-in). Fit a minimum-viable attribute probe on the
   turn-1 states and score it on the turn-2 states. If turn-1 decodes the
   secret attribute and turn-2 does not, H-persistence is directly measured;
   that becomes the headline mechanistic result for M3 and the reason to
   keep self-chosen as the only calibration regime.
3. (Only after 1–2.) Consider a small self-chosen smoke at 4B on the same
   primary question set, to check whether self-chosen answer behavior
   looks closer to the name regime (~85%) or to verbalized_index (~72%).
   This speaks directly to the blog claim.

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
