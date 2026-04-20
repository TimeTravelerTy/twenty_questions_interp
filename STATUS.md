# STATUS

> **First file any agent reads.** The `Next concrete step` is always actionable
> without reading anything else. Update the `Last updated` line on every session.

**Current milestone:** M3 — TSUBAME + Gemma 3 4B, full calibration dataset.
**Last agent:** Codex
**Last updated:** 2026-04-20 (reviewed Claude's latest M3 handoff; repo remains aligned with the self-chosen-first plan, and `scripts/diagnose_selfchosen_ready.py` now supports `--candidates all` / `--question-ids all` directly so the 20-candidate self-chosen run no longer depends on a brittle explicit 20-id CLI string)

**North star:** *Calibration is infra; the scientific claim is self-chosen only.*
Do not publish calibration-only results as the headline.

---

## Next concrete step

Prior notes in order of recency:
**`docs/progress/M3-selfchosen-ready-T07.md`** (job `7219788`),
`docs/progress/M3-selfchosen-ready-smoke.md` (job `7218660`),
`docs/progress/M3-h-persistence.md` (job `7218322`),
`docs/progress/M3-binding-bank-audit.md`,
`docs/progress/M3-3cond-binding-smoke.md` (job `7218265`),
`docs/progress/M3-4cond-binding-smoke.md`, `docs/progress/M3-4b-smoke-diagnostics.md`.

**Result so far (T=0.7 self-chosen replication, done, see D-24):**
120 attempts on the same 4 candidates with T=0.7 sampling across Ready,
question, and reveal generations.

1. **Choice collapse persists.** salmon 96 / frog 24 / tiger 0 / eagle 0.
   Temperature was not the cause; the 4B model's self-chosen prior on
   this panel is ~{salmon: dominant, frog: minor, tiger/eagle: ~0}.
   Salmon-at-position-0 → salmon 39/39; eagle@0 → salmon 31/31;
   tiger@0 → salmon 20 / frog 4. Only frog@0 reliably flips (20/26).
2. **Geometry tightened slightly but did not close the gap.** Balanced
   2-class (salmon vs frog, n=4): post-13 contrast +1.14e-04 (vs T=0's
   +7.85e-05, 1.45×); best NC layer shifted L29 → **L24**; still ~4.5×
   weaker than State B at matched sample size. Comparison vote vs
   persistence: **mixed** (nc tie at L21, contrast still state_b).
3. **Answer correctness dropped to 53%** under sampling — sampling noise
   eats the yes/no fluency that later calibration-based probes assume.

Conclusion: the 4-candidate self-chosen smoke is closed as a 4-way test
at 4B. The fix is not more sampling — it's a less-biased prompt. Ordering
still holds: **A > B > self-chosen Ready**.

**Next concrete step:** full 20-candidate self-chosen smoke, to test the
hypothesis that diluting the salmon attractor with 16 additional animals
(a) broadens the realized-class distribution enough to run a 4+ way test
and (b) is in any case the scientifically cleaner self-chosen setup
required for M4.

1. Decide the question set. Either (a) keep the 4 primary attribute
   questions from persistence/self-chosen-smoke (`is_mammal, is_bird,
   lives_primarily_in_water, has_four_legs`) for direct comparability,
   or (b) use the full bank. Default to (a) for comparability — the
   representational claim is about Ready-state, not question dynamics.
2. Invoke `scripts/diagnose_selfchosen_ready.py` with the full 20-bank
   subset via `--candidates all`. Leave `--question-ids` at the 4 primary
   questions by default for comparability; use `--question-ids all` only
   if the run's purpose changes from Ready-state comparability to broader
   dialogue characterization.
3. On TSUBAME, submit T=0.0 first (to characterize the greedy
   distribution on the 20-candidate prompt) with generous `--max-attempts`
   (say 200) and `--n-per-candidate 2` as a quota floor. Expected runtime
   ≤10 min on H100 given the ~45s/run × 200 bound.
4. If T=0.0 on 20 still collapses to one or two classes, add T=0.7 as a
   follow-up; if that also collapses, scale the model (D-05).
5. Pull results, compute the 4+ way `comparison_to_persistence` (will
   require persistence results to be restricted to the same class set
   for a fair comparison — probably do the comparison in an ad-hoc
   analysis notebook first, like we did for the 2-class T=0.7 run).
6. Write `docs/progress/M3-selfchosen-20bank.md` and update STATUS / D-24.

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
