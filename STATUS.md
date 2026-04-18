# STATUS

> **First file any agent reads.** The `Next concrete step` is always actionable
> without reading anything else. Update the `Last updated` line on every session.

**Current milestone:** M1 — data artifacts (animals, questions, A(c,q)) + feasible-set utility.
**Last agent:** Claude (Opus 4.7)
**Last updated:** 2026-04-18

**North star:** *Calibration is infra; the scientific claim is self-chosen only.*
Do not publish calibration-only results as the headline.

---

## Next concrete step

**Prereq:** install `uv` (`brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`), then `uv sync`. Not done in M0 because `uv` wasn't on the machine when bootstrap ran.

Then: draft `data/animals.yaml` (20 candidates) with attribute diversity coverage
(habitat / diet / body plan / size / domesticity / geography). Every pair must
differ on ≥3 attributes. Schema lives at `src/twenty_q/banks.py` (to be written
in parallel — see M1.1d in the plan).

After that: `data/questions.yaml` (~28 binary predicates), then `data/answers.csv`,
then `scripts/validate_answers.py`, then `src/twenty_q/{banks,permutations,manifest}.py`
with their tests.

Full plan is at `~/.claude/plans/here-is-a-project-calm-hummingbird.md` and
`docs/PLAN.md` (scientific).

---

## Milestone tracker

- [x] **M0 — Repo bootstrap.** Skeleton, pyproject, docs seeded. See commit history.
- [ ] **M1 — Data artifacts + feasible-set utility.** In progress.
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
