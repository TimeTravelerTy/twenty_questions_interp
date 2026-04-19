# STATUS

> **First file any agent reads.** The `Next concrete step` is always actionable
> without reading anything else. Update the `Last updated` line on every session.

**Current milestone:** M3 — TSUBAME + Gemma 3 4B, full calibration dataset.
**Last agent:** Claude (Opus 4.7)
**Last updated:** 2026-04-19 (self-chosen Ready smoke done; choice distribution collapsed to 2/4 classes, Ready geometry ~6.7× weaker than State B; see `docs/progress/M3-selfchosen-ready-smoke.md`)

**North star:** *Calibration is infra; the scientific claim is self-chosen only.*
Do not publish calibration-only results as the headline.

---

## Next concrete step

Prior notes in order of recency:
**`docs/progress/M3-selfchosen-ready-smoke.md`** (job `7218660`),
`docs/progress/M3-h-persistence.md` (job `7218322`),
`docs/progress/M3-binding-bank-audit.md`,
`docs/progress/M3-3cond-binding-smoke.md` (job `7218265`),
`docs/progress/M3-4cond-binding-smoke.md`, `docs/progress/M3-4b-smoke-diagnostics.md`.

**Result so far (self-chosen Ready smoke, done):** 40 attempts on the same
4 candidates as the persistence diagnostic. Two findings:

1. **Choice distribution collapses.** salmon 33 / frog 7 / tiger 0 /
   eagle 0. Strong position bias (salmon- or frog-at-position-0 always
   chosen) + category bias (the two non-mammal-or-bird items dominate).
   Under greedy decoding, "self-chosen" is closer to biased-forced-choice.
2. **Self-chosen Ready is ~6.7× weaker than State B** (2-class balanced
   analysis, salmon vs frog, n=7 each). Post-13 within-between contrast:
   self-chosen +7.85e-05, State B +5.26e-04, State A +1.31e-02. Identity
   is NC-decodable at 100% only from L29 onward — not at L21 where A and
   B both peak.

Ordering: **A > B > self-chosen Ready** (by cosine contrast). The latent
identity is still recoverable, but only at late layers with much weaker
amplitude than either persistence state. See D-23.

**Next concrete step:** patched self-chosen replication that (a) breaks the
greedy choice collapse and (b) falls back to restricted-class analysis
when the full 4-class quota isn't met.

1. Patch `scripts/diagnose_selfchosen_ready.py` to:
   - Emit the ready-analysis block whenever ≥2 classes have ≥`n-per-candidate`
     kept runs (don't abort).
   - Accept a `--temperature` flag (default 0.0 = greedy; non-zero enables
     `do_sample=True` with that temperature) so we can trade determinism
     for distribution variety.
2. On TSUBAME, run a temperature-sampled replication:
   `python scripts/diagnose_selfchosen_ready.py --model google/gemma-3-4b-it --device auto --dtype bfloat16 --n-per-candidate 4 --max-attempts 120 --temperature 0.7 --out-dir runs/diag/selfchosen_ready_T07 --persistence-results runs/diag/persistence_smoke_20260419/results.json`
3. If that still collapses, widen the candidate list (e.g. the full 20)
   and re-run; that's a scientifically-cleaner self-chosen setup for
   M4 anyway.
4. Once 4+ classes are realized, read `ready_analysis.comparison_to_persistence`
   and update `docs/progress/M3-selfchosen-ready-smoke.md` with the
   stronger numbers.

**Open threads kept deliberately on the backlog, not closed:**
- **Bigger-model ladder (D-05):** competence at 12B+ will likely widen
  the self-chosen distribution and clean the geometry; re-run both
  `diagnose_persistence.py` and `diagnose_selfchosen_ready.py` when we
  scale.
- **Bank improvement:** the 20×30 answer table still has documented
  disputed cells (`docs/progress/M3-binding-bank-audit.md`); a broader
  audit is due but not blocking the current research branch.
- **Full 20-candidate self-chosen:** whether the position / category
  biases above survive when the prompt contains all 20 is untested.

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
