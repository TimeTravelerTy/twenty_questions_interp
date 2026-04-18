# Design decisions log

> Append-only, dated. Record any non-obvious choice and its reasoning so a future
> agent (or future self) can tell whether it still holds. If a decision is
> *reversed*, add a new entry with the reversal — do not edit the old one.

## 2026-04-18 — Project bootstrap decisions

### D-01: Calibration is infrastructure, not the scientific result
The blog claim lives in the self-chosen condition. Calibration (secret supplied by
index) exists only to train readouts and validate plumbing. **Never** present a
calibration-only result as the headline. Stated in STATUS.md and enforced by the
feasible-set control `S_t` from M1 onward.

### D-02: Dev env — local Mac CPU with Gemma 3 1B, then TSUBAME
Iteration speed matters more than throughput at M0–M2. Gemma 3 1B is ~2 GB and runs
on CPU. Porting friction to TSUBAME later is acceptable. Locked in until M3.

### D-03: Data artifacts before any model code
Reduces the risk of throwing away runs because a bank changed. Every run manifest
pins the banks it was built against; we want that pin stable before we start
collecting activations.

### D-04: Handoff via STATUS.md + per-milestone progress notes + DECISIONS.md
Durable file-based context that either Claude or Codex can resume from cold.
STATUS.md is authoritative for *where we are*; DECISIONS.md is authoritative for
*why*; `docs/progress/M<n>-*.md` captures what was built and what was surprising.

### D-05: Model ladder — Gemma 3 1B → 4B (main) → 12B (replicate)
Matches Gemma Scope 2 coverage and circuit-tracer PLT support. Pinning the Gemma 3
family; will revisit only if a downstream result blocks on a newer family.

### D-06: Index-based calibration secret, not name-based
"Your secret is candidate #7" rather than "Your secret is tiger". Keeps calibration
distributionally close to self-chosen and avoids training a decoder on the literal
surface token.

### D-07: Randomize candidate display order per run; log the permutation
Without this, a decoder can entangle concept identity with displayed index and
positional bias. Every `RunManifest` carries `permutation` (displayed order of
canonical IDs) and, for calibration, `secret_displayed_index`.

### D-08: Capture all layers in M2, not one middle layer
One forward pass is cheap; re-running at M3 scale will not be. Offline layer sweeps
become trivial. The most common early regret in projects like this is saving one
layer and having to rerun.

### D-09: Build `feasible_set(history)` (`S_t`) from day 1
Central to the scientific control even though M2 has no questions yet. Lives in
`src/twenty_q/banks.py` with a hand-computed-fixture unit test.

### D-10: Structured run manifest (pydantic JSON), not loose `.pt` files
Every run has a machine-readable manifest recording model/tokenizer revisions,
prompt template ID, seed, decoding params, permutation, calibration secret,
turns, optional reveal, and per-layer activation paths. Comparing runs later
depends on this.

### D-11: M2 exit criteria intentionally relaxed
M2 is a smoke test. Thresholds: nearest-centroid LOO > 20% (chance = 5%) at some
layer, ≥1 binary attribute decoder > 70% at some layer, reveal parse success
≥ 80%. Do not over-optimize at M2.

### D-12: M2 uses 8 runs per candidate (160 calibration runs), not 1
Leave-one-run-out CV with 1 run per class is structurally invalid. 8 runs per
class gives enough variance for readouts to learn something. Scale to ~100/class
at M3.

### D-13: Tooling — uv for env; ruff for lint; pytest for tests
Standard modern Python. `uv sync` from `pyproject.toml`. No poetry.

### D-14: Validator thresholds relaxed — yes-count 1..19, pairwise-diff >=2
The plan's original `5 <= yes-count <= 15` and `pairwise-diff >= 3` were too
strict for a 20-animal bank. (1) Rare taxonomic classes (1 amphibian, 1 insect,
2 reptiles, 2 fish) make the 5-yes floor unreachable by construction. (2)
Reaching pairwise-diff >= 3 for (cow, horse), (dog, cat), (eagle, owl) would
require stuffing the bank with ~6 indicator predicates. Relaxed both. Added 4
targeted distinguishers (`is_ridden_by_humans`, `produces_dairy_milk`, `purrs`,
`soars_during_daylight`) and dropped 2 redundant predicates
(`lives_primarily_on_land`, `is_warm_blooded`). Final bank: 30 questions, all
190 candidate pairs distinguishable on >=2 questions.
