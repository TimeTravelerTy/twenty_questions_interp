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

### D-15: M2 uses transformers directly, not NNsight
For pure activation capture at the Ready position, `AutoModelForCausalLM`'s
`output_hidden_states=True` is simpler and lower-friction than wrapping the
model in NNsight. NNsight comes in at M3+ when we start doing interventions
and need its tracing/patching API. The `capture_ready_state` function in
`src/twenty_q/dialogue.py` is the seam to swap.

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

## 2026-04-19 — M3 dialogue plumbing

### D-16: Persist one all-layer tensor per question turn and replay raw answers
M3 needs `h^{(ℓ)}_{r,t}` at each pre-answer position, not just the Ready state.
Store one `.pt` tensor per turn (`turn_<nn>_activations.pt`) and record its path
in `RunManifest.turn_activation_paths` keyed by 1-based turn index. This keeps
M2's Ready-state `activation_paths` backward-compatible while making turn-wise
capture explicit. When constructing later turns, replay the model's **raw**
earlier answers in the chat history, not a normalized yes/no canonicalization,
so the captured state reflects the actual dialogue the model saw.

### D-17: Do not scale M3 calibration until index-based turnful calibration is fixed
A remote Gemma 3 4B smoke on TSUBAME/H100 confirmed the mechanics: model load,
Ready parse, question-turn capture, and `turn_activation_paths` all work. But
the current **index-based** calibration prompt fails semantically once question
turns start. In the smoke, the model gave obviously wrong answers for fixed
secrets (`eagle -> Is it a mammal? Yes`). Two diagnostics sharpened this:
(1) stronger generic "remember the same secret" turn wording did **not** fix it;
(2) a one-off **name-based** secret assignment did behave sensibly on the same
questions. Provisional conclusion: the bottleneck is the index-based secret
binding under turnful dialogue, not the activation-capture plumbing. Do not
launch the full ~2k calibration run until the calibration prompt/condition is
reworked and passes a remote semantic smoke test.

### D-18: Do not reverse D-06 yet; the 4-condition smoke narrowed the choice but did not clear the gate
The TSUBAME 4-condition follow-up (`docs/progress/M3-4cond-binding-smoke.md`,
job `7217900`) made the ranking clear:

- index-based conditions remain bad (`50%`, `40%`);
- name-based conditions are much better (`90%`, `90%`).

But the `STATUS.md` gate for reversing D-06 was `>=95%` answer correctness on
the name-based conditions, and neither variant reached it. Known bank ambiguity
(`tiger/can_swim`) explains only part of the miss; there are still genuine
errors such as `eagle -> is_mammal: Yes` and `salmon -> is_bird: Yes`.

Decision: keep D-06 unreversed for now. Run one final small binding follow-up
before choosing the replacement calibration regime. Also, treat plain
within-secret cosine as too saturated to use alone for this choice; use either
within-vs-between contrast or direct NC/LR at `Ready` as the representational
gate on the next smoke.

## 2026-04-19 — D-19: Do not reverse D-06; stop and investigate representation persistence

The post-4cond follow-up (3 conditions × 4 candidates × 2 seeds; primary
question set `is_mammal,is_bird,lives_primarily_in_water,has_four_legs`;
`docs/progress/M3-3cond-binding-smoke.md`, job `7218265`) again failed the
≥95% primary-correctness gate:

- `name_paraphrase`: 84.4%
- `name_strict`: 84.4% (the extra "answer only about X" clause did not help
  primary correctness and actively hurt secondary)
- `verbalized_index`: 71.9% — and notably, despite the model verbalizing
  the correct name from the index in all 8/8 runs, yes/no answers at
  question time still drifted toward candidate-list priors

The errors are not prompt-strength artifacts. `eagle.is_mammal=Yes` and
`eagle.can_swim=Yes` occur in **every single eagle run** across all three
conditions; `frog.has_four_legs=No` is systematic under name binding. These
are specific representational failures that the calibration harness is
surfacing.

Decision: D-06 remains in effect, not because index binding is good — it is
not — but because **no binding regime we have tried clears the gate at 4B**.
Do not scale to ~2k. Do not switch calibration to name-based in DECISIONS
until a regime exists that actually produces semantically valid training
examples.

Separately, this run sharpens the research hypothesis from H-binding
("index does not instantiate") into **H-persistence**: at 4B the
instantiated-entity representation does not reliably persist across a chat-
turn boundary, even when name retrieval from an index is correct. This is
directly testable with a within-run cross-turn decoding comparison and is
the most scientifically interesting branch to pursue before another prompt
sweep.

## 2026-04-19 — D-20: Bank ambiguity contributes to the binding smoke misses, but does not rescue calibration

An offline rescore of the 3-condition TSUBAME run
(`docs/progress/M3-binding-bank-audit.md`) tested the most plausible disputed
bank cells without mutating `data/answers.csv`.

The only disputed **primary** cell with a clean, consistent effect was
`frog.has_four_legs`, because the actual surface question is
"Does it walk primarily on four legs?" Flipping that cell from `1 -> 0`
improves the best name-based conditions from `84.4%` to `90.6%`, but still
leaves them well below the `>=95%` gate. Flipping `frog.lives_primarily_in_water`
does not help consistently; it fixes one run and breaks others.

Decision: do **not** opportunistically patch the canonical bank mid-M3 just to
rescue this smoke. The bank should be revisited in a broader audit later, but
the current calibration failure is not primarily a table problem. Move on to
the mechanistic H-persistence test.

## 2026-04-19 — D-21: H-persistence refuted in strong form; replace with H-rotation

The persistence diagnostic (`docs/progress/M3-h-persistence.md`, job
`7218322`) captured two all-layer hidden states per `verbalized_index` run:
State A right before the model verbalizes the secret name, and State B
right before Ready one turn later. Per-layer NC LOO + cross-state
transfer + within-vs-between cosine contrast were computed on 4 candidates
× 2 seeds = 8 runs.

Findings:

- NC LOO reaches **100% at layer 21 for both A and B** (chance 25%), and
  100% at B from layer 6 onward. Entity identity is separable at both
  timepoints.
- **Cross A→B at layer 21 is 37.5%**; middle-layer sub-bands (17–30) are
  ≤75%. Centroids learned at A do not classify B.
- Post-L13 within-vs-between cosine contrast is ~25× larger at A than B
  (+1.28e-02 vs +5.17e-04). Entity signal is still present at B but
  much smaller relative to other components.
- Primary correctness reproduces at 71.9% with the same systematic
  mammal-bias errors (eagle, frog, salmon all say is_mammal=Yes).

Decision:

1. H-persistence is **refuted in its strong form**. The entity does
   persist as identity across the chat-turn boundary at 4B.
2. Adopt **H-rotation** as the working hypothesis: across a chat-turn
   boundary, the entity representation undergoes substantial rotation in
   middle layers and ~25× amplitude collapse relative to the point of
   retrieval. Identity probes still work; attribute probes trained at A
   would not transfer to B, and that rotation is what explains the
   answer-drift we kept surfacing.
3. The readout-pipeline implication: probes must be fit at the same
   dialogue position they will be evaluated at. Readouts fit on a
   "fluent verbalization" state will not transfer to a Ready state one
   turn later, at least at 4B.
4. Next scientific branch: run a small self-chosen smoke at 4B on the
   same primary question set and compare self-chosen Ready activations
   to State A vs State B. The shape of that comparison determines
   whether the blog claim can be "fluent latent secret" (self-chosen ≈
   A) or has to be "decodable-but-rotated latent" (self-chosen ≈ B).
5. Do not reverse D-06. Do not scale calibration yet.

Explicitly kept open, not closed:

- **Model ladder (D-05 still stands):** H-rotation at 4B is a claim
  about 4B. At 12B and 27B, competence likely shifts and the rotation
  may shrink. Re-running `diagnose_persistence.py` when scaling up is
  standing backlog.
- **Bank audit:** the disputed cells from D-20 remain on the backlog.
  The representational finding in this run does not depend on answer
  correctness; it uses candidate-ID labels.
- **Other binding conditions:** only `verbalized_index` was tested for
  cross-state transfer. Whether name-based conditions also rotate is
  untested; if it becomes load-bearing, extend the persistence script
  to capture A for each condition.

## 2026-04-19 — D-22: The self-chosen A-vs-B comparator should be a 4-candidate, reveal-labeled Ready smoke

The next branch after D-21 is not a full 20-way self-chosen run. The
question is specifically whether **self-chosen Ready** looks more like
the persistence diagnostic's State A (strong, probe-ready entity geometry)
or State B (still decodable identity, but rotated and amplitude-collapsed).

Decision:

1. Restrict the self-chosen smoke to the same 4 candidates used in
   `diagnose_persistence.py` (`tiger,eagle,frog,salmon`) and the same
   primary question set. This makes the geometry directly comparable and
   keeps chance at 25% for NC LOO.
2. Label each self-chosen run by a **post-dialogue reveal**, not by a
   pre-question reveal. Ready-state analysis should be evaluated against
   the secret the model says it carried through the dialogue.
3. Run until a small quota per candidate is filled, rather than a fixed
   number of attempts. Self-chosen choice frequencies are skewed, so a
   flat `n` can leave one class absent and make NC LOO ill-posed.

Implementation consequence: add a dedicated script
`scripts/diagnose_selfchosen_ready.py` instead of overloading the older
`run_selfchosen_smoke.py` runner.

## 2026-04-19 — D-23: Self-chosen Ready is ~6.7× weaker than State B; probes fit at self-chosen Ready, not transferred

Result from `runs/diag/selfchosen_ready_smoke_20260419/` (see
`docs/progress/M3-selfchosen-ready-smoke.md`). Two concrete findings drive
this decision:

1. **Choice distribution collapses under greedy decoding.** Across 40
   attempts with 4 candidates (`tiger, eagle, frog, salmon`), the model
   picked salmon 33× and frog 7×; tiger and eagle were never chosen.
   Strong position bias: salmon-at-position-0 is chosen 15/15. Tiger/eagle
   at position 0 default to salmon. "Self-chosen" under greedy is closer
   to biased-forced-choice.
2. **Self-chosen Ready geometry is weaker than State B.** In a balanced
   2-class (salmon vs frog, n=7 each) analysis, the post-13 within-between
   cosine contrast is +7.85e-05 vs State B's +5.26e-04 (~6.7×) and State A's
   +1.31e-02 (~166×). Identity is NC-decodable at 100% only from layer 29
   onward — not at the mid layers (~L21) where A and B both peak.

Ordering: A (post-verbalization) > B (pre-Ready after lock-in) > self-chosen
Ready. Self-chosen is the most-collapsed regime of the three.

Decisions:

- **Probe-training location** (reinforces D-21): attribute readouts must be
  fit at self-chosen Ready directly. Transfer from calibration positions
  that resemble State A or State B will fail.
- **SAE feature case studies (M5)** should target layers ≥ L29 if the goal
  is to capture self-chosen identity features. Mid-layer features that work
  during calibration may not be there at self-chosen Ready.
- **Follow-up run design**: the quota-based 4-candidate script
  (`diagnose_selfchosen_ready.py`) cannot satisfy its own A-vs-B vote when
  the choice distribution collapses to 2 classes. The next self-chosen run
  should either (a) use temperature sampling (e.g. T=0.7) to broaden the
  realized reveal distribution, (b) expand to a larger candidate set so
  4+ realized classes are plausible, or (c) both. The script should also
  be patched to fall back to a restricted-class analysis when the full
  quota is not reached, rather than aborting.

Kept open, not closed:

- **D-05 model ladder.** At 12B+, the choice distribution may broaden and
  the geometry may be cleaner. Re-running this diagnostic when scaling is
  standing backlog.
- **Bank audit.** The disputed cells from D-20 still apply to the realized
  salmon/frog runs; the representational finding does not depend on
  answer correctness.
- **Full 20-candidate self-chosen.** Whether the 20-way prompt produces
  a wider distribution under greedy is untested.

### D-24: T=0.7 sampling does not break the 4-candidate choice collapse at 4B

Follow-up to D-23 (`docs/progress/M3-selfchosen-ready-T07.md`, job `7219788`).
120-attempt T=0.7 self-chosen replication, same 4-candidate panel:

- Distribution remains 2-class: **salmon 96, frog 24, tiger 0, eagle 0**.
  Sampling at T=0.7 does not surface tiger or eagle in 120 attempts — the
  model's self-chosen prior at 4B is approximately a mixture of salmon
  (dominant) and frog (minor), with near-zero mass on the other two.
- Position-0 diagnostic: salmon is a strong attractor even when eagle
  (31/31 → salmon) or tiger (20/24 → salmon) is shown first. Only frog@0
  reliably flips the choice (20/26).
- Geometry improved modestly: post-13 within-between contrast rose from
  +7.85e-05 (T=0) to +1.14e-04 (T=0.7, 1.45×), and best-NC layer shifted
  from L29 to L24. Still ~4.5× weaker than persistence State B at matched
  2-class sample size.
- Primary-question correctness dropped to 53% (vs higher at T=0) — sampling
  noise degrades the yes/no fluency that downstream probes assume.

Decisions:

- **The 4-candidate self-chosen smoke is closed as a 4-way test at 4B.**
  Neither greedy nor T=0.7 realizes the 4 classes the diagnostic was
  designed to compare. Any further 4-way self-chosen analysis needs the
  broader 20-candidate prompt (to dilute the salmon attractor) or a bigger
  model (D-05).
- **M4/M5 guidance unchanged from D-23.** Probes fit at self-chosen Ready
  directly; SAE case studies target layers ≥ L24 (relaxed one block earlier
  than D-23's L29 given the T=0.7 shift).
- **Sampling in the diagnostic is a knob, not the fix.** Temperature widens
  the geometric gap by a small factor but does not reach a competence
  regime where the prompt is fair. The next productive self-chosen run
  must widen the candidate set.

Kept open, not closed:

- **D-05 model ladder.** Still a standing retest target.
- **Bank audit.** Unchanged.
- **20-candidate self-chosen.** Now promoted from "optional" to the main
  next self-chosen experiment in STATUS.md.
