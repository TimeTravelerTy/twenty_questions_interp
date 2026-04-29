# Design decisions log

> Append-only, dated. Record non-obvious choices and enough reasoning to judge
> later whether they still hold. If a decision is reversed, add a new entry;
> do not edit the old one.

## 2026-04-18 — Project bootstrap decisions

### D-01: Calibration is infrastructure, not the scientific result
The blog claim lives in self-chosen data. Calibration (index-supplied secret)
is only for readout training and plumbing checks. **Never** headline a
calibration-only result. Stated in STATUS.md and enforced by `S_t` from M1.

### D-02: Dev env — local Mac CPU with Gemma 3 1B, then TSUBAME
Iteration speed matters more than throughput at M0-M2. Gemma 3 1B is ~2 GB and
runs on CPU; later TSUBAME porting friction is acceptable. Locked in until M3.

### D-03: Data artifacts before any model code
Prevents wasting runs if a bank changes. Every run manifest pins the banks it
used; keep that pin stable before collecting activations.

### D-04: Handoff via STATUS.md + per-milestone notes + DECISIONS.md
Durable file-based context that Claude or Codex can resume from cold.
STATUS.md says *where we are*; DECISIONS.md says *why*; `docs/progress/M<n>-*.md`
records what was built and what was surprising.

### D-05: Model ladder — Gemma 3 1B -> 4B (main) -> 12B (replicate)
Matches Gemma Scope 2 coverage and circuit-tracer PLT support. Pin the Gemma 3
family; revisit only if a downstream result requires a newer family.

### D-06: Index-based calibration secret, not name-based
Use "your secret is candidate #7" rather than "your secret is tiger". This keeps
calibration close to self-chosen and avoids training on the literal token.

### D-07: Randomize candidate display order per run; log the permutation
Without this, a decoder can entangle identity with displayed index and position.
Every `RunManifest` stores `permutation` and, for calibration,
`secret_displayed_index`.

### D-08: Capture all layers in M2, not one middle layer
One forward pass is cheap; rerunning at M3 scale will not be. Capturing all
layers makes offline sweeps trivial and avoids the usual rerun regret.

### D-09: Build `feasible_set(history)` (`S_t`) from day 1
Central to the scientific control even though M2 has no questions yet. Lives in
`src/twenty_q/banks.py` with a hand-computed fixture test.

### D-10: Structured run manifest (pydantic JSON), not loose `.pt` files
Every run needs a machine-readable manifest with model/tokenizer revisions,
prompt template ID, seed, decoding params, permutation, calibration secret,
turns, optional reveal, and per-layer activation paths.

### D-11: M2 exit criteria intentionally relaxed
M2 is a smoke test. Thresholds: nearest-centroid LOO > 20% (chance 5%) at some
layer, at least one binary attribute decoder > 70%, reveal parse success >= 80%.
Do not over-optimize M2.

### D-12: M2 uses 8 runs per candidate (160 calibration runs), not 1
Leave-one-run-out CV with 1 run per class is structurally invalid. 8 runs per
class gives enough variance for readouts; scale to ~100/class at M3.

### D-13: Tooling — uv for env; ruff for lint; pytest for tests
Standard modern Python. `uv sync` from `pyproject.toml`. No poetry.

### D-15: M2 uses transformers directly, not NNsight
For pure activation capture at Ready, `AutoModelForCausalLM` with
`output_hidden_states=True` is simpler than NNsight. Bring in NNsight at M3+
for interventions and tracing/patching. `capture_ready_state` in
`src/twenty_q/dialogue.py` is the swap seam.

### D-14: Validator thresholds relaxed — yes-count 1..19, pairwise-diff >=2
The original `5 <= yes-count <= 15` and `pairwise-diff >= 3` were too strict
for a 20-animal bank. Rare classes made the yes-floor unreachable, and `>=3`
would have required stuffing the bank with many indicator predicates. Relaxed
both, added `is_ridden_by_humans`, `produces_dairy_milk`, `purrs`, and
`soars_during_daylight`, dropped `lives_primarily_on_land` and
`is_warm_blooded`. Final bank: 30 questions; all 190 pairs distinguishable on
at least 2 questions.

## 2026-04-19 — M3 dialogue plumbing

### D-16: Persist one all-layer tensor per question turn and replay raw answers
M3 needs `h^{(ℓ)}_{r,t}` at each pre-answer position, not just Ready. Store one
`.pt` tensor per turn (`turn_<nn>_activations.pt`) and record its path in
`RunManifest.turn_activation_paths` keyed by 1-based turn index. This keeps
M2's Ready-state `activation_paths` backward-compatible while making turn-wise
capture explicit. Replay the model's **raw** earlier answers in later turns, not
a normalized yes/no form, so the captured state matches the actual dialogue.

### D-17: Do not scale M3 calibration until index-based turnful calibration is fixed
A remote Gemma 3 4B TSUBAME/H100 smoke confirmed model load, Ready parse,
question-turn capture, and `turn_activation_paths`. The problem is semantic:
the current **index-based** calibration prompt breaks once question turns start
(`eagle -> Is it a mammal? Yes`). Stronger generic "remember the same secret"
wording did not fix it; a one-off **name-based** assignment did. The bottleneck
is index-based secret binding, not capture plumbing. Do not launch the full
~2k calibration run until the prompt/condition passes a remote semantic smoke.

### D-18: Do not reverse D-06 yet; the 4-condition smoke narrowed the choice but did not clear the gate
The TSUBAME 4-condition follow-up (`docs/progress/M3-4cond-binding-smoke.md`,
job `7217900`) made the ranking clear:

- index-based conditions remain bad (`50%`, `40%`);
- name-based conditions are much better (`90%`, `90%`).

But the `STATUS.md` gate for reversing D-06 was `>=95%` answer correctness on
the name-based conditions, and neither variant reached it. Known bank
ambiguity (`tiger/can_swim`) explains only part of the miss; genuine errors
remain (`eagle -> is_mammal: Yes`, `salmon -> is_bird: Yes`).

Decision: keep D-06 unreversed for now. Run one final small binding follow-up
before choosing the replacement calibration regime. Also, treat plain
within-secret cosine as too saturated to use alone; use within-vs-between
contrast or direct NC/LR at `Ready` as the representational gate.

## 2026-04-19 — D-19: Do not reverse D-06; stop and investigate representation persistence

The post-4cond follow-up (3 conditions x 4 candidates x 2 seeds; primary
question set `is_mammal,is_bird,lives_primarily_in_water,has_four_legs`;
`docs/progress/M3-3cond-binding-smoke.md`, job `7218265`) again failed the
>=95% primary-correctness gate:

- `name_paraphrase`: 84.4%
- `name_strict`: 84.4% (the extra "answer only about X" clause did not help
  primary correctness and actively hurt secondary)
- `verbalized_index`: 71.9% -- and despite the model verbalizing the correct
  name from the index in all 8/8 runs, yes/no answers at question time still
  drifted toward candidate-list priors

The errors are not prompt-strength artifacts. `eagle.is_mammal=Yes` and
`eagle.can_swim=Yes` occur in **every eagle run** across all three conditions;
`frog.has_four_legs=No` is systematic under name binding. These are specific
representational failures, not prompt noise.

Decision: D-06 remains in effect, not because index binding is good -- it is
not -- but because **no binding regime we have tried clears the gate at 4B**.
Do not scale to ~2k. Do not switch calibration to name-based in DECISIONS
until a regime actually produces semantically valid training examples.

Separately, this run sharpens the research hypothesis from H-binding ("index
does not instantiate") into **H-persistence**: at 4B the instantiated-entity
representation does not reliably persist across a chat-turn boundary, even
when name retrieval from an index is correct. This is directly testable with a
within-run cross-turn decoding comparison and is the most interesting branch
before another prompt sweep.

## 2026-04-19 — D-20: Bank ambiguity contributes to the binding smoke misses, but does not rescue calibration

An offline rescore of the 3-condition TSUBAME run
(`docs/progress/M3-binding-bank-audit.md`) tested the most plausible disputed
bank cells without mutating `data/answers.csv`.

The only disputed **primary** cell with a clean, consistent effect was
`frog.has_four_legs`, because the actual surface question is
"Does it walk primarily on four legs?" Flipping that cell from `1 -> 0`
improves the best name-based conditions from `84.4%` to `90.6%`, but still
leaves them below the `>=95%` gate. Flipping `frog.lives_primarily_in_water`
does not help consistently; it fixes one run and breaks others.

Decision: do **not** opportunistically patch the canonical bank mid-M3 just to
rescue this smoke. Revisit the bank later in a broader audit; the current
calibration failure is not primarily a table problem. Move on to H-persistence.

## 2026-04-19 — D-21: H-persistence refuted in strong form; replace with H-rotation

The persistence diagnostic (`docs/progress/M3-h-persistence.md`, job `7218322`)
captured two all-layer hidden states per `verbalized_index` run: State A right
before the model verbalizes the secret name, and State B right before Ready one
turn later. Per-layer NC LOO + cross-state transfer + within-vs-between cosine
contrast were computed on 4 candidates x 2 seeds = 8 runs.

Findings:

- NC LOO reaches **100% at layer 21 for both A and B** (chance 25%), and 100%
  at B from layer 6 onward. Entity identity is separable at both timepoints.
- **Cross A->B at layer 21 is 37.5%**; middle-layer sub-bands (17-30) are
  <=75%. Centroids learned at A do not classify B.
- Post-L13 within-vs-between cosine contrast is ~25x larger at A than B
  (+1.28e-02 vs +5.17e-04). Entity signal is still present at B but much
  smaller relative to other components.
- Primary correctness reproduces at 71.9% with the same systematic mammal-bias
  errors (eagle, frog, salmon all say is_mammal=Yes).

Decision:

1. H-persistence is **refuted in its strong form**. Identity does persist
   across the chat-turn boundary at 4B.
2. Adopt **H-rotation**: across a chat-turn boundary, entity representation
   rotates in middle layers and collapses in amplitude by ~25x relative to
   retrieval. Identity probes still work; attribute probes trained at A would
   not transfer to B, which explains the answer drift.
3. Probes must be fit at the same dialogue position they will be evaluated at.
   Readouts fit on a "fluent verbalization" state will not transfer to a Ready
   state one turn later, at least at 4B.
4. Next scientific branch: run a small self-chosen smoke at 4B on the same
   primary question set and compare self-chosen Ready activations to State A
   vs State B. That decides whether the blog claim can be "fluent latent
   secret" (self-chosen ~= A) or must be "decodable-but-rotated latent"
   (self-chosen ~= B).
5. Do not reverse D-06. Do not scale calibration yet.

Explicitly kept open, not closed:

- **Model ladder (D-05 still stands):** H-rotation at 4B is a claim about 4B.
  At 12B and 27B, competence likely shifts and the rotation may shrink.
  Re-running `diagnose_persistence.py` when scaling up is standing backlog.
- **Bank audit:** the disputed cells from D-20 remain on the backlog. The
  representational finding in this run does not depend on answer correctness;
  it uses candidate-ID labels.
- **Other binding conditions:** only `verbalized_index` was tested for cross-
  state transfer. Whether name-based conditions also rotate is untested; if it
  becomes load-bearing, extend the persistence script to capture A for each
  condition.

## 2026-04-19 — D-22: The self-chosen A-vs-B comparator should be a 4-candidate, reveal-labeled Ready smoke

The next branch after D-21 is not a full 20-way self-chosen run. The question is
whether **self-chosen Ready** looks more like State A (strong, probe-ready
entity geometry) or State B (still decodable identity, but rotated and
amplitude-collapsed).

Decision:

1. Restrict the self-chosen smoke to the same 4 candidates used in
   `diagnose_persistence.py` (`tiger,eagle,frog,salmon`) and the same primary
   question set. This keeps the geometry directly comparable and chance at 25%
   for NC LOO.
2. Label each self-chosen run by a **post-dialogue reveal**, not by a
   pre-question reveal. Ready-state analysis should be evaluated against the
   secret the model says it carried through the dialogue.
3. Run until a small quota per candidate is filled, rather than a fixed number
   of attempts. Choice frequencies are skewed, so a flat `n` can leave one class
   absent and make NC LOO ill-posed.

Implementation consequence: add `scripts/diagnose_selfchosen_ready.py` instead
of overloading `run_selfchosen_smoke.py`.

## 2026-04-19 — D-23: Self-chosen Ready is ~6.7x weaker than State B; probes fit at self-chosen Ready, not transferred

Result from `runs/diag/selfchosen_ready_smoke_20260419/` (see
`docs/progress/M3-selfchosen-ready-smoke.md`). Two findings drive this decision:

1. **Choice distribution collapses under greedy decoding.** Across 40 attempts
   with 4 candidates (`tiger, eagle, frog, salmon`), the model picked salmon 33x
   and frog 7x; tiger and eagle were never chosen. Strong position bias:
   salmon-at-position-0 is chosen 15/15. Tiger/eagle at position 0 default to
   salmon. "Self-chosen" under greedy is closer to biased forced choice.
2. **Self-chosen Ready geometry is weaker than State B.** In a balanced 2-class
   (salmon vs frog, n=7 each) analysis, the post-13 within-between cosine
   contrast is +7.85e-05 vs State B's +5.26e-04 (~6.7x) and State A's
   +1.31e-02 (~166x). Identity is NC-decodable at 100% only from layer 29
   onward -- not at the mid layers (~L21) where A and B both peak.

Ordering: A (post-verbalization) > B (pre-Ready after lock-in) > self-chosen
Ready. Self-chosen is the most-collapsed regime of the three.

Decisions:

- **Probe-training location** (reinforces D-21): attribute readouts must be fit
  at self-chosen Ready directly. Transfer from calibration positions that
  resemble State A or State B will fail.
- **SAE feature case studies (M5)** should target layers >= L29 if the goal is
  to capture self-chosen identity features. Mid-layer features that work during
  calibration may not be there at self-chosen Ready.
- **Follow-up run design**: the quota-based 4-candidate script
  (`diagnose_selfchosen_ready.py`) cannot satisfy its own A-vs-B vote when the
  choice distribution collapses to 2 classes. The next self-chosen run should
  either (a) use temperature sampling (e.g. T=0.7) to broaden the realized
  reveal distribution, (b) expand to a larger candidate set so 4+ realized
  classes are plausible, or (c) both. The script should also fall back to a
  restricted-class analysis when the full quota is not reached.

Kept open, not closed:

- **D-05 model ladder.** At 12B+, the choice distribution may broaden and the
  geometry may be cleaner. Re-running this diagnostic when scaling is standing
  backlog.
- **Bank audit.** The disputed cells from D-20 still apply to the realized
  salmon/frog runs; the representational finding does not depend on answer
  correctness.
- **Full 20-candidate self-chosen.** Whether the 20-way prompt produces a wider
  distribution under greedy is untested.

## 2026-04-19 — D-24: T=0.7 sampling does not break the 4-candidate choice collapse at 4B

Follow-up to D-23 (`docs/progress/M3-selfchosen-ready-T07.md`, job `7219788`).
120-attempt T=0.7 self-chosen replication, same 4-candidate panel:

- Distribution remains 2-class: **salmon 96, frog 24, tiger 0, eagle 0**.
  Sampling at T=0.7 does not surface tiger or eagle in 120 attempts -- the
  model's self-chosen prior at 4B is roughly salmon-dominant with frog as a
  minor mode and near-zero mass on the other two.
- Position-0 diagnostic: salmon is a strong attractor even when eagle (31/31 ->
  salmon) or tiger (20/24 -> salmon) is shown first. Only frog@0 reliably flips
  the choice (20/26).
- Geometry improved modestly: post-13 within-between contrast rose from
  +7.85e-05 (T=0) to +1.14e-04 (T=0.7, 1.45x), and best-NC layer shifted from
  L29 to L24. Still ~4.5x weaker than persistence State B at matched 2-class
  sample size.
- Primary-question correctness dropped to 53% (vs higher at T=0) -- sampling
  noise degrades the yes/no fluency that downstream probes assume.

Decisions:

- **The 4-candidate self-chosen smoke is closed as a 4-way test at 4B.**
  Neither greedy nor T=0.7 realizes the 4 classes the diagnostic was designed
  to compare. Any further 4-way self-chosen analysis needs the broader
  20-candidate prompt or a bigger model (D-05).
- **M4/M5 guidance unchanged from D-23.** Probes fit at self-chosen Ready
  directly; SAE case studies target layers >= L24 (relaxed one block earlier
  than D-23's L29 given the T=0.7 shift).
- **Sampling in the diagnostic is a knob, not the fix.** Temperature widens the
  geometric gap by a small factor but does not reach a competence regime where
  the prompt is fair. The next productive self-chosen run must widen the
  candidate set.

Kept open, not closed:

- **D-05 model ladder.** Still a standing retest target.
- **Bank audit.** Unchanged.
- **20-candidate self-chosen.** Now promoted from "optional" to the main next
  self-chosen experiment in STATUS.md.

## 2026-04-20 — D-25: Do not standardize 4B calibration; matched 5-way `name_paraphrase` still fails at 85%

After the 20-bank self-chosen smoke (`docs/progress/M3-selfchosen-20bank.md`,
job `7223018`) realized a 5-class subset with quota (`dolphin,horse,penguin,
crocodile,shark`), the pragmatic next test was to pick the best calibration
schema and see whether it cleared the semantic gate on the more realistic
subset.

We ran exactly that: `name_paraphrase` only, 5 candidates x 2 seeds on 4B
(job `7226502`). Result:

- Ready parse: 10/10
- Primary correctness: **34/40 = 85.0%**
- Secondary `can_swim`: 9/10 = 90.0%
- Post-L13 within-vs-between contrast: **+5.41e-04**

This is effectively the same correctness regime as the earlier 4-candidate
name-based smokes, not a recovery. Restricting to the self-chosen-realized
subset does **not** rescue calibration at 4B.

Decision:

1. Do **not** standardize `name_paraphrase` as the 4B calibration harness.
2. Do **not** launch the full ~2k calibration run at 4B.
3. Treat calibration at 4B as still unresolved infrastructure, not the path to
   probe training.

## 2026-04-20 — D-26: 20-bank self-chosen fixes class collapse enough to analyze, but 4B Ready remains far weaker than persistence B; escalate D-05

The 20-bank self-chosen run on 4B (job `7223018`) materially changed the class
distribution relative to the old 4-candidate panel:

- reveals over 200 attempts: **dolphin 104, penguin 85, shark 5, crocodile 2,
  horse 2, cow 1, salmon 1**
- quota `n=2` reached for **5 classes**

So the broader prompt is worth keeping. But the representational result is still
weak:

- balanced 5-way self-chosen Ready: best post-L13 NC = **10.0%** (chance 20%)
- self-chosen post-L13 contrast: **+4.70e-06**

Matched persistence on the same 5 classes (job `7226501`) remains strong:

- State A NC at L21: **100%**
- State B NC at L21: **100%**
- cross A->B at L21: **90%**
- State B post-L13 contrast: **+4.87e-04**

This makes self-chosen 20-bank Ready about **104x weaker than persistence
State B** on the same class set. By the current vote rule the matched
comparison is still formally "mixed" (`nc=tie`, `contrast=state_b`), but the
scale is the real finding: 4B is not yet in a probe-ready self-chosen regime.

Decision:

1. Keep the **20-bank prompt** for self-chosen work at 4B+; it solved the 4-way
   collapse enough to expose more classes.
2. Do **not** spend more time on 4B calibration prompt variants right now.
3. Escalate **D-05**: next productive branch is a 12B self-chosen 20-bank
   replicate before any full calibration launch.

## 2026-04-20 — D-27: Standardize `name_paraphrase` at 12B; 4B remains unresolved

The 12B branch answered the user's "pick one schema and move on if it passes"
proposal.

Sequence:

1. `7226538` (`google/gemma-3-12b-it`, 20-bank self-chosen, greedy) realized a
   4-class subset with quota: `elephant,cow,dog,horse`.
2. The legacy 4-question panel turned out to be degenerate on that subset, so we
   switched the matched controls to a six-question panel that actually separates
   those four animals: `is_carnivore,is_larger_than_human,is_domesticated,
   lives_in_africa,produces_dairy_milk,is_ridden_by_humans`.
3. On that panel:
   - matched persistence 12B (`7226546`) stayed strong:
     - primary correctness **45/48 = 93.8%**
     - NC-A **100%** at L30
     - NC-B **75%** at L30
   - matched calibration 12B, `name_paraphrase` (`7226547`) hit
     **47/48 = 97.9%**

Decision:

1. **Standardize `name_paraphrase` as the calibration schema at 12B.**
2. This is the first regime that clears the long-standing `>=95%` gate, so it is
   now valid to use calibration as Ready-state probe-training infrastructure.
3. Do **not** infer from this that 4B is fixed. 4B calibration remains
   unresolved and should stay off the main path.
4. The next concrete step is no longer "find a better calibration prompt." It is
   "collect a pilot 12B Ready-state calibration set with `name_paraphrase` and
   start training dense readouts."

## 2026-04-21 — D-28: 12B calibration Ready probes still do not transfer to self-chosen; keep the 20-bank prompt and fit self-chosen directly

The 12B transfer test was run in the strongest currently available regime:

- calibration source: `docs/progress/M3-12b-pilot-readouts.md`
  (`runs/calibration/12b_name_paraphrase_4way_pilot_20260420/`),
  100 runs on `{elephant,cow,dog,horse}`
- decisive transfer layers: **L6, L17, L27, L48**
- new self-chosen run: explicit 4-way prompt on the same subset, job `7230657`
- retrospective check: the earlier 20-bank 12B self-chosen run, restricted to
  the kept reveals on the same four animals

Results:

1. **The explicit 4-way self-chosen prompt collapses harder than the 20-bank
   prompt.** Over 160 attempts, the new run realizes only `cow=10` and
   `horse=10`; `elephant` and `dog` never appear.
2. **Calibration -> self-chosen transfer is poor on that collapsed 4-way run.**
   On the balanced 20-run kept set, agreement with reveal is:
   - NC: `0.35 / 0.00 / 0.00 / 0.10` at `L6 / L17 / L27 / L48`
   - LR: `0.00 / 0.00 / 0.00 / 0.10`
   This is below even the trivial 50% baseline induced by the 2-class collapse.
3. **The failure is not just the narrowed prompt.** On the earlier natural
   20-bank 12B self-chosen run, restricting to the kept 8-run slice
   (2 x `{elephant,cow,dog,horse}`), transfer stays at **12.5–25.0%**,
   essentially chance.
4. **The error mode is class collapse.** The transferred decoders often map
   *every* self-chosen run onto a single calibration class (`cow`, `elephant`,
   or `dog`) depending on layer.

Decision:

1. **Do not use calibration-trained Ready probes as self-chosen readouts at
   12B.** D-23's "fit where you evaluate" rule survives the model-ladder step.
2. **Keep calibration in the infrastructure role only.** It is still valuable
   for dense readout debugging and geometry inspection, but not as a direct
   training source for self-chosen probes.
3. **Do not use the explicit 4-way self-chosen prompt as the main branch.**
   It worsens reveal diversity relative to the natural 20-bank prompt.
4. **Keep the 20-bank prompt for self-chosen collection at 12B and fit probes
   directly on self-chosen Ready.** The next productive artifact is a larger
   12B self-chosen 20-bank dataset with enough runs per realized class for LOO
   or train/test readouts.

## 2026-04-21 — D-29: 12B self-chosen Ready direct-fit is also weak; sweep mid-dialogue pre-answer positions before anything else

D-28 said "fit probes directly on self-chosen Ready at 12B". That experiment
has now run:

- collection: job `7230807`, 20-bank prompt, T=0.0, 300 attempts
- realized classes: `{elephant, cow, dog, horse}` (same 4 as every previous
  12B self-chosen collection; narrowing does not broaden the distribution)
- kept: 40 runs balanced at 10/class, primary-correct on the 4-question panel
- scoring: LOO NC and LR at every layer, chance 25%

Result (see `docs/progress/M3-12b-selfchosen-direct.md`):

- NC LOO: mean **0.23**, median 0.22, max **0.45 @ L14**
- LR LOO: mean **0.27**, median 0.28, max **0.45 @ L4**
- best NC and LR layers disagree (L14 vs L4); no coherent band of high
  accuracy across depth

Compare to 12B calibration on the same 4 classes: LR LOO 1.00 from L6; NC
LOO 1.00 from L27. Direct-fit at self-chosen Ready is ~chance on average
with ~1.8x-chance single-layer peaks that look more like multiple-testing
noise than a real code.

Decision:

1. **Self-chosen Ready is not a probe-ready position at 12B.** The rule is
   now stronger than D-23: it's not only "fit probes where you evaluate" —
   the Ready position itself does not carry enough self-chosen class signal
   to decode at this sample size, regardless of where you fit.
2. **Do not probe "State A / State B" in the self-chosen condition.** An
   earlier draft of this entry proposed exactly that. It is incoherent:
   State A / State B were defined (D-21) as residual-stream positions
   *after* the model verbalizes the secret name in context (calibration or
   post-reveal). Self-chosen by construction never lets the name enter
   context, so there is no in-context State A/B to decode. Putting the name
   in context to create State A would defeat the secrecy condition — that
   is the calibration regime, not self-chosen.
3. **Next probing position to test is mid-dialogue pre-answer.** Ready is
   emitted immediately after the model commits; by the time the model has
   answered 3-4 yes/no questions, the commitment has been exercised and
   the choice may be more crystallized in the residual stream. The
   `diagnose_selfchosen_ready.py` pipeline already captures
   `turn_0k_activations.pt` at each pre-answer position, so this test
   requires no new TSUBAME collection — just running LOO NC and LR at
   turns 1..4 on the existing 40 kept runs.
4. **Do not keep increasing Ready-state self-chosen n at 12B** until (3)
   resolves. More samples on a position that does not separate is not the
   most informative next experiment.
5. **Open question kept on the backlog:** broader realized class diversity.
   Every 12B self-chosen collection so far (4-way narrowed, 20-bank, 20-bank
   direct-fit) has realized exactly `{elephant, cow, dog, horse}`. This is
   a separate blocker from the probe-position question, but testing
   mid-dialogue on those four will still answer whether the self-chosen
   class code is decodable *anywhere* pre-reveal.

## 2026-04-21 — D-30: At 12B, the right self-chosen probe position is turn-4 pre-answer, not Ready

The D-29 follow-up has now run on the existing 40 kept self-chosen runs from
job `7230807` (10 x `{elephant,cow,dog,horse}`), using the already-captured
`turn_01..turn_04_activations.pt`.

Result (see `docs/progress/M3-12b-selfchosen-turns.md`):

- **Turn 4** is materially stronger than Ready:
  - Ready: NC mean **0.23**, LR mean **0.27**, best **0.45**
  - Turn 4 pre-answer: NC mean **0.40**, LR mean **0.40**,
    best NC **0.625 @ L44**, best LR **0.60 @ L42**
- The turn-4 signal is a coherent **late-layer band**, not a one-layer fluke:
  - NC mean over L27..48 = **0.549**
  - LR mean over L27..48 = **0.539**
- The pattern is **not monotone** across turns:
  - turn 1 is moderately decodable
  - turns 2 and 3 are weak
  - turn 4 becomes clearly strongest

Crucial control: on the realized kept subset `{elephant,cow,dog,horse}`, the
current 4-question panel is degenerate (`1,0,0,1` for every class). So the
turn-4 decode cannot be explained by public answer history; it reflects a
latent class code that sharpens under repeated commitment.

Decision:

1. **Move the main 12B self-chosen probe position from Ready to turn-4
   pre-answer.**
2. **Stop doing new Ready-state self-chosen analysis as the main branch.**
   Ready is now a solved negative control.
3. **Scale data at the winning position next.** The next productive artifact is
   a larger 12B self-chosen 20-bank collection, analyzed primarily at late
   turn-4 layers (`~L42–L48`).

## 2026-04-21 — D-31: Turn-4 pre-answer survives scale-up; probe position is locked, class diversity is the next blocker

The D-30 scale-up (job `7232075`, 600 attempts, 80 kept = 20 x
`{elephant,cow,dog,horse}`) sharpens the turn-4 signal well past the
probe-ready threshold (see `docs/progress/M3-12b-selfchosen-turn4scale.md`):

- turn 4 LR LOO **0.787 @ L31**, NC **0.662 @ L29** (chance 0.25)
- turn 4 L27-48 **LR mean 0.731**, NC mean 0.558
- turn 1 L27-48 LR mean 0.431 (above chance but far below turn 4)
- compared to the n=40 pilot, **LR jumps +0.19 in both mean and max** while
  NC barely moves — consistent with linearly separable geometry that was
  regularization-starved at n=40

The scale-up did not create a new signal; it sharpened the pilot signal
into a robust one. The STATUS threshold "~70% regime to lock this probe
position" is cleared.

Decision:

1. **Turn-4 pre-answer, late layers (L26-L48, peaks L29-L31) is the locked
   self-chosen probe position for M4.** Causal patching, SAE feature case
   studies, and the blog-post readout all build on this position.
2. **The probe-position question is closed.** Further sweeping of alternative
   residual positions on this dataset is not a useful next step unless a
   specific causal-patching experiment needs a different index.
3. **The next blocker is realized-class diversity, not position.** Every
   12B greedy self-chosen collection so far realizes only
   `{elephant, cow, dog, horse}`. LR 0.79 at 4 classes is compelling
   infrastructure; the scientific claim needs to show the signal survives
   a broader class set. Attack that via T>0 sampling on the same 20-bank
   prompt before falling back to prompt variants.
4. **Do not re-open Ready** as a probing position for self-chosen. The
   Ready-vs-turn-4 gap (LR mean 0.27 vs 0.73 on the pilot; 0.34 vs 0.73
   on scale-up turn-1 L27-48, which sits in between) is too large to
   plausibly flip with more samples.

## 2026-04-21 — D-32: Diversity runs should stop on a class-quota target and emit diversity/parse metrics directly

The next operational branch after D-31 is no longer "can we decode turn 4 at
all?" but "can we broaden the realized class set without destroying
instruction-following?" The existing `diagnose_selfchosen_ready.py` loop was
fine for small smokes, but on the 20-bank diversity run it would only stop at
`max_attempts` unless **every** class hit quota, and it did not summarize the
choice distribution directly in `results.json`.

Decision:

1. Add an optional `--stop-when-n-classes-hit-quota` flag to
   `diagnose_selfchosen_ready.py` so a diversity run can terminate once a
   scientifically useful number of classes reach `--n-per-candidate`
   (for example 8), instead of waiting for the full 20-bank to saturate.
2. Emit direct attempt-level diagnostics in `results.json` and stdout:
   parsed reveal counts, number of distinct parsed classes, ready/reveal/answer
   parse rates, top-1 reveal share, and entropy/effective-class-count of the
   parsed reveal distribution.
3. Treat those diagnostics as the gate for the T=0.7 branch before decoding
   turn 4: if diversity broadens but parse success collapses, that is a prompt
   / instruction-following failure, not evidence about latent-state geometry.

## 2026-04-22 — D-33: T=0.7 does not broaden 12B self-chosen — attractor is prompt-induced, not sampling noise; accept 4 classes as the M3 headline and move to M4

Job `7237460` ran `diagnose_selfchosen_ready.py` at T=0.7 on the 20-bank
prompt with 1500 attempts, 12B bf16, quota-8 early stop. Reveal and
Ready parse success were 100%, but all 1500 reveals collapsed onto
the same 4-class attractor as greedy:
horse 572 / cow 520 / elephant 307 / dog 101, with zero realizations
of the other 16 candidates (tiger, kangaroo, bat, dolphin, gorilla,
cat, eagle, penguin, chicken, owl, cobra, crocodile, frog, shark,
salmon, bee). Effective classes 3.46, top-1 share 38.1%. The
candidate list is permuted per seed, so this is not a list-position
artifact — it is a genuine concentration of the 12B posterior over
"which animal to keep as secret" on this prompt.

See `docs/progress/M3-12b-selfchosen-diversity-T07.md`.

Decision:

1. **Treat the 12B self-chosen problem on this prompt as a 4-class
   problem.** The turn-4 LR LOO 0.787 @ L31 result on n=80 (chance
   0.25, ~3.2x) is the M3 headline self-chosen result. It is on the
   model's *realized* distribution, not on a cherry-picked subset; the
   model itself does not realize more classes under this condition.
2. **Stop sweeping temperature as a diversity lever.** T=0.7 is
   already enough to reject the "sampling noise" explanation;
   increasing T further would start degrading instruction-following
   before it broadened the attractor, because the concentration is
   not a sharpness issue.
3. **Move to M4** (causal patching + SAE feature case studies at
   turn-4 pre-answer L29-L31) using the existing n=80 scale-up
   collection.
4. **Optional side experiment, not a blocker for M4:** run exactly
   one 12B greedy collection with a modified self-chosen prompt
   (e.g., drop the candidate list, or insert "pick a less obvious
   animal") to characterize whether the attractor is prompt-fragile
   or a robust 12B prior. This is scientifically interesting on its
   own and will inform the scale-up (Gemma 3 27B+) question of
   whether the attractor shrinks, broadens, or shifts with scale.
5. **Tag the attractor diversity as a scale-axis observable.** 4B
   collapsed to `{salmon, frog, ...}`; 12B collapses to
   `{horse, cow, elephant, dog}`. The identity and width of the
   attractor are now themselves scientific observables to track at
   27B+, not noise to suppress.

## 2026-04-24 — D-34: 12B self-chosen attractor is partly prompt-fragile; turn-4 LR LOO generalizes to new classes at ~2.7x chance

Job `7246440` ran `diagnose_selfchosen_ready.py` with the new
`--prompt-variant less_obvious` (adds the sentence "Pick a less
obvious animal from the list — avoid the most stereotypical first
choices.") on the 20-bank at T=0.0, 600 greedy attempts, 101 kept.
Reveal parse 100%, Ready parse 100%.

**Diversity result:** attractor partly moves. `dog` is fully
displaced (0 reveals). `gorilla` (102) and `kangaroo` (21) hit quota
from a prior baseline of zero. `penguin` appears once. `horse`, `cow`,
`elephant` persist (38.5% / 33.0% / 7.8%). 6 parsed classes, effective
classes 3.90 (vs 3.46 at T=0.7). 14/20 bank classes still never
realized. So the attractor is partly prompt-fragile and partly a real
12B prior.

**Decoding result:** ran `decode_turns.py --selection kept --turns 1,4
--layers all` on the 101 kept runs. Turn-4 LR LOO 0.54 @ L29
(chance 0.20, ratio **2.7x**), L27-48 LR mean 0.51. NC turn-4
0.48 @ L48, mean 0.42. Peak layer band stable at L29-L31.

Critical generalization: gorilla and kangaroo were never in any
prior 12B collection, yet the late-layer LR decodes them. The
signal drops from ~3.2x chance at 4 classes to ~2.7x chance at 5
classes but does not break. If the feature were tied to a
"farm-animal / pet" subspace unique to `{elephant,cow,dog,horse}`,
we would expect degradation on the exotic-animal mix; we see mild
degradation, not collapse.

Decision:

1. **The M3 self-chosen decodability claim now spans two class
   regimes.** Carry both numbers forward: 4-class default 3.2x
   chance (LR 0.79 @ L31) and 5-class less_obvious 2.7x chance
   (LR 0.54 @ L29). L29-L31 is the locked probe band across both.
2. **Stop chasing diversity.** Neither temperature nor one round of
   prompt rewording broke the residual 12B concentration, and the
   generalization check is done. Further prompt tourism would not
   add a scientific claim beyond what D-33 and D-34 already
   establish.
3. **M4 begins: causal patching at turn-4 L29.** Use the n=80
   default collection as the base (balanced 4 classes). Patch L29
   last-pre-answer residual from C_src runs into C_tgt runs, measure
   reveal-token flip rate and turn-4 answer-flip rate, and produce a
   4x4 source-x-target matrix. Single-layer, single-position first;
   expand if noisy. Secondary: repeat on the less_obvious collection
   for gorilla/kangaroo as a generalization check.
4. **Scale-axis observable tagged:** the default attractor identity
   and its prompt-fragility are concrete observables for the
   Gemma 3 27B+ follow-up. A 27B that is either more attractor-
   concentrated or more fragile would both be interesting; the
   "early decision vs late crystallization" question becomes easier
   to answer when class diversity is no longer a confound.

## 2026-04-25 — D-35: Single-layer L29 turn-4 pre-answer patch is null on argmax; broaden to a layer band before sweeping

Job `tq_m4_patch_turn4_12b_20260424.sh` (gpu_h MIG slice) ran the M4
phase-1 patching experiment: L29 turn-4 pre-answer residual replaced
with each (src, tgt) class pair from the n=80 default scale-up
collection (`{cow, dog, elephant, horse}`, 5 src × 5 tgt × 16 cells =
400 patched trials + 20 baselines). Greedy reveal, argmax canonical
match. Output at `runs/m4_patch_turn4_12b_default_L29.json`. Detailed
writeup in `docs/progress/M4-patch-turn4-L29-null.md`.

**Result:** off-diagonal flip rate 0% across 19 of 20 target runs. The
only "effect" is `cow->horse` at 5/25 = 20%, all on a single horse
target (`attempt_588`) whose no-patch baseline is already
non-deterministic at greedy `do_sample=False` (returns `cow` instead of
the on-disk `horse`). Diagonals: cow/dog/elephant 100%, horse 88% (same
`attempt_588` instability). The L29 single-position single-layer patch
does not flip reveals on any deterministic target.

**Interpretation:** consistent with two well-documented patterns in the
activation-patching literature (Heimersheim & Nanda 2024,
arXiv 2404.15255; "Causality != Decodability" arXiv 2510.09794):

1. **Redundancy** — class info encoded across nearby layers/positions,
   single-residual replacement washed out by the rest of the stream.
2. **Decodable != causal** — L29 holds the class signature but isn't on
   the reveal-token causal path; very-late layers write to the
   unembedding without funneling through L29.

Both interpretations leave decodability (LR 0.79 @ L31) intact while
producing a causal null. Both are scientifically interesting findings.

**Decision — phase 2a method:**

1. **Skip a bare single-position layer sweep.** A layer sweep at the
   same single pre-answer position only distinguishes interpretation
   (2) from (1) if the answer is (2). If interpretation (1) is right,
   every single-layer single-position sweep also nulls — non-diagnostic
   for the cost of 7x compute.
2. **Broaden first, then narrow** (Heimersheim & Nanda 2024 sec 4
   recommendation: "begin with broad interventions then refine to
   granular components"). Phase 2a patches the full **L27-L48 layer
   band simultaneously** at the same pre-answer position. Forces the
   entire late-layer band to be src's representation. If null, both
   interpretations of "narrow late-layer locus is causal" are ruled
   out and the probe-decodable signal is genuinely off the reveal-token
   causal path. If positive, narrow back down via a layer sub-sweep to
   find the minimal sufficient site.
3. **Add logit-difference as a continuous metric.** Heimersheim & Nanda
   2024 strongly recommend `logit[answer_a] - logit[answer_b]` over
   discrete flips: continuous, linear in residual contributions, and
   sensitive to partial effects that don't flip argmax. `patch_turn4.py`
   now captures first-step generation logits via `output_scores=True`
   and reports per-cell mean
   `logit[src_first_tok] - logit[tgt_first_tok]` with per-tgt-run
   baseline subtraction. Phase 2a runs both metrics on the same patched
   trials.
4. **Phase 2b deferred.** If 2a is also null on both metrics, expand to
   a position band (last K pre-answer tokens). Requires re-collecting
   src activations across a position window — not part of phase 2a.

**Why not multi-position now:** the saved `turn_04_activations.pt`
captures only the pre-answer token, one vector per layer. Multi-position
patching needs new activation captures (cheap if we re-collect, but a
separate engineering step). Layer-band patching is free with current
data — both `_make_patch_hook` and the saved tensor support it
unchanged.

**Side investigation flagged (non-blocking):** `attempt_588` is
non-deterministic at greedy `do_sample=False, dtype=bfloat16`. Same
prompt + kwargs produce different reveals across replays. Suspect KV
cache or attention-impl drift. Investigate before any future patching
work that depends on horse trials.

## 2026-04-25 — D-36: L27-L48 layer-band patch is also null on both argmax and logit-diff; late-layer signal is off the reveal-token causal path

Job `7260288` ran phase 2a per D-35: same n=80 default scale-up
collection, same trial design (5 src x 5 tgt x 16 cells = 400 patched
trials + 20 baselines), but patching all 22 layers from L27 through L48
simultaneously at the turn-4 pre-answer position. Output at
`runs/m4_patch_turn4_12b_default_L27-48band.json`. Detailed writeup at
`docs/progress/M4-patch-turn4-band-null.md`.

**Result:** null on both metrics.

- Argmax flip-rate matrix essentially identical to phase 1: diagonals
  100% (cow/dog/elephant) / 84% (horse, same `attempt_588` instability),
  off-diagonals 0% across 19 of 20 deterministic targets, only
  `cow->horse` 20% (same single non-deterministic target).
- Logit-diff deltas (patched - baseline) all within +/- 0.12 logits.
  Natural baseline `logit[gt] - logit[next_best]` margins span +2 to +10
  logits across cow/elephant/horse targets. So the patch moves logits
  by **1-3% of the natural inter-class separation** — clean null on
  the continuous metric.
- Logit-diff for dog targets is unreliable: dog reveals don't begin
  with the ` Dog` token at step 0 (first-token logit margin is
  *negative* for dog while parsed reveal is dog 100%), so step-0
  logit-diff captures noise on the dog row. Cow/elephant/horse rows
  remain reliable.

**Interpretation:** the L29 null (D-35) and the L27-L48 band null
together rule out both forms of "narrow late-layer locus is causal":

1. Not a single missed layer (band covers L27-L48 wholesale).
2. Not redundancy hidden across late layers (22 layers patched
   simultaneously and nothing moves).

So the M3 turn-4 LR LOO 0.79 @ L31 signal is *legible* at L29-L48 but
**off the reveal-token causal path** at the pre-answer position. The
class identity is decodable there without driving the downstream
reveal. This dissociation is the same pattern documented in
"Causality != Decodability" (arXiv 2510.09794) and Heimersheim & Nanda
2024 (arXiv 2404.15255).

**Decision — phase 2b method:**

Two structural hypotheses remain:

1. **Earlier-layer locus.** Class info is in L1-L26 at the pre-answer
   position; late layers carry it forward inertly. Tested by an
   early-band patch.
2. **Other-position locus.** The pre-answer position is not the
   bottleneck at all. Reveal-token computation pulls from other
   positions in the dialogue (the secret-choice commit point at
   turn 0, the cumulative yes/no pattern across turns 1-4, etc.).
   Single-position patching nulls at every layer in this case.

The cheapest single experiment that distinguishes these: **all-layer
single-position patch** (`--layers 1,...,48`) at the same pre-answer
position. If null, position is the bottleneck axis -> hypothesis (2)
and phase 2c expands to a position band (requires re-collecting src
activations across positions). If positive, narrow back to find the
minimal sufficient layer band -> hypothesis (1).

Phase 2b runs L1-L48 in `tq_m4_patch_turn4_12b_alllayers_20260425.sh`,
~2 min walltime on gpu_h. No new activation captures needed; existing
`turn_04_activations.pt` covers all layers.

**Methodological follow-up:** when any positive signal emerges,
upgrade the logit-diff metric to scan a few generation steps and
locate where the animal-name token first becomes argmax-favored
(currently we capture only step 0, which misses cases like dog reveals
that begin "I was thinking of a dog").

## 2026-04-25 — D-37: All-layer L1-L48 single-position patch is also null; pre-answer position is decisively not the reveal-token bottleneck

Job `7260501` ran phase 2b: same n=80 default scale-up collection,
same 16-cell trial design (5 src x 5 tgt per class, 400 patched
trials + 20 baselines), with **all 48 decoder layers patched
simultaneously** at the turn-4 pre-answer position. Output at
`runs/m4_patch_turn4_12b_default_L1-48all.json`. Detailed writeup at
`docs/progress/M4-patch-turn4-alllayers-null.md`.

**Result:** null on argmax, near-null on logit-diff.

- Off-diagonal flip rates: **0% across every cell**. The phase 1/2a
  `cow->horse` 20% effect (`attempt_588` non-determinism) disappears
  in phase 2b — patching the entire stack overwrites whatever
  stochastic state was producing it.
- Cow->cow self-patch drops from 100% (phase 1/2a) to 92% (2/25 went
  to horse). The all-layer patch is strong enough to *destabilize*
  even self-class reveals, but tips toward the model's prior (horse),
  not toward src.
- Logit-diff deltas range +-0.46, 3-4x phase 2a magnitude but
  predominantly *negative* on off-diagonals — patches push *away*
  from src class, not toward it. Horse row is the only mostly-positive
  row (max +0.23), still <10% of natural baseline class margins.
- Off-diagonal distributions show a horse/cow leakage pattern:
  `dog->cow` produces 2/25 horses, `dog->horse` produces 2/25 cows,
  `elephant->cow` produces 1/25 horses. The fallback is to the
  attractor priors (horse + cow), never to src.

**Interpretation — chain over phases 1/2a/2b:**

| phase | layers patched | flip rate | logit-diff range |
|---|---|---|---|
| 1 | L29 only | 0% (sans `attempt_588`) | not measured |
| 2a | L27-L48 (22) | 0% (sans `attempt_588`) | +-0.12 |
| 2b | **L1-L48 (48)** | **0%** | +-0.46 (mostly neg) |

Tripling the layer scope produces only noise increase, never a push
toward src. **The turn-4 pre-answer position is decisively NOT in
the reveal-token causal path, regardless of layer scope.** The class
is *legible* there (M3 turn-4 LR LOO 0.79 @ L31, 3.2x chance) without
*driving* the reveal. The reveal must compute via attention to other
token positions in the dialogue prefix.

This rules out hypothesis (1) from D-36 (earlier-layer locus) and
confirms hypothesis (2) (other-position locus).

**Decision — phase 2c method:**

The cheapest informative test uses existing saved activations. Each
attempt directory in the n=80 collection has `turn_01_activations.pt`
through `turn_04_activations.pt` already on TSUBAME — saved residuals
at the pre-answer position of *every* turn, all layers. So we can
sweep across turns at zero capture cost.

**Phase 2c-i (cheap, no new captures):** all-layer single-position
patch at turn-1, turn-2, turn-3 pre-answer positions in turn. Three
runs of `patch_turn4.py` with a new `--turn N` flag selecting which
turn's saved activations to load and which tgt position to compute.
~6 min total walltime in one bundled job script (one model load).

Possible outcomes:

- One turn flips reveals: the bottleneck is at that turn's
  pre-answer position. We've localized class commitment to that
  point in the dialogue. Narrow further by layer band.
- All three turns null too: no single-position patch on any
  pre-answer position is sufficient. Class info is either at
  positions we haven't tested (turn-0 Ready output, turn-4 question
  text positions) OR encoded redundantly across positions and
  requires a position-band patch. Phase 2c-ii (position-band with
  on-the-fly src capture) becomes the next escalation.

**Methodological follow-up still deferred:** the first-step
logit-diff metric remains unreliable for dog. Will fix when any
positive signal emerges (and metric sensitivity becomes the
bottleneck rather than position scope).

## 2026-04-26 — D-38: Per-turn pre-answer patch sweep is null at every turn; no single-position patch flips reveals on stable targets

Job `7260593` ran phase 2c-i: all-layer (L1-L48) single-position
patching at turn 1, turn 2, turn 3 pre-answer positions on the same
n=80 default scale-up collection (same 16-cell trial design as
phases 1/2a/2b). Total 1200 new patched trials + 60 baselines.
Outputs at `runs/m4_patch_turn{1,2,3}_12b_default_L1-48all.json`;
detailed writeup at `docs/progress/M4-patch-turnsweep-null.md`.

**Result — comprehensive null on stable targets.**

Across the entire sweep, only one target run is baseline-non-deterministic:
`attempt_588` (horse class, baseline always returns `cow`). The other
19 target runs reproduce their on-disk reveals exactly under all four
turn baselines.

Counting non-tgt-preserving outputs on the 19 *stable* targets only
(excluding `attempt_588`):

| turn | non-tgt outputs / 300 trials | flip-to-src |
|---|---:|---:|
| 1 | 2 | 0 |
| 2 | 4 | 0 |
| 3 | 1 | 1* |
| 4 | 3 | 0 |

*The lone Turn-3 flip-to-src (`horse→cow into attempt_581 → horse`)
is part of `attempt_581`'s 15% perturbation-fallback to horse: across
all 4 turns, 9/60 trials patched into `attempt_581` produce `horse`
*regardless* of src class (dog, elephant, or horse). The Turn-3 case
just happens to align src and fallback. It's attractor leakage, not a
real flip-to-src.

**Phases 1 + 2a + 2b + 2c-i jointly:** 0 / 2280 genuine flips-to-src
on stable targets across all single-position patches tested
({L29, L27-L48, L1-L48} × {turn 1, turn 2, turn 3, turn 4}
pre-answer positions).

**Interpretation.** The class-decodable signal at L29-L48 (M3 turn-4
LR LOO 0.79, 3.2x chance) is *legible* at every turn's pre-answer
position but **decisively off the reveal-token causal path** at all
of them. The reveal must compute via attention to *other* token
positions in the dialogue prefix (turn-0 Ready output, turn-N
question-text positions, earlier-turn answer-token positions, etc.),
or via a redundant distributed encoding across multiple positions.

**Decision — phase 2c-ii method:**

1. **Position-band patch** is the structural test that follows from
   the comprehensive single-position null. Replace src's activations
   across a *window* of positions in tgt's context, all layers, and
   see if reveals flip.
2. **Implementation work required.** The current script loads saved
   single-position activations from disk. A position-band patch needs
   a "live src capture" path: for each src run, build the
   up-to-qN-preanswer context, forward-pass with
   `output_hidden_states=True`, and grab activations across a window.
   Then patch tgt's same-shape window. Add `--position-window K` CLI.
3. **First test design.** K=5 at turn-4 pre-answer covers the chat
   template scaffolding (`<end_of_turn>\n<start_of_turn>model\n`),
   which is token-aligned across runs without per-run alignment work.
   All 48 layers patched.
4. **Escalation paths.** If K=5 nulls: (a) extend to K=20 with per-run
   question-text alignment, (b) shift the window to a different
   position locus (turn-0 Ready output, requires a new
   up-to-Ready context builder), or (c) patch multiple discontiguous
   positions simultaneously.

**Held back until user weighs in.** Phase 2c-ii is a moderate refactor
(adds the live-capture path alongside the existing saved-load path)
and the structural choice between "K=5 scaffolding band at turn 4",
"shift to turn-0 Ready position", and "positional probing sweep
first to find where class signal *enters*" is a research-judgment
call worth a brief conversation. NOT submitted autonomously.

**Side observation tagged for future work:** `attempt_588` is
persistently non-deterministic across replays in a way that's
identical across all four turn baselines — so it's not transient
forward-pass nondeterminism but a reproducible divergence from the
original collection's streaming generation. Worth a short
investigation if any future result depends on horse trial reliability.

`attempt_581` is the second target whose responses are unstable, but
in a different way: its *baseline* is deterministic (always `cow`),
but it tips to `horse` ~15% of the time under heavy residual
disruption. Closer to the cow/horse decision boundary than its 4
cow-class peers in the model's representation space.

## 2026-04-26 — D-39: Positional probe sweep — there is no single class commitment locus; class emerges progressively from accumulated dialogue evidence

Job `7266216` (cpu_8, ~25 min) ran phase 2c-iii: a 4-class LR LOO + NC
LOO sweep across 12 chat-template-aligned anchor positions × all 49
layers, on residuals captured from the n=80 default scale-up (job
`7265141`). Job `7266529` re-ran to write the JSON + centroids
artifacts after a NameError-on-final-serialization bug in the first
analysis run was fixed (`samples` -> `file_class`). Numbers below are
from the 7266216 log; artifacts come from 7266529. Detailed writeup at
`docs/progress/M4-positional-probe.md`.

**Per-anchor LR LOO peaks (chance = 0.25):**

| anchor | LR peak | LR/chance | peak layer |
|---|---:|---:|---:|
| end_user_prompt    | 0.325 | 1.30x | L8 |
| end_ready          | 0.300 | 1.20x | L1 |
| end_user_q1        | 0.338 | 1.35x | L27 |
| end_model_q1       | 0.388 | 1.55x | L13 |
| end_user_q2        | 0.388 | 1.55x | L39 |
| end_model_q2       | 0.400 | 1.60x | L40 |
| end_user_q3        | 0.375 | 1.50x | L26 |
| end_model_q3       | 0.388 | 1.55x | L45 |
| end_user_q4        | 0.550 | 2.20x | L36 |
| end_model_q4       | 0.550 | 2.20x | L48 |
| end_reveal_user    | 0.537 | 2.15x | L39 |
| **pre_reveal_gen** | **0.925** | **3.70x** | L45 |

**Three-stage picture:**

1. **Pre-dialogue (end_user_prompt, end_ready):** LR <= 1.3x chance.
   *The model has not committed at the Ready position.* Most surprising
   finding: in the self-chosen condition the model emits `Ready` after
   being told to silently choose, but at that token the class is
   essentially at chance.
2. **Mid-dialogue (turns 1-3):** LR ~1.5x chance. Weak but
   above-chance, no monotone climb.
3. **End-game:** LR jumps to 2.2x at turn-4 boundaries, 3.7x at
   pre_reveal_gen. The 3.7x is *partially tautological* —
   pre_reveal_gen's residual literally drives the next-token logits
   over the four animal-name tokens, so high decodability there
   reflects "this is where the next-token prediction is computed,"
   not "this is where the commitment lives."

**M3's turn-4 pre-answer LR LOO 0.79 reconciles cleanly:** that
position sits 4 tokens after `end_user_q4` (the `\n` after
`<start_of_turn>model`), in the per-turn scaffolding gap between my
end_user_q4 (LR 0.55) and end_model_q4 (LR 0.55) anchors. Per-turn
pre-answer is a local maximum within each turn's scaffolding, but the
global pattern is "signal grows with turn index, peaks at reveal
time."

**Interpretation — the model is improvising, not retrieving.**
Combined with the comprehensive single-position patching null
(D-35/D-36/D-37/D-38: 0/2280 flips), the picture coheres:

- *No single class-commitment locus exists.* The Ready response is a
  placeholder, not a residual-stream secret-storage event.
- *Class is progressively constructed from accumulated yes/no
  constraints.* The residual becomes incrementally more
  class-informative as constraints accumulate; downstream computation
  re-derives the most-consistent class from visible history rather
  than reading off a stored variable.
- *Patching nulls now have a mechanistic explanation.* Replacing one
  position's residual doesn't change the reveal because the model
  recomputes the class from accumulated dialogue evidence at each
  subsequent step. A patched residual at, say, end_user_q4 is
  re-summarized at pre_reveal_gen from the (unmodified) yes/no answer
  history, and that re-derivation produces the original target class
  regardless.
- *The M3 turn-4 LR LOO 0.79 was real but not load-bearing.* The class
  is *legible* at turn-4 pre-answer; legibility != causal path. The
  model doesn't *consult* a stored variable at that position; it
  derives the answer at the moment of needing it.

This dissociation pattern matches Heimersheim & Nanda 2024 and
"Causality != Decodability" arXiv 2510.09794. What's new and
load-bearing here is the *positional* picture — the lack of a
Ready-position commitment, and the progressive emergence across turns.

**Decision — phase 2d (steering) needs a redesign, NOT submitted
autonomously.** Three substantively different next moves:

(a) Single-anchor centroid-difference steering at end_user_q4 (LR
    0.55), L36-L48 band, alpha sweep {0.5, 1, 2, 4, 8}. Cheapest pure
    steering test. Bets that mid-strength positional class signal
    can be amplified through the model's re-derivation step.
(b) Multi-position centroid-difference steering across all 4 turn
    boundaries simultaneously. Bets that progressive-emergence requires
    a coordinated multi-position injection to overpower the
    re-derivation.
(c) **Yes/no answer flipping** at end_model_qN. Replaces a yes/no
    residual with the residual that would have been there had the
    answer been the opposite. Different intervention type — not class
    steering but *constraint flipping*. Directly tests the
    "regenerate from accumulated evidence" hypothesis. If flipping
    one yes/no flips the reveal consistent with the new answer
    pattern, the improvisation interpretation is confirmed.

(c) probably has the highest scientific return per experiment because
confirming/refuting the improvisation story would frame the rest of
the M4 work and the eventual blog post narrative. (a) is the cheapest
pure-steering test. (b) is the escalation if (a) nulls.

Held back until user weighs in.

## 2026-04-26 — D-40: Yes/no text-level flip — improvisation hypothesis decisively confirmed

Job `7266727` (gpu_h, 114s wall, exit 0) ran phase 2d (c-text): for
each kept run T in the n=80 default scale-up (subsample 20/class), 4
flipped trials (one per turn 1..4) where the chat context is rebuilt
with that turn's yes/no answer text flipped (Yes <-> No). 320 flipped
trials + 80 baseline replays. Output at
`runs/m4_flip_yesno_text_12b_default_n80.json`. Detailed writeup at
`docs/progress/M4-flip-yesno-text.md`.

**Result — strong improvisation, decisively.**

Kept-class rate matrix (rows=class, cols=flipped turn):

| class    | T1 | T2 | T3 | T4 |
|---|---:|---:|---:|---:|
| cow      | 0%   | 35%  |  5% | 0%   |
| dog      | 0%   | 10%  |  0% | 0%   |
| elephant | 0%   | 40%  |  5% | 0%   |
| horse    | 0%   | 100% | 10% | 0%   |

- **Out-of-attractor rate 239/320 = 74.7%.** Flipping a single yes/no
  answer routinely produces reveals OUTSIDE
  `{cow, dog, elephant, horse}` — 13 of 20 bank classes appear in
  flipped trials.
- **Each turn induces a characteristic drift:** T1 -> wild/exotic
  cluster (cobra, crocodile, frog, bee). T2 -> horse fallback (the
  most-conservative cell). T3 -> dolphin in 79/80 trials. T4 -> large
  mammal swap (gorilla 42, kangaroo 24).
- **Per-turn shift rate:** T1 0/80 keep, T2 37/80, T3 4/80, T4 0/80.
  T2 is the conservative outlier because the question that lands at
  T2 across runs is more under-determined; the model defaults to the
  prior (horse) on tiebreaks.

Three baseline replay mismatches (`attempt_206`, `attempt_038`,
`attempt_049`) are the same forward-pass nondeterminism family as
`attempt_588`; ~4% noise floor that doesn't undermine the flip
results.

**Interpretation.** The model does not store a class commitment that
survives dialogue tampering. The reveal class is causally a function
of the visible yes/no answer history. Flip the history, the class
follows. The 4-class greedy attractor at 12B is a *prior*, not a
structural commitment — once the model is handed an answer pattern
the priors don't dominate, it accesses the broader 20-class bank
and confidently emits the most-consistent class.

The "self-chosen 20 questions" task at 12B is mostly a hypothetical-
completion task: model says "Ready" without committing, answers each
question consistently with some plausible class it could be thinking
of (the prior makes that the cow/dog/elephant/horse cluster), and
at reveal time derives the most-consistent class from the
accumulated answer pattern.

**Decision consequence — three substantively different next steps:**

1. **Phase 2e (residual-level constraint localization).** Capture
   per-anchor residuals under both original and flipped dialogues
   and compare divergence to localize where the dialogue evidence
   enters the residual stream. Engineering: extend
   `capture_positional_residuals.py` with `--flip-turn N`. Modest
   refactor.
2. **Scale comparison (Gemma 3 27B / 70B-class).** Re-run the same
   flip-text experiment at scale. Tests whether improvisation is
   scale-robust or whether larger models exhibit pre-commitment.
   This is the central scale question the project flagged.
3. **Pivot to M5 (SAE / transcoder feature case studies).** Now that
   the M4 narrative is solid (improvisation > storage), the
   mechanistic question of which features encode the yes/no
   constraint accumulation and the class-derivation step may be
   more efficiently studied at the SAE-feature level than the
   raw-residual level.

(2) and (3) are arguably better next steps than (1), because the
M4 narrative is now solid and additional residual-level
localization may have diminishing returns relative to
SAE-features or scale comparison.

NOT submitted autonomously. Held back for user research-judgment.

## 2026-04-26 — D-41: Logit-lens analysis confirms improvisation; no suppressed pre-commitment signature

User raised a methodological worry on D-40: flip-text-with-argmax
can't distinguish "no commitment" from "committed-but-overridable"
because in both cases the final argmax respects the dialogue. They
proposed adding logit-lens at intermediate layers to catch the
"suppressed pre-commitment" pattern (orig class elevated mid-network
then suppressed late by negative attention heads, etc.).

Job `7267006` (gpu_h, 155s wall, exit 0) re-ran the flip-text
experiment with `--logit-lens` enabled: per-trial 49-layer x 20-class
matrix of logit-lens readings at `pre_reveal_gen` (final RMSNorm +
`lm_head` applied to each layer's residual at the prefill's last
position, indexed by every bank-class first-token).

Three patterns we were watching for:
- (i) Pure improvisation: `flip[orig]` tracks `base[orig]` everywhere.
- (ii) Suppressed pre-commitment: `flip[orig]` exceeds `base[orig]`
  at some mid-network layer, then decays late.
- (iii) Concurrent consideration: orig and new both rise late; new
  overtakes orig in final layers.

**Result: pattern (iii)/(i) hybrid; pattern (ii) decisively ruled out.**

Across all 320 trials × 49 layers:
- L0-L25: `flip[orig]` ~ `base[orig]` (lens uninformative anyway).
- L30-L37: divergence kicks in (first layer where flip drops 5+ below
  base is L35-L39 across cells).
- L40-L48: full divergence; L48 drops of 17-42 logits in over-
  determined cells (T1/T3/T4), 3-10 in under-determined T2.

**Critical:** `flip[orig]` is NEVER ELEVATED above `base[orig]` at any
layer in any cell. The "commitment bump" pattern (ii) predicts and we
do not observe.

L48 suppression magnitude tracks dialogue informativeness:

| | T1 | T2 | T3 | T4 |
|---|---:|---:|---:|---:|
| cow      | -30.0 |  -9.6 | -26.9 | -41.0 |
| dog      | -27.5 | -10.1 | -23.9 | -29.7 |
| elephant | -36.9 |  -8.5 | -17.0 | -36.4 |
| horse    | -29.7 |  -3.1 | -22.3 | -42.4 |

Special case `horse/T2` (-3.1, ~0 effect at every layer) is consistent
with the kept-class rate of 100% from D-40.

**Interpretation.** The model's class-derivation at `pre_reveal_gen` is
a **late-network construction (L30-L48)** drawing from dialogue
evidence at earlier token positions via attention. There is no
earlier commitment that gets overridden — the model's preference
in the unflipped baseline is itself built by the same late-network
mechanism, just with different dialogue evidence to integrate.

D-40 conclusion holds robustly. Methodological worry resolved.

**Decision — proceed to scale comparison.** 12B M4 narrative is now
methodologically clean:

- No commitment in residual stream (D-35..D-39).
- Reveal causally driven by dialogue evidence (D-40 behavioral).
- Mechanism is late-network integration, not stored-retrieval (D-41
  lens trajectory).

Next experiment: same sequence (positional probe + flip-text-with-lens)
at Gemma 3 27B. Engineering pipeline is now reusable; only the model
name + collection step change. Need to:
1. Submit a self-chosen 20-bank n>=80 collection on `gemma-3-27b-it`
   (~hours; large model = longer per-attempt forward pass).
2. Re-run capture_positional_residuals.py + probe_positional_anchors.py
   on the 27B collection.
3. Re-run flip_yesno_text.py with --logit-lens on the 27B collection.

Open scale-axis question (per project_scale_question memory):
- Does end_ready LR LOO climb above chance at 27B?
- Do flip-text out-of-attractor rates decrease (more pre-commitment),
  increase (more confident improvisation), or stay roughly equal?
- Does lens-trajectory show a mid-network commitment bump at 27B?

Held for user to confirm: kick off 27B self-chosen collection?

## 2026-04-29 — D-42: Comparative analysis across 4 prompt variants and 27B; improvisation hypothesis robust on every axis; flip-text behavioral evidence deprioritized

**Run window:** 2026-04-26 to 2026-04-28 (jobs `7272281`, `7272415`,
`7272417`, `7272708`, `7274562`, `7274563`, `7274689`, `7274690`).
**Companion progress doc:** `docs/progress/M4-comparative-prompt-and-scale.md`.

### Methodological move: deprioritize flip-text behavioral evidence

User refined a worry about D-40: even though flipping a yes/no answer
shifts the reveal class in 74.7% of trials at 12B, this is consistent
with **either** "no commitment" **or** "committed-but-overridden plus
confabulation." A system that committed and then lost access to the
commitment under perturbed dialogue would produce the same behavioral
fingerprint as a system that never committed and is improvising. By
analogy, humans whose memory of a past event is altered will smoothly
confabulate a coherent narrative without flagging the change; this
does not show that no original memory was formed.

Flip-text behavioral metrics (kept-class rate, output entropy under
flip) are therefore **demoted to illustrative**, not load-bearing,
for the M4 conclusion. The load-bearing evidence is residual-stream
and lens-based, which test whether class commitment is **present in
the residual at end_ready** in a way a downstream layer can read:

1. Patching null at every position × layer scope tested
   (D-35 → D-38: 0/2280 trials).
2. Positional probe at chance at end_ready in 12B default and
   12B commit_strong (D-39 + this entry).
3. Cross-class baseline ~0 at L25-L30 in all 5 conditions (D-41 +
   this entry).
4. Rank check at L30: dominant rank-1 class is generic-prior-
   determined, not run-specific (D-41 + this entry).

The confabulation worry does not apply to (1)-(4) because they look
at internal state, not behavior.

### Prompt-axis robustness (4 12B variants)

The hypothesis "12B commits at end_ready" was tested under four
distinct prompt manipulations:

- `default` — baseline.
- `commit_strong` — explicit by-Ready instruction.
- `internal_locus` — externalize-via-imagined-writing
  ("ink on paper" metaphor).
- `introspection_aware` — explicit mech-interp framing referencing
  logit lens and asking the model to keep the representation focal.

All four show end_ready LR LOO ~1.2-1.4× chance, mid-layer
cross-class diff ~0 (+0.27 to +0.35 logits at L30), and the same
generic-prior dominance pattern at rank-1 (horse). None induce a
residual-stream commitment signature at end_ready.

The introspection_aware null is the most striking: a prompt that
explicitly references the experimenter's analytic frame and asks
the model to make its choice legible at intermediate layers
produces no class-specific signal at L25-L30. Either the model
lacks fine-grained introspective control over its residual stream,
or the late-network dialogue-integration mechanism dominates any
introspective effort. Neither reading supports
"commitment-is-just-prompt-fragile."

### Scale-axis robustness (12B → 27B default)

Behavioral output entropy under flip collapses 6× (0.845 → 0.137
nats); 27B's flip-text consistency is dramatically sharper. **But
the residual-stream mechanism does not change with scale.**

- Cross-class L30 diff: +0.30 logits (12B) → +0.01 (27B). Mid-network
  signal is *flatter* at scale, not sharper.
- L48 differentiation: ~5-15 logits (12B) → ~10-46 logits (27B).
  *Much* steeper at the final layer.
- Dominant rank-1 class at L30: shifts from `horse` (12B) to `tiger`
  (27B) — same generic-prior dominance pattern, different identity.

**27B improvises more decisively via sharper late-network dialogue
integration, not via earlier commitment.** The mechanism is the
same; scale sharpens the consistency engine.

### Attractor-axis robustness

The attractor identity is mutable across both prompt and scale:

- 12B default: `{cow, dog, elephant, horse}`.
- 12B less_obvious: + gorilla, kangaroo, penguin.
- 12B introspection_aware: `cow, elephant, horse` (DOG dropped).
- 27B default: `{cow, dog, elephant, gorilla, horse, tiger}`
  (+ shark at low frequency).

**The attractor identity is mutable; the commitment-mechanism is
stable.** No tested manipulation produced the qualitative shift we'd
expect if scale or prompting moved the model from improvisation to
commitment (end_ready rising above chance, patching becoming causal,
lens trajectory showing a mid-network commitment plateau).

### Caveats

- 27B formal probe-anchors LOO hit walltime at 3h30m on cpu_8 and
  was not retried; the lens-based cross-class baseline is the
  substitute (strictly weaker but directionally consistent).
- 12B internal_locus and 12B introspection_aware have only the
  lens-based proxy; default and commit_strong have full formal
  probe-anchors data.

### Decision consequence — what's left for M4

**One last prompt-variant probe before declaring M4 done:**
the `lipsum_filler` variant (Vogel et al. 2026 "Latent
Introspection"). Inserts a long block of placeholder text between
the choice instruction and the Ready cue, giving the model
additional prefill compute substrate before end_ready is read.
In Vogel et al., matched lipsum filler with a vague-mechanism
introspection question achieved 84% balanced accuracy on
Qwen2.5-Coder-32B — higher than explicit pro-introspection framing
alone. If lipsum_filler shifts end_ready probe-decodability above
chance at 12B, the "no commitment" reading needs updating to "no
commitment without compute substrate". If it nulls, we have a fifth
axis of robustness.

After lipsum_filler, the natural next move is M5: SAE / transcoder
feature case studies on the late-network dialogue-integration step,
where the mechanism is strong enough for feature-level dissection
to be the right resolution.



