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
