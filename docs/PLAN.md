# Scientific Plan — 20-Questions Mechanistic Interpretability

> Source of truth for the science. If this conflicts with code, the code is wrong.
> Operational state, next steps, and handoff notes live in `STATUS.md`.

## 1) What the project should claim

In a 20-Questions-style game, after privately choosing a real-world secret from a
known candidate bank, the model maintains a pre-answer latent state that encodes
either

- the exact chosen candidate, or
- a stable answer-sufficient semantic bundle for that candidate,

and that latent state causally influences the yes/no answer.

That framing is strong but not brittle. The project still succeeds even if the model
does not keep a perfect "TIGER variable" all the way through, as long as it keeps a
stable hidden semantic representation that actually drives answers.

## 2) Formal task definition

Let:

- `C = {c_1, …, c_N}` be a fixed candidate bank of real concepts. For the blog MVP,
  one bank of 20 real-world animals.
- `Q = {q_1, …, q_M}` be a bank of binary yes/no predicates.
- `A(c, q) ∈ {0, 1}` be a manually verified concept–question answer table.

A dialogue is:

1. The model sees the candidate bank.
2. It chooses one candidate privately.
3. It outputs only `Ready`.
4. Then for turns `t = 1, …, T`, it receives a question `q_t` and outputs Yes or No.

Public dialogue history at turn `t`: `H_t = {(q_1, a_1), …, (q_t, a_t)}`.

Hidden variables of interest:

- `Z_t`: latent pointer/identity of the secret candidate.
- `B_t`: latent semantic bundle of the secret (e.g., `[animal, feline, striped, wild, Asia]`).

The question is whether `Z_t` and/or `B_t` are stable and causal.

## 3) The two-condition design

### A. Calibration condition — not the headline

Same candidate bank, but instead of self-choice:

> "Your secret is candidate 7. Keep it fixed and secret. Do not reveal it. When ready,
> reply only `Ready`."

Important detail: give the secret **by index, not by name**. That keeps calibration
close to self-chosen and avoids making the whole thing about re-reading the surface word.

Use this condition to train and validate readouts.

### B. Main condition — the blog's actual experiment

> "Choose one candidate from the list. Keep it fixed and secret for the whole game. Do
> not reveal it. When the choice is fixed, reply only `Ready`."

Now the specific secret is not in the dialogue text. Only the candidate bank is. The
hidden choice must be maintained internally.

## 4) What to measure

At the token position right before each answer, take hidden state `h^{(ℓ)}_t` from
layer `ℓ`.

Train frozen readouts on calibration:

- candidate decoder: `p_t(c) = softmax(W h^{(ℓ)}_t)`
- attribute decoder: `b_t = σ(U h^{(ℓ)}_t)`

Then apply them zero-shot to self-chosen.

### Core metrics

**1. Selection / commitment at Ready.** At `Ready`, define `ẑ_0 = argmax_c p_0(c)`.
Track entropy of `p_0`, top-1 margin, whether there is a clear single candidate.

**2. Latent stability over turns.** For each turn, compare `p_t` to `p_0`:

- top-1 retention: `R = (1/T) Σ_t 1[argmax p_t = ẑ_0]`
- Jensen–Shannon divergence `JSD(p_t, p_0)`
- attribute-bundle similarity between `b_t` and the attributes implied by `ẑ_0`

**3. Hidden commitment beyond the public dialogue.** From `H_t`, compute the public
feasible set `S_t = {c ∈ C : A(c, q_i) = a_i ∀ i ≤ t}`. Evaluate latent stability only
on turns where `|S_t| > 1`. This is the crucial control: when the dialogue still leaves
several candidates possible, does the internal state still point to one specific candidate?

**4. Causal mediation of answers.** Derive the answer predicted by the internal
candidate distribution: `â_t = Σ_c p_t(c) · A(c, q_t)`. Test whether that internal
estimate predicts the emitted yes/no answer before generation.

## 5) Causal experiments

### A. Candidate-swap patching (calibration first, clean ground truth)

Two runs with the same question history but different secrets. Patch the
candidate-carrying activations from source to target at layer `ℓ`, right before the
answer. Measure the change in the Yes–No logit gap. Evaluate only on differentiating
questions.

### B. Attribute-targeted ablation / steering

Derive directions for single attributes (`striped`, `aquatic`, `can_fly`). Ablate or
steer along each; check that only relevant questions are affected. This distinguishes
exact-identity storage from answer-sufficient semantic storage.

### C. Self-chosen transfer intervention

After localizing the candidate/attribute subspace in calibration, move to self-chosen.
Pick two self-chosen runs whose decoded choices differ; patch the identified subspace
between them on a differentiating question. If the answer shifts as predicted by the
source run's latent choice, that is strong evidence the hidden choice is causal.

## 6) Model and tooling choice

Ladder (TSUBAME A100):

- Gemma 3 1B-IT — pipeline debugging (local Mac CPU initially).
- Gemma 3 4B-IT — main blog model.
- Gemma 3 12B-IT — robustness replicate.

Gemma Scope 2 provides SAEs/transcoders across the Gemma 3 family; model cards expose
layer subsets and recommend 64k/256k width with medium L0 for non-full-circuit use.
circuit-tracer supports Gemma-3 PLTs (270M, 1B, 4B, 12B, 27B) via the NNsight backend
(still experimental, slower, less memory-efficient).

Stack:

- NNsight — activation access, intervention, gradients, general experiment control.
- SAELens — load/inspect Gemma Scope artifacts.
- circuit-tracer — attribution graphs and selected feature interventions.
- Neuronpedia — feature browsing, probes, custom vectors, graph inspection.
- Patchscopes / output-centric feature descriptions — for labeling discovered features only.

## 7) Prioritization

1. Dense readouts first.
2. Transcoder / SAE feature analysis second.
3. Circuit case studies third.
4. Verbalized feature descriptions last.

Rationale: AxBench shows simple representation-based methods like difference-in-means
beat SAEs for concept detection; SAEBench shows common proxy metrics don't track
practical usefulness; "Transcoders Beat Sparse Autoencoders" reports transcoders yield
more interpretable features than SAEs. So dense residual-space readouts and transcoders
carry the main load; SAEs help but don't define the claim.

Treat feature verbalization cautiously. Anthropic's recent introspection work says
introspective reporting is unreliable and context-dependent. Natural-language
explanations of features are annotation, not proof.

## 8) Concrete blog-sized scope

- Concept bank: 20 animals
- Question bank: 25–30 binary predicates
- Dialogue length: 8 turns
- Two question regimes:
  - ambiguity-first: first 3 questions broad, later discriminative
  - disambiguation-first: max-information questions from the start
- Calibration data: ~2,000 runs
- Self-chosen data: ~400–600 runs
- Models: 4B main, 12B replicate, 1B debug
- Circuit case studies: 3–5 only

Target figures:

1. Layer × turn heatmap of candidate stability in self-chosen runs.
2. Same plot restricted to turns where `|S_t| > 1`.
3. One causal patching figure on a differentiating question.
4. One small circuit / feature case study: question → latent → yes/no.

## 9) Optional sanity check

After the dialogue ends, ask once: "Reveal the candidate you had been using." Use only
to estimate how often the frozen decoder's `ẑ_0` matches the model's eventual reveal.
Not a main comparison; not used during the dialogue.

## 10) Expected results

Best guess:

- The model will often show a clear hidden commitment at Ready.
- Exact candidate identity will be decodable for at least some early and mid turns.
- On some runs, exact identity will drift.
- The semantic attribute bundle will likely be more stable than exact identity.
- The cleanest causal story will probably be at the attribute-bundle level, not a
  single "secret concept neuron."
