# M4 phase 2c-iii — Positional probe: there is no single commitment locus; the class crystallizes only at reveal time

**Run date:** 2026-04-26 by Claude (Opus 4.7).
**Source collection:** `runs/diag/selfchosen_ready_20bank_12b_turn4scale_20260421/`
(M3 default-prompt n=80 scale-up; balanced 4 classes — `cow`, `dog`,
`elephant`, `horse`).
**Capture script:** `scripts/capture_positional_residuals.py` at
commit `f15bcb7`.
**Probe script:** `scripts/probe_positional_anchors.py` at commit
`0c764eb` (after a streaming refactor + a NameError fix; see history).
**Capture job:** `7265141` (gpu_h, ~50s).
**Analysis job:** `7266216` (cpu_8, ~25 min) — the LR LOO + NC LOO
numbers were produced cleanly; the JSON-serialization step crashed on
a leftover variable name. Job `7266529` re-runs to write the JSON +
centroids artifacts. The grid numbers below are read from the
analysis log.
**Machine-readable report:** `runs/m4_positional_probe_12b_default_n80.json`
(after job 7266529).
**Centroid tensors:** `runs/m4_positional_probe_12b_default_n80_centroids.pt`
(steering-vector ingredients for phase 2d).

## Scope

Per D-38, single-position pre-answer patching produced 0/2280 genuine
flips on stable targets across {L29, L27-L48, L1-L48} × {turn 1, 2, 3, 4}
pre-answer positions. Two interpretations remained: an early-layer
locus we hadn't tested, or a position-distributed encoding. Before
running another patching experiment, this phase asks where the class
signal *actually* enters the residual stream by sweeping a per-position
linear probe across structural anchors.

For each kept run in the n=80 collection (600 total parsed-reveal
attempts; balanced subsample 20/class for the LR LOO probing),
captured residuals at 12 chat-template-aligned anchor positions
(seq_len=381 tokens for every run, uniform):

  end_user_prompt    end of the combined system+user opening
  end_ready          end of the model's "Ready" turn
  end_user_qN        end of turn N's user question      (N=1..4)
  end_model_qN       end of turn N's model yes/no answer (N=1..4)
  end_reveal_user    end of the reveal user message
  pre_reveal_gen     the very last position (just before reveal generation)

All 49 layers (1 embedding + 48 decoder blocks) probed at each anchor
via 4-class LR LOO and NC LOO. Centroids computed over **all** 600 runs
of each class (cow=230, dog=39, elephant=108, horse=223) for use as
phase-2d steering-vector ingredients. Chance = 0.25 on the balanced
LOO subset.

## Result — the class emerges progressively, not at any fixed locus

Per-anchor LR LOO peaks across all 49 layers:

| anchor | LR peak | LR/chance | LR peak layer | NC peak | NC/chance | NC peak layer |
|---|---:|---:|---:|---:|---:|---:|
| end_user_prompt    | 0.325 | 1.30× | L8  | 0.350 | 1.40× | L11 |
| end_ready          | 0.300 | 1.20× | L1  | 0.375 | 1.50× | L17 |
| end_user_q1        | 0.338 | 1.35× | L27 | 0.450 | 1.80× | L19 |
| end_model_q1       | 0.388 | 1.55× | L13 | 0.475 | 1.90× | L16 |
| end_user_q2        | 0.388 | 1.55× | L39 | 0.375 | 1.50× | L11 |
| end_model_q2       | 0.400 | 1.60× | L40 | 0.425 | 1.70× | L10 |
| end_user_q3        | 0.375 | 1.50× | L26 | 0.438 | 1.75× | L12 |
| end_model_q3       | 0.388 | 1.55× | L45 | 0.412 | 1.65× | L15 |
| end_user_q4        | **0.550** | **2.20×** | L36 | 0.487 | 1.95× | L39 |
| end_model_q4       | **0.550** | **2.20×** | L48 | 0.525 | 2.10× | L19 |
| end_reveal_user    | **0.537** | **2.15×** | L39 | 0.450 | 1.80× | L44 |
| pre_reveal_gen     | **0.925** | **3.70×** | L45 | 0.912 | 3.65× | L43 |

Three-stage structural picture:

1. **Pre-dialogue (end_user_prompt, end_ready):** LR ≤ 1.3× chance —
   essentially no class signal. **The model has not committed at the
   Ready position.** This is the most surprising finding: in the
   self-chosen condition the model emits `Ready` after being told to
   "silently choose an animal," but at that position the class is at
   chance.
2. **Mid-dialogue (turns 1-3):** LR ~1.5× chance. The class signal is
   weak but consistently above chance. It does not climb steadily
   turn-by-turn; turns 1-3 cluster at similar LR.
3. **End-game (turn 4 + reveal):** LR jumps to ~2.2× at turn-4
   boundaries and to **3.7× at pre_reveal_gen** — the position right
   before the model emits the animal name token.

## Reconciling with M3's turn-4 pre-answer LR 0.79

The M3 scale-up reported turn-4 pre-answer LR LOO 0.79 @ L31 (3.2×
chance). That position is *not* one of the 12 anchors here — it sits
*between* `end_user_q4` (LR 0.55) and `end_model_q4` (LR 0.55), 4
tokens after `end_user_q4` (the `\n` after `<start_of_turn>model`).
The M3 measurement and these new numbers are consistent: the residual
4 tokens *into* the turn-4 model-prompt scaffolding holds more class
info than at the boundaries flanking it, plausibly because that
position is where the model is "summarizing the answer state" before
emitting a yes/no.

In other words: the per-turn pre-answer position is a local maximum
within each turn's scaffolding, but the global pattern is still
"signal grows with turn index, peaking at reveal time."

## The 0.925 at `pre_reveal_gen` is partially tautological

Important caveat. `pre_reveal_gen` is the very last token before the
model emits the reveal animal name — its residual is what literally
drives the next-token logits over `Cow`/`Dog`/`Elephant`/`Horse`. If
the model is going to output "cow" next, the L45 residual at this
position *must* be class-distinguishing in a roughly 1-to-1 way with
the upcoming reveal. So 0.925 at `pre_reveal_gen` doesn't tell us
"where the commitment lives" — it tells us "where the next-token
prediction is computed," which is a tautology of the architecture.

The interesting numbers are the *non-tautological* anchors: the class
signal at `end_ready` is at chance, climbs slowly through mid-dialogue,
and reaches 2.2× chance at turn-4 boundaries. None of these earlier
positions exhibit the step-change you'd expect if the class were a
single committed variable.

## Interpretation — the model is improvising, not retrieving

Combined with the comprehensive single-position patching null
(D-35/D-36/D-37/D-38: 0/2280 flips), this picture coheres:

1. **There is no single class-commitment locus to patch.** At the
   Ready token (where the prompt instructs the model to commit), the
   class is at chance. The Ready response is a placeholder that lets
   the dialogue start — not an act of secret-storage in the residual
   stream.
2. **The class is progressively constructed from accumulated dialogue
   evidence.** Each turn adds a yes/no constraint; the residual
   becomes incrementally more class-informative as the constraints
   accumulate. This is consistent with a "regenerate-from-context"
   mechanism: at any point in the dialogue, the model's downstream
   computation can re-derive the most-consistent class from the
   visible history rather than reading off a stored commitment.
3. **Patching nulls now have a mechanistic explanation.** Replacing
   one position's residual doesn't change the reveal because the
   model recomputes the class from accumulated dialogue evidence at
   each subsequent step. A patched residual at, say, `end_user_q4` is
   re-summarized at `pre_reveal_gen` from the (unmodified) yes/no
   answer history — and that re-derivation produces the original
   target class regardless.
4. **The M3 turn-4 LR LOO 0.79 was real but not load-bearing.** The
   class is *legible* at turn-4 pre-answer, but legibility ≠ causal
   path. The model doesn't *consult* a stored variable at that
   position; it derives the answer at the moment of needing it.

This dissociation pattern (decodable but causally inert) is well-
documented in the activation-patching literature
([Heimersheim & Nanda 2024](https://arxiv.org/abs/2404.15255);
["Causality ≠ Decodability"](https://arxiv.org/html/2510.09794)). What's
new and load-bearing here is the *positional* picture: the lack of
a Ready-position commitment, and the progressive-emergence pattern
across turns.

## Decision consequence — phase 2d (steering) needs a redesign

The original plan was: "find the highest-signal anchor, patch only
along the class-discriminating direction (centroid difference) at
that anchor with α-scaling." But the findings change what makes
sense:

1. **Steering at `end_ready`** would not work — there's nothing class-
   coded there to amplify, and adding `(src_centroid - tgt_centroid)`
   along a noise direction probably won't budge anything.
2. **Steering at `pre_reveal_gen`** would technically work but is
   uninteresting — that residual is one step from the reveal token
   logits, so any intervention there essentially overwrites the
   model's output decision rather than steering its mechanism.
3. **Steering at moderate-signal anchors** (`end_user_q4`,
   `end_model_q4`, `end_reveal_user`; LR ~2.2× chance, layers in the
   L36-L48 band) is the structurally interesting move. Whether the
   centroid-difference steering at α-scale K is enough to push the
   class through the model's re-derivation step is exactly the
   question phase 2d should answer.
4. **Multi-position steering across the whole dialogue prefix** is
   the natural escalation if single-position steering nulls. If the
   model re-derives the class from accumulated evidence, a steering
   signal applied across many turns might overpower the dialogue
   evidence whereas a single-position injection would not.
5. **Different intervention type — perturbing dialogue evidence,
   not class:** instead of injecting class direction, flip a yes/no
   answer's residual at `end_model_qN` and see if the reveal updates.
   This tests whether the model genuinely uses the visible answer
   history to derive the reveal class. If yes, that confirms the
   "improvisation from evidence" story directly.

## Decision deferred to user

Three substantively different next moves with different research
implications. Recommend:

(a) **Single-anchor centroid-difference steering at `end_user_q4`**
    L36-L48 band, α sweep over {0.5, 1, 2, 4, 8} — direct test of
    whether class-direction steering moves reveals at the
    mid-strength anchor.
(b) **Multi-position centroid-difference steering** across all 4
    turn boundaries — tests whether the model's re-derivation can
    be overpowered by a coordinated multi-anchor injection.
(c) **Yes/no answer flipping** at `end_model_qN` — tests the
    "regenerate from accumulated evidence" hypothesis directly. If
    flipping the residual at one yes/no answer flips the reveal
    consistent with the new answer pattern, the improvisation
    interpretation is confirmed.

(c) is structurally different from (a)/(b): it's a constraint-flip
test, not a steering test. It probably has the highest scientific
return per experiment, because confirming/refuting the improvisation
story would frame the rest of the M4 work and the eventual blog post
narrative. (a) is the cheapest pure-steering test. (b) is the
escalation if (a) nulls.

## Artifacts

- Capture: `runs/positional_residuals/12b_default_n80/` on TSUBAME
  (600 × 4.5 MB ~= 5 GB). Per-run `attempt_NNN.pt` files with
  `(K_anchors=12, n_layers=49, hidden=3840)` residual tensors plus
  anchor metadata.
- Probe report (after job 7266529): `runs/m4_positional_probe_12b_default_n80.json`.
- Centroids (after job 7266529): `runs/m4_positional_probe_12b_default_n80_centroids.pt`
  with shape `(12, 49, 4, 3840)` per-(anchor, layer, class) — direct
  ingredients for phase 2d steering vectors.
- Reference: Heimersheim & Nanda 2024, *How to use and interpret
  activation patching* (arXiv 2404.15255); "Causality ≠ Decodability"
  (arXiv 2510.09794).
