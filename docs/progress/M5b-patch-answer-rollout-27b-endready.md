# M5b — Answer-rollout patch at 27B end_ready

**Status:** Job 7779618 submitted (2026-05-27, gpu_1 12h walltime). Result
pending.

## The gap this fixes

The M5 27B end_ready patch sweep (jobs 7654937, 7737407, 7737409) returned
0/843 off-diagonal class transfers. That null was uninformative because
`scripts/patch_anchor.py` teacher-forced the target manifest's yes/no
answers into the chat history as plain text, then patched at end_ready,
then generated only the reveal token. With four visible (Q, A) pairs
pinned in the prompt, the model can simply re-derive the target class
from text — the patched residual at end_ready never gets a chance to
re-route the model's own answer behavior.

The real question for "is end_ready load-bearing?" is whether a patched
residual there biases the model's *own* answer choices on the (fixed)
target questions, and through them the reveal. A latent commitment
might manifest via answer behavior rather than direct reveal-logit
shift.

## What changed

`scripts/patch_anchor.py` gets `--answer-rollout`. With it on, per trial:

1. Build the chat context up through user message Q_1 only.
2. Patch end_ready (re-applied during the prefill via the existing
   hook; decode steps pass through unchanged).
3. Greedy-decode the model's A_1 under the patched state.
4. Append A_1 as an assistant turn, append Q_2 as a user turn, rebuild
   context, patch end_ready again, generate A_2.
5. Repeat through A_4.
6. Append the reveal-user message, patch end_ready one more time,
   generate the reveal. Record first-token logits per class.

Each rollout step is a full prefill (KV cache discarded between steps).
The end_ready position is stable across steps because the prompt+Ready
prefix doesn't change. The hook fires on every prefill (shape[1] > 1)
and is a no-op on the actual decode steps (shape[1] == 1) — same
convention as before.

The no-patch baseline also rolls out (same flow with the hook absent)
so we can subtract baseline non-determinism from the patched
answer-flip counts.

## Metrics

Per trial:
- `regen_answers_bool` — model's regenerated A_1..A_4 (Yes/No/None).
- `target_answers_bool` — original A_1..A_4 from the target manifest.
- `answer_flip_mask` — boolean per turn: regenerated ≠ target.
- `n_answer_flips` — 0..4 per trial.
- `reveal_canonical` — parsed reveal class.
- `class_logits` — first-token logit for each realized class.

Per (src_class, tgt_class) cell:
- `flip_to_src` — fraction of trials whose reveal flipped to source class.
- `kept_tgt` — fraction whose reveal stayed at target class.
- `answer_flips_mean` — mean number of answer flips (out of 4).
- `answer_flips_baseline_mean` — baseline non-determinism subtractand.
- `answer_flips_delta_mean` — patch-attributable answer drift.
- `per_turn_flip_rate_patched` / `_baseline` — flip rate by turn index.

Both axes matter:
- **Answer-flip delta** tests whether the patch propagates into answer
  behavior at all. If zero, end_ready is not steering the model's
  yes/no head.
- **Reveal-flip-to-source** tests whether that propagated bias (if any)
  is strong enough to reach the reveal across 4 dialogue turns.

## First run

- 27B, end_ready anchor, L14-L18 (tight band around the probe peak
  L16, LR LOO 0.508 / 3.55× chance).
- Same source/target pools as the existing 27B band jobs: 5/class src,
  5/class tgt, 5 classes realized → 625 patched trials + 25 baselines.
- Job 7779618 on gpu_1, 12h walltime (~5× the non-rollout cost).
- Output: `runs/m5b_patch_27b_default_endready_L14-18_rollout.json`.

## Escalation if null on both axes

1. Widen to L12–L20 (matches existing band-sweep range).
2. Full-residual L1–L62 at end_ready.
3. Multi-anchor end_ready + end_model_q1..q4.

Only escalate if the prior layer is null on BOTH answer-flip-delta AND
reveal-flip-to-source. A nonzero answer-flip delta with null reveal
flips is itself interesting (end_ready steers the answer head but the
model re-recovers the original class) and would be worth its own
writeup.

## Reading the result

Headline matrix to look at first: `answer_flips_delta_mean`,
off-diagonal cells. Diagonal cells (self-patch) should be ~0; that's
the within-class fragility control. Off-diagonal > 0 means the patched
residual is shifting the model's regenerated answer behavior away
from what the unperturbed target run produced.

Secondary matrix: `flip_to_src`. Off-diagonal > 0 means the patch
reaches the reveal — load-bearing in the strong sense.

If both are null, the M5 headline holds and strengthens: not only does
end_ready not move the reveal, it doesn't even move the model's own
answer choices to its own questions. The decodable class direction at
end_ready is fully epiphenomenal for downstream behavior, not just for
the reveal token.
