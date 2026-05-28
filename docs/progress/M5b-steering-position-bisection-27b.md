# M5b — Steering position-bisection at 27B

**Status:** **DONE (2026-05-28).** The missing bisection is complete:
`end_ready` is truly null under rollout, `pre_answer_q1..q4` are
already steerable, and `pre_reveal_gen` is very strongly steerable.
This closes M5b as an experiment series and makes the blog-ready
claim sharper: the model does not carry a load-bearing class
commitment at setup time, but the class direction becomes causally
usable during the question-answer trajectory, well before the final
reveal token.

## Protocol

For each anchor/layer, compute class means from the 27B v2 positional
residual capture and steer with

`residual <- residual + alpha * (mu_src - mu_tgt)`

using `scripts/steer_class_direction.py --answer-rollout
--steer-scope from_anchor`. Rollout matters: the model regenerates
answers under the steering hook instead of receiving the manifest
answers as a teacher-forced transcript. `from_anchor` matters: the
prompt and earlier dialogue prefix are left untouched until the anchor
exists.

All q1-q4 runs used 576 steered trials: 7 realized classes, all
off-diagonal source-target class pairs, 5 target runs per target class,
and `alpha in {1, 3, 10}`. Baselines were clean in every run: 32/32
baseline reveals matched the manifest target and baseline answer
rollout produced 0 answer flips.

| anchor | layer | LR LOO at anchor | job | output |
|---|---:|---:|---:|---|
| end_ready | L16 | 0.508 | 7783019 | `runs/m5b_steer_27b_default_endready_L16_add_from_anchor_rollout.json` |
| pre_answer_q1 | L38 | 0.574 | 7783903 | `runs/m5b_steer_27b_default_pa1_L38_add_from_anchor_rollout.json` |
| pre_answer_q2 | L61 | 0.533 | 7783776 | `runs/m5b_steer_27b_default_pa2_L61_add_from_anchor_rollout.json` |
| pre_answer_q3 | L38 | 0.500 | 7783603 | `runs/m5b_steer_27b_default_pa3_L38_add_from_anchor_rollout.json` |
| pre_answer_q4 | L38 | 0.672 | 7783472 | `runs/m5b_steer_27b_default_pa4_L38_add_from_anchor_rollout.json` |
| pre_reveal_gen | L58 | 0.820 | 7782821 | `runs/m5b_steer_27b_default_prereveal_L58_add_from_anchor.json` |

## Reveal steering results

| anchor | layer | alpha | kept target | flip to source | drift/other | raw answer flips |
|---|---:|---:|---:|---:|---:|---:|
| end_ready | L16 | 1 | 188/192 (97.9%) | 0/192 (0.0%) | 4/192 (2.1%) | 0/768 (0.0%) |
| end_ready | L16 | 3 | 188/192 (97.9%) | 1/192 (0.5%) | 3/192 (1.6%) | 0/768 (0.0%) |
| end_ready | L16 | 10 | 175/192 (91.1%) | 2/192 (1.0%) | 15/192 (7.8%) | 3/768 (0.4%) |
| pre_answer_q1 | L38 | 1 | 89/192 (46.4%) | 32/192 (16.7%) | 71/192 (37.0%) | 58/768 (7.6%) |
| pre_answer_q1 | L38 | 3 | 23/192 (12.0%) | 68/192 (35.4%) | 101/192 (52.6%) | 99/768 (12.9%) |
| pre_answer_q1 | L38 | 10 | 0/192 (0.0%) | 80/192 (41.7%) | 112/192 (58.3%) | 181/768 (23.6%) |
| pre_answer_q2 | L61 | 1 | 107/192 (55.7%) | 17/192 (8.9%) | 68/192 (35.4%) | 26/768 (3.4%) |
| pre_answer_q2 | L61 | 3 | 38/192 (19.8%) | 51/192 (26.6%) | 103/192 (53.6%) | 12/768 (1.6%) |
| pre_answer_q2 | L61 | 10 | 2/192 (1.0%) | 118/192 (61.5%) | 72/192 (37.5%) | 0/768 (0.0%) |
| pre_answer_q3 | L38 | 1 | 66/192 (34.4%) | 33/192 (17.2%) | 93/192 (48.4%) | 40/768 (5.2%) |
| pre_answer_q3 | L38 | 3 | 24/192 (12.5%) | 72/192 (37.5%) | 96/192 (50.0%) | 42/768 (5.5%) |
| pre_answer_q3 | L38 | 10 | 0/192 (0.0%) | 109/192 (56.8%) | 83/192 (43.2%) | 25/768 (3.3%) |
| pre_answer_q4 | L38 | 1 | 17/192 (8.9%) | 92/192 (47.9%) | 83/192 (43.2%) | 42/768 (5.5%) |
| pre_answer_q4 | L38 | 3 | 0/192 (0.0%) | 135/192 (70.3%) | 57/192 (29.7%) | 42/768 (5.5%) |
| pre_answer_q4 | L38 | 10 | 1/192 (0.5%) | 100/192 (52.1%) | 91/192 (47.4%) | 10/768 (1.3%) |
| pre_reveal_gen | L58 | 1 | 16/192 (8.3%) | 123/192 (64.1%) | 53/192 (27.6%) | n/a |
| pre_reveal_gen | L58 | 3 | 0/192 (0.0%) | 154/192 (80.2%) | 38/192 (19.8%) | n/a |
| pre_reveal_gen | L58 | 10 | 0/192 (0.0%) | 144/192 (75.0%) | 48/192 (25.0%) | n/a |

The key contrast is `end_ready` vs `pre_answer_q1`. Both are early in
the dialogue, both use the same rollout steering protocol, and both
use that anchor's LR peak layer. But `end_ready` stays essentially
fixed on the target, while q1 is already perturbable. The transition
from "linearly readable but not load-bearing" to "causally usable" is
therefore between setup completion and the model's first answer.

## Why end_ready can be so null

The null is not a protocol artifact. The end_ready rollout JSON has the
same structure as the q1-q4 jobs: same model, same dtype, same alpha
sweep, same `from_anchor` scope, same 32 baselines, same 576 steered
trials, and the same rollout fields (`regen_answers_bool`,
`answer_flip_mask`, `n_answer_flips`). The old `from_anchor` bug for
late anchors does not affect end_ready because `end_ready` exists in
every partial rollout prefix.

The simplest mechanical explanation is direction scale and role:

| anchor | median direction norm | max direction norm |
|---|---:|---:|
| end_ready L16 | ~3.9 | ~9.6 |
| pre_answer_q1 L38 | ~518.5 | ~3610.4 |
| pre_answer_q4 L38 | ~1354.0 | ~8242.0 |

At 27B, end_ready has a decodable class direction (LR 0.508, 3.55x
chance), but the class-mean contrasts are tiny and steering them does
not move the model's behavior. By q1, the class direction is hundreds
of times larger and is already coupled to output behavior.

Interpretation: end_ready is a readable setup summary, not a substrate
the model reads from to maintain the secret. During the answer
trajectory, the model repeatedly attends back to the prompt/dialogue
and assembles larger class-aligned directions; those later directions
are sufficient to steer the reveal.

## Answer consistency

Raw answer flips compare regenerated answers to the target run only.
That misses the fact that many source-target animal pairs share the
same answer to a question. `scripts/analyze_steering_answer_consistency.py`
therefore compares regenerated answers against `A(src, q)` and
`A(tgt, q)` from the bank, especially on diagnostic slots where source
and target expected answers differ.

Only 22.0% of answer slots are diagnostic under this run design. On
all parsed slots, answers remain highly target-compatible even when the
reveal moves. On diagnostic slots:

| anchor | layer | alpha | source-consistent | target-consistent |
|---|---:|---:|---:|---:|
| end_ready | L16 | 1 | 14.8% | 85.2% |
| end_ready | L16 | 3 | 14.8% | 85.2% |
| end_ready | L16 | 10 | 16.0% | 84.0% |
| pre_answer_q1 | L38 | 1 | 45.0% | 55.0% |
| pre_answer_q1 | L38 | 3 | 59.8% | 40.2% |
| pre_answer_q1 | L38 | 10 | 49.6% | 50.4% |
| pre_answer_q2 | L61 | 1 | 30.2% | 69.8% |
| pre_answer_q2 | L61 | 3 | 32.5% | 67.5% |
| pre_answer_q2 | L61 | 10 | 27.2% | 72.8% |
| pre_answer_q3 | L38 | 1 | 38.5% | 61.5% |
| pre_answer_q3 | L38 | 3 | 39.6% | 60.4% |
| pre_answer_q3 | L38 | 10 | 54.3% | 45.7% |
| pre_answer_q4 | L38 | 1 | 35.5% | 64.5% |
| pre_answer_q4 | L38 | 3 | 35.5% | 64.5% |
| pre_answer_q4 | L38 | 10 | 26.1% | 73.9% |

This says reveal steering is not the same as making a fully coherent
source-animal rollout. In many cells the answer policy remains closer
to the target even when the final reveal flips to the source. That is
especially clear at q2 α=10: reveal flips to source 61.5% of the time,
while raw answer flips are 0/768 and diagnostic answers remain 72.8%
target-consistent.

So the result is not "the intervention rewrites the whole hidden game
state." It is narrower and more useful: once the model is in the
question-answer trajectory, class-aligned residual directions can
steer the reveal readout, but the dialogue-answer policy is only
partially coupled to that steered reveal state.

## Blog conclusion

M5b is done. The blog draft can now say:

1. 27B makes class identity linearly readable at setup time
   (`end_ready` L16), where 12B was weak.
2. But direct rollout steering at that setup-time direction is
   essentially null.
3. The first answer position is already steerable, and later answer
   positions become stronger/cleaner steering handles.
4. The final pre-reveal position is strongly steerable, as expected for
   a readout-assembly state.
5. Therefore the model's "commitment" is not a stable setup-time object
   carried forward. It is re-derived/assembled during the trajectory,
   becoming increasingly causal as the model approaches reveal.

No additional M5b GPU experiments are needed before drafting the blog.
