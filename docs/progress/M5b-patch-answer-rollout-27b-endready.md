# M5b — Answer-rollout patch at 27B end_ready

**Status:** **DONE (2026-05-28).** Job 7779768 on gpu_1; 1024 patched
trials + 32 baselines; 616s wall (model load ~6min). Result: **null on
both axes, modulo one isolated tiger→shark trial.** End_ready L14-L18
patch reroutes neither the model's regenerated answers nor the reveal,
confirming the M5 "improvisation is scale-robust" headline against the
stronger answer-rollout protocol.

## Headline

| metric | value |
|---|---|
| trials | 1024 (5 src/class × 5 tgt/class, 7 classes; shark has 2 kept tgts) |
| baseline answer-flip non-determinism | 0/32 runs |
| baseline reveal drift | 0/32 runs |
| kept-target rate | 1007/1024 = **98.3%** |
| off-diagonal flip-to-source | **2/870** (0.23%) |
| off-diagonal answer-flips | **3/3480** answer slots (0.086%) |
| answer-flips-delta-mean across all cells | 0.000 everywhere except tiger→shark (+0.300) |

The aggregate is indistinguishable from a non-causal null: 99.8% of
off-diagonal trials hold target class, and 99.91% of regenerated answer
slots match the unperturbed manifest.

## The one trial that moved

A single off-diagonal cell shows movement:

- **tiger source `attempt_009` → shark target `attempt_431`**: reveal
  flipped to `tiger`, with **3/4** regenerated answers different from
  the manifest target answers. The patch propagated into answer
  behavior AND reached the reveal — exactly the mechanism the rollout
  protocol was designed to detect.

But it's a single trial out of 1024 and the structure of the
neighborhood argues against generalisation:

- attempt_431 is **not** broadly fragile. Patching it with cow/dog/
  elephant/gorilla/horse sources (5 each, 25 trials) and with shark
  self-patches (2) leaves it at shark every time. Only **tiger**
  sources move it — and only 1 of 5 tiger sources at that.
- The other shark target (`attempt_592`) is rock-solid: 0/32 trials
  flipped under any source.
- The matched control cell `cow→horse` also shows 1/25 (cow source
  `attempt_561` into horse `attempt_586`) flip-to-source with **0**
  answer flips, suggesting a baseline reveal-only fragility rather
  than patch-driven answer rerouting.

Net: tiger/attempt_009 → shark/attempt_431 is a class-specific 1-trial
hit, not the kind of broad fragility attempt_593 exhibited in M5
(which split 50/50 horse/cow under any source). It's interesting
enough to flag but too sparse to call signal.

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

## Escalation decision

The pre-registered chain (L12-L20 → L1-L62 full → multi-anchor +
end_model_q1..q4) was queued for the case of a strict null on both
axes. The aggregate is effectively null: 99.91% of answer slots
unchanged, 99.77% of off-diagonal reveals unchanged. The one
tiger→shark/attempt_431 hit is not "this layer is causal"; it is
one source-target pair whose entire neighborhood is otherwise null.

**Not auto-escalating.** The right narrow follow-up is to widen the
source pool for tiger-source patches into attempt_431 (single tgt ×
20 src ≈ 20 trials, cheap) to see whether attempt_009 is
idiosyncratic or tiger sources broadly dislodge attempt_431. That
answers signal-vs-noise without spending another 10-min/1024-trial
sweep — and the answer-flip-delta of 0 across all other 48 cells
already says the broader layer band is not load-bearing.

## Reading against M5

This *strengthens* the M5 headline. The prior M5 end_ready patch
null could be dismissed as an artifact of teacher-forcing the
target's visible (Q, A) history. Answer-rollout removes that
confound: under per-turn answer regeneration the model still
overwhelmingly re-derives the target class, and the patched
residual at end_ready L14-L18 fails to bias its yes/no answer
head on 99.91% of answer slots. The decodable class direction at
end_ready (LR LOO 0.508 @ L16) is epiphenomenal for the model's
downstream behavior, not just for the reveal token.

## Multi-anchor escalation (2026-05-28)

Job 7779896: same band L14-L18 and same answer-rollout protocol, but
anchors = `end_ready + end_model_q1..q4` (every "model-side" position
in the dialogue). Hypothesis: if the model refreshes its class state
at each model-turn residual, single-anchor end_ready patches get
overwritten by the next answer's computation — patching all five
positions installs source-class state continuously across the
dialogue. Skipped full-residual L1-L62: replacing the entire running
residual at one position is off-manifold (Heimersheim & Nanda flag
this), so a positive there would be uninterpretable.

Result: **virtually identical to single-anchor**.

| metric | single (end_ready) | multi (+end_model_q1..4) |
|---|---:|---:|
| kept-target | 1007/1024 (98.3%) | 1007/1024 (98.3%) |
| off-diagonal flip-to-source | 2/870 (0.23%) | 3/870 (0.34%) |
| off-diagonal answer-flips | 3/3480 (0.086%) | 3/3480 (0.086%) |
| baseline non-determinism | 0/32 | 0/32 |

The one extra reveal flip is `cow → horse/attempt_593` (now 2/25
instead of 1/25), with **0 answer flips** in both extra trials —
i.e. it's the known attempt_593 reveal-fragility (M5 documented this
target as splitting ~50/50 horse/cow regardless of source class)
re-triggering under the slightly stronger intervention, not a
patch-driven answer rerouting. The tiger/attempt_009 →
shark/attempt_431 hit from the single-anchor run persists in
multi-anchor with the same `[True, False, False, True]` regenerated
answer pattern (3 of 4 flipped) — same trial, same mechanism, not
amplified.

The verdict: across two complementary protocol upgrades — (i) the
answer-rollout fix to the teacher-forced-(Q,A) confound, and (ii)
the multi-anchor extension across every model-side position —
the patch null at L14-L18 holds. There is no per-turn refresh
mechanism that multi-anchor would unmask: the signal at end_ready
alone is the same signal at end_ready + every subsequent model
anchor. M5b joins M5 in concluding that the linearly-decodable
class direction at end_ready is epiphenomenal for the model's
downstream behavior, full stop.

## Late-band multi-anchor escalation (L35-L45, 2026-05-28)

Job 7782327: same multi-anchor set and rollout protocol as 7779896,
but the layer band is moved to **L35-L45** — the depth where the
probe table peaks (`pre_answer_q4` LR 0.672 @ L38). Tests whether
the commitment is held in a late-layer state that gets refreshed at
each model turn, rather than initialized at end_ready.

Aggregate: marginally more movement, still overwhelmingly null.

| metric | L14-L18 multi | **L35-L45 multi** |
|---|---:|---:|
| kept-target | 1007/1024 (98.3%) | 1002/1024 (97.9%) |
| off-diag flip-to-source | 3/870 (0.34%) | 6/870 (0.69%) |
| off-diag answer-flips | 3/3480 (0.086%) | 6/3480 (0.17%) |
| baseline non-determinism | 0/32 | 0/32 |

The new movement is class-pair-specific and concentrated on the
known fragile target **shark/attempt_431**:

- `elephant → shark/attempt_431`: reveal flipped to **tiger** (not
  source class elephant or target shark), 3/4 answer flips,
  regenerated answers `[True, False, False, True]`.
- `tiger → shark/attempt_431`: same pattern — reveal to tiger, 3/4
  answer flips, same regenerated-answer fingerprint.
- `cow → horse`: 5/25 reveals to cow with **0 answer flips** in
  every case; all five trials hit `attempt_593` — the boundary-
  degenerate horse target M5 documented.

The shark/attempt_431 result is the interesting one: under late-band
multi-anchor patches it has a hidden **"tiger" attractor** that any
large-mammal source (tiger AND elephant) unlocks via the same
answer-flip pattern. This is a *class-pair* effect, not a general
"shark is fragile" effect — attempt_592 (the other shark target) is
0/32 under any source. Suggests late-layer multi-anchor patches do
expose a narrow off-manifold pull toward a single attractor class
for one specific target, but not broad causal control of the reveal.

## Reproduction

```
qsub -g tga-sip_arase jobs/tq_m5b_patch_27b_endready_rollout_L14-18_20260527.sh
```

Outputs:
- [runs/m5b_patch_27b_default_endready_L14-18_rollout.json](../../runs/m5b_patch_27b_default_endready_L14-18_rollout.json) (single-anchor, job 7779768)
- [runs/m5b_patch_27b_default_multi_L14-18_rollout.json](../../runs/m5b_patch_27b_default_multi_L14-18_rollout.json) (multi-anchor, job 7779896)

Code: [scripts/patch_anchor.py](../../scripts/patch_anchor.py) (`--answer-rollout`).

Multi-anchor reproduction:
```
qsub -g tga-sip_arase jobs/tq_m5b_patch_27b_multi_rollout_L14-18_20260528.sh
```
