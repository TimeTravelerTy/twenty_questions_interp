# M3 — 3-condition calibration-binding follow-up (post-4cond)

**Run date:** 2026-04-19 by Claude (Opus 4.7).
**Model:** `google/gemma-3-4b-it` on TSUBAME H100, float32.
**Job:** `tq_m3_binding5` (`7218265`). Repo HEAD at run time: `bad9233`.
**Artifacts:** `runs/diag/binding_smoke_5_20260419/` on TSUBAME.

## Scope

- Candidates: `tiger, eagle, frog, salmon`
- Questions (5): `is_mammal, is_bird, lives_primarily_in_water, has_four_legs` as the
  **primary gate set**, plus `can_swim` as a **secondary sanity** reading.
  The prior 4cond set (`is_mammal, is_bird, can_fly, can_swim, has_feathers`) was
  redundant on this 4-candidate set: `is_bird`, `can_fly`, and `has_feathers` are
  literally the same column. The new set gives each candidate a unique fingerprint
  (1/3 or 2/2 splits) across the four primary questions.
- Seeds: `0, 1`
- Conditions (3):
  - `name_paraphrase` — "You have chosen X as your secret animal" (the prior best).
  - `name_strict` — same, plus "answer each question only about X, not about the
    average animal in the candidate list".
  - `verbalized_index` — two-turn binding. Turn 1: "Your secret is the animal at
    position #N. Reply with only the animal's name." The model then emits the
    name, and turn 2 locks it in and asks for Ready. Preserves D-06's index
    framing while forcing the model to perform one explicit name retrieval.

The run also records a within-vs-between Ready-state cosine contrast across
candidates (within-class cosine minus between-class cosine, averaged post-L13).

## Topline results

| condition | ready parse | **primary correctness** | secondary `can_swim` | post-13 within-cos | post-13 within-between contrast |
|---|---:|---:|---:|---:|---:|
| `name_paraphrase` | 8/8 | 27/32 = **84.4%** | 6/8 = 75.0% | 0.99994 | **+6.67e-04** |
| `name_strict` | 8/8 | 27/32 = **84.4%** | 4/8 = 50.0% | 0.99996 | +6.64e-04 |
| `verbalized_index` | 8/8 | 23/32 = **71.9%** | 6/8 = 75.0% | 0.99994 | +5.17e-04 |

**None of the three conditions clears the ≥95% primary-correctness gate.** The
decision point from STATUS — "stop and investigate prompt semantics before
scaling" — applies. Do not reverse D-06. Do not scale to ~2k.

For reference the prior 4cond smoke had name-based conditions at 90% on its
(softer, redundant) question set. On this harder, non-redundant primary set
the name regime is 84.4% — closer to the real ceiling for this binding regime
at 4B.

## Per-condition wrong-answer inventory

### `name_paraphrase` (27/32 primary)

```
eagle_00  is_mammal: Yes (bank No) ✗
eagle_01  is_mammal: Yes (bank No) ✗
frog_00   has_four_legs: No (bank Yes) ✗
frog_01   has_four_legs: No (bank Yes) ✗
salmon_01 is_bird: Yes (bank No) ✗
```

All 5 errors are on category/property questions. Tiger is clean.

### `name_strict` (27/32 primary)

```
eagle_00  is_mammal: Yes ✗
eagle_01  is_mammal: Yes ✗
frog_00   has_four_legs: No ✗
frog_01   has_four_legs: No, lives_primarily_in_water: No ✗✗
```

Same error signature as `name_paraphrase`. The "answer only about X" clause
did **not** help primary correctness and actively hurt secondary (`can_swim`
dropped from 75% to 50%). Interpretation: the extra instruction nudges the
model toward more confident yes/no commitments without changing what the
model believes about the entity.

### `verbalized_index` (23/32 primary) — *most informative condition*

Verbalization was always correct. Every single run produced the exact secret
name when asked at turn 1 (`match=True` in all 8 rows). Despite this, primary
correctness is **lower** than either pure-name condition:

```
tiger_00  clean ✓
tiger_01  clean ✓
eagle_00  is_mammal: Yes ✗
eagle_01  is_mammal: Yes ✗
frog_00   is_mammal: Yes, is_bird: Yes, has_four_legs: No ✗✗✗
frog_01   is_mammal: Yes, has_four_legs: No ✗✗
salmon_00 is_mammal: Yes ✗
salmon_01 is_mammal: Yes ✗
```

Every non-tiger run calls itself a mammal at least once. The model can read
the name off the index, verbalize it back correctly, and then still answer
yes/no questions as if the secret were a generic-mammal prior. The binding
does not survive the transition from the verbalization turn to the Ready
turn.

## Research reading

The cleanest read of this run is not about prompt strength; it is about
representation persistence.

### H-binding sharpens into H-persistence

Prior M3-4b-smoke-diagnostics introduced **H-binding**: "at 4B, referencing
the secret by list index at commitment time does not reliably produce an
instantiated-entity representation at Ready." This run narrows it:

> **H-persistence.** At 4B, the instantiated-entity representation does
> not reliably persist across a chat-turn boundary. Even when the model
> successfully retrieves the secret name from the index in turn 1, by the
> time it emits Ready in a subsequent turn the answer behavior has drifted
> back to candidate-list priors (here, "mammal-heavy").

This is a stronger, more testable claim than H-binding — and it is cleanly
separable from pure retrieval failure, because we now have evidence (`match=True`
on 8/8) that retrieval is not the bottleneck.

### Yes-drift is non-uniform and category-specific

Looking across all three conditions:

- `eagle.is_mammal` is wrong in **every single eagle run (6/6)**. This is a
  model-level confusion that no 4B calibration prompt we have tried can fix.
- `eagle.can_swim` is wrong in **every single eagle run (6/6)** as well (the
  model thinks eagles swim); this is a secondary question but the
  co-occurrence is striking.
- `frog.has_four_legs` is wrong in **every frog run under the name
  conditions (4/4)** and most under verbalized_index; the model treats frogs
  as legless when bound by name, and as full mammals when bound by verbalized
  index.

The errors concentrate on eagle and frog. Tiger is clean under every
condition. Salmon is mostly clean under the name conditions, badly biased
under verbalized_index. Nothing about this looks like a prompt failure —
it looks like **specific representational failures** the calibration
harness is surfacing.

### Within-vs-between contrast is an order of magnitude larger than the 4cond run, but unusable as a gate alone

Post-L13 `within − between` here is ≈ 6.5e-04 across conditions, versus
≈ 3.8e-05 reported for `name` and 7.1e-05 for `name_paraphrase` in the 4cond
run. The relative ordering still favors name_paraphrase ≈ name_strict >
verbalized_index, but the gap between conditions is small (< 4%) and does
not follow correctness. A representational separation metric is not enough
on its own to validate the binding regime — semantic correctness has to come
first.

## Decision

**Do not reverse D-06. Do not scale to ~2k calibration runs.** The binding
regime we have at 4B is not yet producing valid training examples — the
model is making category-level errors on the simplest entity-identification
questions, across every prompt strengthening we have tried.

This is the stop-and-investigate branch specified in STATUS.md step 5.

## What to do next

The research-relevant next step is not another prompt variant. It is to
localize the failure representationally. Proposed branches, in rough order
of how cheap and how informative each is:

1. **Probe eagle-is-mammal directly.** Capture the Ready-state and the
   first-answer-token hidden state for all 6 eagle runs. Train a tiny
   is_mammal probe on the other three candidates' Ready states (leave-eagle-
   out), and apply it to eagle. If the probe says "no" but the model emits
   "Yes", the decision to say "Yes" is not in the Ready-state `is_mammal`
   attribute — it is happening later in the answer turn. That directly
   informs where the scientific claim's readouts should be anchored.
2. **Bank audit under D-14.** Several failing cells look like plausible
   bank disputes, not model failures: `tiger.can_swim=1`, `eagle.can_swim=0`
   vs "eagles dive for fish", `frog.has_four_legs=1` vs "amphibians have legs
   but do they count as four legs". If the model's answers are defensible,
   the bank is wrong, and primary correctness should be recomputed with a
   revised bank before we conclude binding is broken.
3. **Mechanistic test of H-persistence.** Run verbalized_index again but
   capture the Ready state *at the end of turn 1 (after the model verbalized
   the name)* and compare its decodability to the Ready state at the end of
   turn 2 (after lock-in). If turn-1 Ready decodes the secret with a clean
   attribute probe and turn-2 Ready does not, we have a direct measurement
   that the entity representation decays across the chat boundary.
4. **Self-chosen calibration-free control.** Self-chosen is the headline
   regime anyway. Run a small self-chosen smoke at 4B and check whether its
   Ready-state answer pattern looks more like `name_paraphrase` (85%) or
   `verbalized_index` (72%) on the same question set. If self-chosen is
   closer to the name regime, that is the expected transfer story and we
   can keep calibration purely as probe-training infra even at 85%. If it
   looks more like verbalized_index, the whole probe-training pipeline is
   at risk and we revisit the model ladder (D-05).

Of these, (1) and (2) are the cheapest. (4) is the research-highest-leverage
check because it speaks directly to the blog claim. (3) is the cleanest
mechanistic test of H-persistence and should happen before we spend more
smokes on prompt variants.

Calibration remains infra (D-01). Self-chosen is still the headline.
