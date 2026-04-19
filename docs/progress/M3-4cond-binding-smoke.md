# M3 — 4-condition calibration-binding smoke

**Run date:** 2026-04-19 by Codex (GPT-5).
**Model:** `google/gemma-3-4b-it` on TSUBAME H100, float32.
**Job:** `tq_m3_binding4` (`7217900`).
**Artifacts:** `runs/diag/binding_smoke_4cond_20260419/` on TSUBAME.

## Scope

- Candidates: `tiger, eagle, frog, salmon`
- Questions: `is_mammal, is_bird, can_fly, can_swim, has_feathers`
- Seeds: `0, 1`
- Conditions:
  - `index` — current D-06 prompt
  - `index_reminder` — index + explicit per-turn position reminder
  - `name` — "Your secret animal is X"
  - `name_paraphrase` — "You have chosen X as your secret animal"

Unlike the earlier binding diagnostic, this run persisted `Ready` and per-turn
activations for all four conditions via `scripts/diagnose_index_binding.py`.

## Topline results

| condition | Ready parse | answer correctness | post-L13 within-secret cosine | post-L13 within-vs-between margin |
|---|---:|---:|---:|---:|
| `index` | 8/8 | 20/40 = **50.0%** | 0.99984 | 0.0000098 |
| `index_reminder` | 8/8 | 16/40 = **40.0%** | 0.99986 | 0.0000109 |
| `name` | 8/8 | 36/40 = **90.0%** | 0.99993 | 0.0000380 |
| `name_paraphrase` | 8/8 | 36/40 = **90.0%** | 0.99994 | 0.0000710 |

The accuracy split is decisive: both name-based conditions massively outperform
both index-based conditions. But neither name-based condition clears the
pre-specified **95%** correctness gate from `STATUS.md`, so D-06 should **not**
be reversed yet.

## Findings

### 1. D-17 is confirmed, not weakened

The original M3 smoke result was not a tiger/eagle fluke. On the 4-candidate,
5-question grid, both index-based conditions are poor:

- `index`: 50.0%
- `index_reminder`: 40.0%

The explicit position reminder hurts rather than helps, matching the earlier
qualitative result.

### 2. Name binding is better, but still misses the gate

Residual wrong answers under the two name-based conditions were:

- `name`
  - `tiger`: `can_fly -> Yes` once, `can_swim -> No` twice
  - `eagle`: `is_mammal -> Yes` once
- `name_paraphrase`
  - `tiger`: `can_swim -> No` once
  - `eagle`: `is_mammal -> Yes` twice
  - `salmon`: `is_bird -> Yes` once

`tiger/can_swim` is the known bank edge case from M1, but it does **not**
fully explain the miss:

- excluding the `tiger/can_swim` errors, `name` would be 36/38 = **94.7%**
- excluding the `tiger/can_swim` error, `name_paraphrase` would be 36/39 = **92.3%**

So there are still genuine binding/answer errors in the name regime.

### 3. Plain within-secret cosine is too saturated to use alone

The planned second gate, "within-secret Ready-state cosine", is nearly 1.0 in
every condition. That makes it weak as a standalone decision metric here.

The more useful quantity from this run is the **within-vs-between** contrast:

- `index`: `+9.8e-06`
- `index_reminder`: `+1.09e-05`
- `name`: `+3.80e-05`
- `name_paraphrase`: `+7.10e-05`

So the name conditions do show better secret-specific clustering, but the raw
cosine itself is too compressed. Future binding decisions should use either:

- within-secret minus between-secret cosine, or
- direct NC/LR decoding at `Ready`

not plain within-secret cosine by itself.

## Interpretation

This run strengthens the scientific direction of the project:

- the hidden-secret question is still the right one;
- calibration is still only infrastructure;
- index-based calibration remains the wrong training condition unless it can be
  repaired;
- self-chosen is still the headline regime.

But the result is **not** strong enough to switch calibration to name-based
and move on. The safer read is:

> Name binding is much closer to a usable calibration condition than index
> binding, but it still has unresolved semantic errors at the current prompt
> strength and question set.

## Next concrete step

Run one final small 4B binding follow-up before deciding D-06:

1. Keep the same 4 candidates and 2 seeds.
2. Use a **non-ambiguous primary score set**:
   `is_mammal, is_bird, can_fly, has_feathers`.
   Report `can_swim` separately as a secondary sanity check.
3. Compare:
   - best current name/paraphrase prompt,
   - one slightly stronger name prompt ("answer only about X, not about the average animal in the list"),
   - the earlier verbalized-index control as a reserve option.
4. Score on both:
   - answer correctness;
   - Ready-state **within-vs-between** contrast or a tiny NC/LR check.
5. If one non-index condition clears ≥95% on the primary score set, write the
   D-06 reversal/replacement explicitly in `docs/DECISIONS.md`; otherwise stop
   and investigate prompt semantics before scaling.
