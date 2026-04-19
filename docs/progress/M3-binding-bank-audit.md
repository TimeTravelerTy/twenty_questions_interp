# M3 — Binding bank audit / offline rescore

**Run date:** 2026-04-19 by Codex (GPT-5).
**Input artifact:** local copy of TSUBAME results
`runs/diag/binding_smoke_5_20260419.results.json`
from job `7218265`.
**Tooling:** `scripts/rescore_binding_results.py`

## Why this audit happened

Claude's 3-condition follow-up correctly identified that the latest prompt
variants still missed the calibration gate. But before treating that as a pure
Gemma-4B or prompt-structure failure, we needed to settle whether the answer
table itself was creating artificial misses.

The main disputed primary cells were:

- `frog.has_four_legs=1`
  - surface question text is **"Does it walk primarily on four legs?"**
  - for frogs, `0` is at least as defensible as `1`
- `frog.lives_primarily_in_water=1`
  - already logged as ambiguous in M1

Secondary-only disputed cells:

- `tiger.can_swim=1`
- `eagle.can_swim=0`

## Audit runs

### A. Flip only `frog.has_four_legs` to `0`

Command:

```bash
uv run python scripts/rescore_binding_results.py \
  --results-json runs/diag/binding_smoke_5_20260419.results.json \
  --override frog.has_four_legs=0
```

Result:

| condition | original primary | rescored primary |
|---|---:|---:|
| `name_paraphrase` | 27/32 = 84.4% | **29/32 = 90.6%** |
| `name_strict` | 27/32 = 84.4% | **29/32 = 90.6%** |
| `verbalized_index` | 23/32 = 71.9% | **25/32 = 78.1%** |

This is the only disputed primary cell that improves all three conditions in a
consistent direction.

### B. Flip both frog primary cells: `frog.has_four_legs=0`, `frog.lives_primarily_in_water=0`

Command:

```bash
uv run python scripts/rescore_binding_results.py \
  --results-json runs/diag/binding_smoke_5_20260419.results.json \
  --override frog.has_four_legs=0 \
  --override frog.lives_primarily_in_water=0
```

Result:

| condition | original primary | rescored primary |
|---|---:|---:|
| `name_paraphrase` | 27/32 = 84.4% | 27/32 = 84.4% |
| `name_strict` | 27/32 = 84.4% | **29/32 = 90.6%** |
| `verbalized_index` | 23/32 = 71.9% | 23/32 = 71.9% |

`frog.lives_primarily_in_water=0` is not a clean fix. It helps exactly one
`name_strict` run and hurts others. So this cell remains ambiguous and should
not be flipped casually.

## What this means

### 1. The answer table is contributing noise, but it is not the main blocker

The generous, most defensible primary override (`frog.has_four_legs=0`) raises
the best name-based conditions from **84.4%** to **90.6%**. That is real, but it
still misses the `>=95%` gate by a wide margin.

So the answer table is **partly at fault**, but it does **not** explain the main
failure mode.

### 2. `eagle.is_mammal` remains the clearest model/prompt failure

Even after the rescoring audit, the repeated `eagle -> Is it a mammal? Yes`
errors remain untouched. That is the strongest evidence that:

- the failure is not just a bad bank row;
- prompt structure still matters a lot;
- Gemma 4B is not maintaining or applying the right entity-level semantics
  reliably in this regime.

### 3. The latest Claude note overstates the bank-independent conclusion a bit

The 3-condition note's high-level conclusion still points in the right
direction, but the bank audit shows one of the primary gate questions was not
actually clean. The right read now is:

> Prompt structure matters a lot.
> The bank contributes some avoidable noise.
> But even after the cleanest bank correction we still do not have a usable
> calibration regime at 4B.

## Decision impact

- **Do not reopen D-06.**
- **Do not scale to ~2k.**
- Do **not** mutate the canonical answer table yet just to rescue this smoke.
  The disputed frog cell should be revisited in a broader bank review, not
  patched opportunistically mid-milestone.

## Next concrete step

Proceed to the mechanistic test of **H-persistence**:

1. Re-run `verbalized_index` on the same 4 candidates × 2 seeds.
2. Capture two pre-generation states:
   - end of turn 1, immediately after the model has verbalized the secret name;
   - end of turn 2, immediately before the normal `Ready`.
3. Compare whether simple attribute/readout signal survives from turn 1 to turn 2.

If turn-1 states look entity-consistent and turn-2 states do not, the project
has a clean mechanistic story: retrieval succeeds, persistence fails.
