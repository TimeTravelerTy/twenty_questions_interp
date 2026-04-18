# M0 — Repo bootstrap

**Closed:** 2026-04-18 by Claude (Opus 4.7).

## What was built
- Directory skeleton (`src/twenty_q/`, `scripts/`, `data/`, `tests/`, `docs/`,
  `runs/` w/ gitignored subdirs).
- `pyproject.toml` with uv + hatchling + ruff + pytest; Python pinned to 3.11.
- Scientific plan dropped verbatim into `docs/PLAN.md`.
- `STATUS.md` seeded with M1 as current milestone and a `Next concrete step`
  pointing at `data/animals.yaml`.
- `docs/DECISIONS.md` seeded with D-01 through D-13 (bootstrap choices).
- Package stub `src/twenty_q/__init__.py` with version; one vacuous test so
  `pytest` exits 0.

## Surprises / watch-outs
- None yet. `uv sync` may take a few minutes the first time due to torch +
  transformers + nnsight.
- `runs/` is gitignored but has `.gitkeep` in calibration/ and selfchosen/ so
  downstream code can assume the directories exist.

## Next agent
- Start M1. `Next concrete step` in STATUS.md is authoritative. The first real
  decisions are *which 20 animals*: aim for pairwise distinguishability ≥3
  attributes, not obscurity.
