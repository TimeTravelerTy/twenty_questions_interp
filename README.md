# twenty_questions_interp

Mechanistic interpretability of a 20-Questions-style game with LLMs.

**Question.** When a model privately chooses a secret from a candidate bank and
answers yes/no questions about it, does it maintain a stable latent state
(either exact identity or an answer-sufficient attribute bundle) that causally
drives its answers?

## Start here

1. [STATUS.md](STATUS.md) — live state + next concrete step. Always read first.
2. [docs/PLAN.md](docs/PLAN.md) — scientific plan (source of truth for the science).
3. [docs/DECISIONS.md](docs/DECISIONS.md) — why things are the way they are.
4. [docs/progress/](docs/progress/) — what was done at each milestone, and surprises.

## Setup

```bash
uv sync
uv run pytest
```

## Handoff convention

This project is worked on by alternating agents (Claude / Codex). Before writing
any code, read `STATUS.md`. Before ending a session, update `Next concrete step`
and append to `docs/DECISIONS.md` if a non-obvious choice was made. Close each
milestone with a note in `docs/progress/`.
