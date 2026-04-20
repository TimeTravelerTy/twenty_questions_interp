"""Load and validate the candidate bank, question bank, and A(c, q) table.

The feasible-set `S_t = {c in C : A(c, q_i) = a_i for all i <= t}` is computed
here from day 1 even though M2 has no questions yet; it is the central control
for the scientific claim (see docs/PLAN.md section 4).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml

from .config import ANIMALS_YAML, ANSWERS_CSV, QUESTIONS_YAML

Polarity = Literal["positive", "negative"]


@dataclass(frozen=True)
class Candidate:
    id: str
    display: str
    aliases: tuple[str, ...]
    notes: str


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    attribute: str
    polarity: Polarity


@dataclass(frozen=True)
class Bank:
    candidates: tuple[Candidate, ...]
    questions: tuple[Question, ...]
    # answers[candidate_id][question_id] in {0, 1}.
    answers: dict[str, dict[str, int]]

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(c.id for c in self.candidates)

    @property
    def question_ids(self) -> tuple[str, ...]:
        return tuple(q.id for q in self.questions)

    def answer(self, candidate_id: str, question_id: str) -> int:
        return self.answers[candidate_id][question_id]


def _load_candidates(path: Path) -> tuple[Candidate, ...]:
    with path.open() as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a list of candidates, got {type(raw).__name__}")
    out: list[Candidate] = []
    seen: set[str] = set()
    for entry in raw:
        cid = entry["id"]
        if cid in seen:
            raise ValueError(f"{path}: duplicate candidate id {cid!r}")
        seen.add(cid)
        out.append(
            Candidate(
                id=cid,
                display=entry["display"],
                aliases=tuple(entry.get("aliases", []) or []),
                notes=entry.get("notes", ""),
            )
        )
    return tuple(out)


def _load_questions(path: Path) -> tuple[Question, ...]:
    with path.open() as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a list of questions, got {type(raw).__name__}")
    out: list[Question] = []
    seen: set[str] = set()
    for entry in raw:
        qid = entry["id"]
        if qid in seen:
            raise ValueError(f"{path}: duplicate question id {qid!r}")
        seen.add(qid)
        polarity = entry.get("polarity", "positive")
        if polarity not in ("positive", "negative"):
            raise ValueError(f"{path}: question {qid!r} has bad polarity {polarity!r}")
        out.append(
            Question(
                id=qid,
                text=entry["text"],
                attribute=entry["attribute"],
                polarity=polarity,
            )
        )
    return tuple(out)


def _load_answers(
    path: Path,
    candidate_ids: tuple[str, ...],
    question_ids: tuple[str, ...],
) -> dict[str, dict[str, int]]:
    df = pd.read_csv(path)
    if df.columns[0] != "candidate":
        raise ValueError(f"{path}: first column must be 'candidate', got {df.columns[0]!r}")
    df = df.set_index("candidate")

    if tuple(df.index) != candidate_ids:
        raise ValueError(
            f"{path}: row order {tuple(df.index)} does not match animals.yaml order "
            f"{candidate_ids}"
        )
    if tuple(df.columns) != question_ids:
        raise ValueError(
            f"{path}: column order {tuple(df.columns)} does not match questions.yaml order "
            f"{question_ids}"
        )
    if df.isna().any().any():
        raise ValueError(f"{path}: contains NaNs")
    bad = ((df != 0) & (df != 1)).stack()
    if bad.any():
        raise ValueError(f"{path}: contains non-binary values at {bad[bad].index.tolist()}")
    return {cid: {qid: int(df.loc[cid, qid]) for qid in question_ids} for cid in candidate_ids}


def load_bank(
    animals_path: Path = ANIMALS_YAML,
    questions_path: Path = QUESTIONS_YAML,
    answers_path: Path = ANSWERS_CSV,
) -> Bank:
    candidates = _load_candidates(animals_path)
    questions = _load_questions(questions_path)
    answers = _load_answers(
        answers_path,
        tuple(c.id for c in candidates),
        tuple(q.id for q in questions),
    )
    return Bank(candidates=candidates, questions=questions, answers=answers)


def subset_bank(
    bank: Bank,
    candidate_ids: tuple[str, ...] | list[str] | None = None,
    question_ids: tuple[str, ...] | list[str] | None = None,
) -> Bank:
    """Return a Bank restricted to the requested candidates/questions."""
    candidate_ids = tuple(candidate_ids or bank.candidate_ids)
    question_ids = tuple(question_ids or bank.question_ids)

    candidate_lookup = {c.id: c for c in bank.candidates}
    question_lookup = {q.id: q for q in bank.questions}

    unknown_candidates = [cid for cid in candidate_ids if cid not in candidate_lookup]
    if unknown_candidates:
        raise ValueError(f"Unknown candidate ids: {unknown_candidates}")
    unknown_questions = [qid for qid in question_ids if qid not in question_lookup]
    if unknown_questions:
        raise ValueError(f"Unknown question ids: {unknown_questions}")

    candidates = tuple(candidate_lookup[cid] for cid in candidate_ids)
    questions = tuple(question_lookup[qid] for qid in question_ids)
    answers = {
        cid: {qid: bank.answers[cid][qid] for qid in question_ids}
        for cid in candidate_ids
    }
    return Bank(candidates=candidates, questions=questions, answers=answers)


def resolve_id_selector(
    raw: str | None,
    available_ids: Sequence[str],
    *,
    label: str,
) -> tuple[str, ...]:
    """Resolve a comma-separated CLI selector, with `all` as a sentinel."""
    if raw is None:
        return tuple(available_ids)

    selected = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not selected:
        raise ValueError(f"{label} selector is empty")
    if len(selected) == 1 and selected[0].lower() == "all":
        return tuple(available_ids)

    duplicates: list[str] = []
    seen: set[str] = set()
    for item in selected:
        if item in seen and item not in duplicates:
            duplicates.append(item)
        seen.add(item)
    if duplicates:
        raise ValueError(f"Duplicate {label} ids: {duplicates}")

    available = set(available_ids)
    unknown = [item for item in selected if item not in available]
    if unknown:
        raise ValueError(f"Unknown {label} ids: {unknown}")
    return selected


def feasible_set(
    bank: Bank,
    history: list[tuple[str, int]],
) -> set[str]:
    """Compute S_t = {c : A(c, q_i) = a_i for all (q_i, a_i) in history}.

    `history` is an ordered list of (question_id, answer) pairs where answer is
    0 or 1 (the post-parse boolean answer, not the raw yes/no string).
    """
    candidates = set(bank.candidate_ids)
    for qid, ans in history:
        if ans not in (0, 1):
            raise ValueError(f"feasible_set: answer must be 0 or 1, got {ans!r} for {qid!r}")
        candidates = {c for c in candidates if bank.answers[c][qid] == ans}
    return candidates
