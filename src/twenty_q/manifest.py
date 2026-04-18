"""Structured run manifest. Every run writes one of these as JSON next to its
activation files; comparing runs later depends on having this metadata
(DECISIONS.md D-10).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class TurnRecord(BaseModel):
    question_id: str
    question_text: str
    raw_model_output: str
    answer_bool: bool | None  # None if parse failed


class RunManifest(BaseModel):
    run_id: str
    condition: Literal["calibration", "self_chosen"]

    # Model + tokenizer identity.
    model_name: str
    model_revision: str
    tokenizer_revision: str
    torch_dtype: str
    device: str

    # Prompt + decoding.
    prompt_template_id: str
    seed: int
    decoding_params: dict = Field(default_factory=dict)

    # Candidate display.
    permutation: list[str]  # displayed order of canonical candidate ids
    secret_canonical_id: str | None = None  # calibration only
    secret_displayed_index: int | None = None  # calibration only, 1-based

    # Dialogue.
    ready_raw_output: str | None = None
    ready_parse_ok: bool | None = None
    turns: list[TurnRecord] = Field(default_factory=list)
    end_of_game_reveal: str | None = None
    reveal_canonical_id: str | None = None  # post-parsed against aliases

    # Activations — layer index -> absolute or repo-relative path.
    activation_paths: dict[int, str] = Field(default_factory=dict)
    # Question-turn activations — turn index (1-based) -> path to the all-layer
    # tensor captured immediately before the answer at that turn.
    turn_activation_paths: dict[int, str] = Field(default_factory=dict)
    hidden_size: int | None = None

    # Bookkeeping.
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    notes: str = ""

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.model_dump(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> RunManifest:
        with path.open() as f:
            return cls.model_validate(json.load(f))
