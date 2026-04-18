"""Paths and model identifiers used across the project."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = REPO_ROOT / "data"
ANIMALS_YAML: Path = DATA_DIR / "animals.yaml"
QUESTIONS_YAML: Path = DATA_DIR / "questions.yaml"
ANSWERS_CSV: Path = DATA_DIR / "answers.csv"

RUNS_DIR: Path = REPO_ROOT / "runs"
CALIBRATION_RUNS_DIR: Path = RUNS_DIR / "calibration"
SELFCHOSEN_RUNS_DIR: Path = RUNS_DIR / "selfchosen"

# Model ladder (see DECISIONS.md D-05).
MODEL_DEBUG: str = "google/gemma-3-1b-it"
MODEL_MAIN: str = "google/gemma-3-4b-it"
MODEL_REPLICATE: str = "google/gemma-3-12b-it"
