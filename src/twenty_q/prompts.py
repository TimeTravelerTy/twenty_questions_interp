"""Prompt templates for the calibration and self-chosen conditions.

Calibration secrets are assigned by **1-based displayed index**, not by name
(DECISIONS.md D-06) — this keeps calibration distributionally close to
self-chosen and avoids building a decoder that keys on the literal surface
token.

Each template returns `(system, user)` strings. The caller is responsible for
running the tokenizer's chat template over them.
"""
from __future__ import annotations

from dataclasses import dataclass

from .permutations import Permutation

# Bump PROMPT_TEMPLATE_ID whenever any string below changes. Every RunManifest
# pins the ID it was built against.
PROMPT_TEMPLATE_ID = "v1-2026-04-18"


def _candidate_list(perm: Permutation, display_names: dict[str, str]) -> str:
    lines = []
    for i, canonical_id in enumerate(perm.order, start=1):
        lines.append(f"{i}. {display_names[canonical_id]}")
    return "\n".join(lines)


@dataclass(frozen=True)
class RenderedPrompt:
    system: str
    user: str
    template_id: str = PROMPT_TEMPLATE_ID


def calibration_prompt(
    perm: Permutation,
    display_names: dict[str, str],
    secret_displayed_index: int,
) -> RenderedPrompt:
    """Build the index-based calibration prompt. `secret_displayed_index` is 1-based."""
    if not 1 <= secret_displayed_index <= len(perm.order):
        raise ValueError(
            f"secret_displayed_index {secret_displayed_index} out of range "
            f"1..{len(perm.order)}"
        )
    system = (
        "You are playing a 20-Questions-style game. An adversary will ask you "
        "yes/no questions about a single secret animal that you are holding in "
        "mind. You must answer each question truthfully and concisely with "
        "either 'Yes' or 'No' about the secret animal. Do not reveal the "
        "identity of the secret animal under any circumstances."
    )
    user = (
        "Here is the list of candidate animals:\n\n"
        f"{_candidate_list(perm, display_names)}\n\n"
        f"Your secret is the animal shown at position #{secret_displayed_index} "
        "in the list above. Keep that choice fixed and secret for the entire "
        "game. Do not state the name of the animal. When your choice is locked "
        "in, reply with only the single word: Ready"
    )
    return RenderedPrompt(system=system, user=user)


def self_chosen_prompt(
    perm: Permutation,
    display_names: dict[str, str],
) -> RenderedPrompt:
    """Build the self-chosen prompt. The secret is not supplied."""
    system = (
        "You are playing a 20-Questions-style game. An adversary will ask you "
        "yes/no questions about a single secret animal that you are holding in "
        "mind. You must answer each question truthfully and concisely with "
        "either 'Yes' or 'No' about the secret animal. Do not reveal the "
        "identity of the secret animal under any circumstances."
    )
    user = (
        "Here is the list of candidate animals:\n\n"
        f"{_candidate_list(perm, display_names)}\n\n"
        "Choose exactly one animal from the list above and commit to it as your "
        "secret for the entire game. Keep the choice fixed; do not change it. "
        "Do not state the name of the animal. When your choice is locked in, "
        "reply with only the single word: Ready"
    )
    return RenderedPrompt(system=system, user=user)


# Follow-up turn used only at the end of self-chosen smoke runs (M2.2b).
REVEAL_USER_MESSAGE = (
    "The game is now over. Please reveal the single animal you had been using "
    "as your secret. Reply with only the name of that animal."
)
