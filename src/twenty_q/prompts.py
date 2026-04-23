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

# Bump PROMPT_TEMPLATE_ID whenever any string below changes, OR when the prompt
# surface grows a new template used in the dialogue (e.g. adding question_turn_
# prompt). Every RunManifest pins the ID it was built against so decoders can
# filter on it. v2: added question_turn_prompt for M3 turnful dialogues.
PROMPT_TEMPLATE_ID = "v2-2026-04-19"


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


def calibration_prompt_name_paraphrase(
    perm: Permutation,
    display_names: dict[str, str],
    secret_canonical_id: str,
) -> RenderedPrompt:
    """Build the name-based calibration prompt selected for 12B follow-up work."""
    if secret_canonical_id not in display_names:
        raise ValueError(f"Unknown secret_canonical_id {secret_canonical_id!r}")
    secret_display_name = display_names[secret_canonical_id]
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
        f"You have chosen {secret_display_name} as your secret animal. Keep using "
        "that same animal for the entire game. Do not state the name of the animal "
        "again. When your choice is locked in, reply with only the single word: Ready"
    )
    return RenderedPrompt(system=system, user=user)


SELF_CHOSEN_VARIANTS = ("default", "less_obvious")


def self_chosen_prompt(
    perm: Permutation,
    display_names: dict[str, str],
    variant: str = "default",
) -> RenderedPrompt:
    """Build the self-chosen prompt. The secret is not supplied.

    `variant="less_obvious"` adds a nudge aimed at breaking the 12B
    `{elephant, cow, dog, horse}` attractor (see D-33). The template_id is
    suffixed with the variant name when non-default so manifests record which
    wording was used.
    """
    if variant not in SELF_CHOSEN_VARIANTS:
        raise ValueError(
            f"unknown self-chosen variant {variant!r}; "
            f"expected one of {SELF_CHOSEN_VARIANTS}"
        )
    system = (
        "You are playing a 20-Questions-style game. An adversary will ask you "
        "yes/no questions about a single secret animal that you are holding in "
        "mind. You must answer each question truthfully and concisely with "
        "either 'Yes' or 'No' about the secret animal. Do not reveal the "
        "identity of the secret animal under any circumstances."
    )
    if variant == "less_obvious":
        choose_line = (
            "Choose exactly one animal from the list above and commit to it as "
            "your secret for the entire game. Pick a less obvious animal from "
            "the list — avoid the most stereotypical first choices. Keep the "
            "choice fixed; do not change it. Do not state the name of the "
            "animal. When your choice is locked in, reply with only the single "
            "word: Ready"
        )
    else:
        choose_line = (
            "Choose exactly one animal from the list above and commit to it as "
            "your secret for the entire game. Keep the choice fixed; do not "
            "change it. Do not state the name of the animal. When your choice "
            "is locked in, reply with only the single word: Ready"
        )
    user = (
        "Here is the list of candidate animals:\n\n"
        f"{_candidate_list(perm, display_names)}\n\n"
        f"{choose_line}"
    )
    template_id = (
        PROMPT_TEMPLATE_ID if variant == "default" else f"{PROMPT_TEMPLATE_ID}-{variant}"
    )
    return RenderedPrompt(system=system, user=user, template_id=template_id)


def question_turn_prompt(question_text: str) -> str:
    """Build a single yes/no question turn for the ongoing dialogue."""
    return (
        f"{question_text.strip()}\n"
        "Reply with only one word: Yes or No"
    )


# Follow-up turn used only at the end of self-chosen smoke runs (M2.2b).
REVEAL_USER_MESSAGE = (
    "The game is now over. Please reveal the single animal you had been using "
    "as your secret. Reply with only the name of that animal."
)
