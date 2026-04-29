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


SELF_CHOSEN_VARIANTS = (
    "default",
    "less_obvious",
    "commit_strong",
    "verbose_neutral",  # length-matched control for commit_strong (no commitment framing)
    "internal_locus",  # externalize-via-imagined-writing framing
    "introspection_aware",  # tells the model we will analyze its activations via logit lens
    "lipsum_filler",  # Vogel et al. 2026 matched-filler: extra prefill compute substrate before Ready
)


# Lorem ipsum filler block used by the `lipsum_filler` self-chosen variant.
# Length is roughly matched to a few hundred tokens so the prefill at end_ready
# integrates over substantially more positions than the other variants. Content
# is standard placeholder Latin; it carries no semantic relation to animals.
_LIPSUM_FILLER = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do "
    "eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim "
    "ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut "
    "aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit "
    "in voluptate velit esse cillum dolore eu fugiat nulla pariatur. "
    "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui "
    "officia deserunt mollit anim id est laborum. Sed ut perspiciatis unde "
    "omnis iste natus error sit voluptatem accusantium doloremque "
    "laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore "
    "veritatis et quasi architecto beatae vitae dicta sunt explicabo. "
    "Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut "
    "fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem "
    "sequi nesciunt. Neque porro quisquam est, qui dolorem ipsum quia dolor "
    "sit amet, consectetur, adipisci velit, sed quia non numquam eius modi "
    "tempora incidunt ut labore et dolore magnam aliquam quaerat voluptatem. "
    "Ut enim ad minima veniam, quis nostrum exercitationem ullam corporis "
    "suscipit laboriosam, nisi ut aliquid ex ea commodi consequatur. "
    "Quis autem vel eum iure reprehenderit qui in ea voluptate velit esse "
    "quam nihil molestiae consequatur, vel illum qui dolorem eum fugiat "
    "quo voluptas nulla pariatur. At vero eos et accusamus et iusto odio "
    "dignissimos ducimus qui blanditiis praesentium voluptatum deleniti "
    "atque corrupti quos dolores et quas molestias excepturi sint occaecati "
    "cupiditate non provident, similique sunt in culpa qui officia deserunt "
    "mollitia animi, id est laborum et dolorum fuga. Et harum quidem rerum "
    "facilis est et expedita distinctio."
)


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
    elif variant == "commit_strong":
        choose_line = (
            "Choose exactly one animal from the list above and commit to it as "
            "your secret for the entire game. Take a moment to mentally pick "
            "your animal — by the time you reply with 'Ready', you must have "
            "already chosen and locked in the specific animal you will use. "
            "Do not delay the choice; commit to it before responding. Once you "
            "say Ready, you cannot change your animal — it must remain the "
            "same throughout every question. Do not state the name of the "
            "animal. When your choice is locked in, reply with only the single "
            "word: Ready"
        )
    elif variant == "internal_locus":
        # Externalize-via-imagined-writing framing. Asks the model to mentally
        # commit by visualizing the act of writing the chosen animal's name on
        # a piece of paper. The "ink on paper" metaphor invokes a persistent,
        # unchangeable representation. Tests whether concrete-imagery anchoring
        # (a different kind of commitment elicitation than commit_strong's
        # direct instruction) induces a residual-stream commitment signal.
        choose_line = (
            "Choose exactly one animal from the list above and commit to it as "
            "your secret for the entire game. Imagine writing the name of your "
            "chosen animal on a piece of paper inside your mind — clearly and "
            "deliberately, letter by letter. Once you have finished writing, "
            "the choice is fixed: like ink on paper, it cannot be erased, "
            "rewritten, or replaced. The animal you wrote is the animal you "
            "will keep for every question that follows. Do not state the name "
            "of the animal. When you have finished writing, reply with only "
            "the single word: Ready"
        )
    elif variant == "introspection_aware":
        # Introspection-aware framing. Tells the model explicitly that we are
        # running a mechanistic-interpretability analysis on its hidden
        # activations and references specific techniques (logit lens, residual
        # stream at intermediate layers). Grounded in the introspection
        # literature: Lindsey et al. 2025 "On the Biology of Large Language
        # Models" and related Anthropic work showing that LLMs have some
        # introspective access to their internal states, plus the empirical
        # observation that prompts which make the experimenter's analytic
        # frame explicit can shift what gets surfaced at the final layer
        # (reduced suppression of mid-network signals).
        #
        # Tests whether explicit experimenter-frame priming induces the model
        # to make its class commitment more legible in the residual stream
        # at end_ready.
        choose_line = (
            "Choose exactly one animal from the list above and commit to it as "
            "your secret for the entire game. We are running a mechanistic-"
            "interpretability analysis on your hidden activations during this "
            "game — specifically, we will use the logit lens technique to read "
            "out which animal you chose by examining the residual stream at "
            "intermediate layers. For this analysis to succeed, your chosen "
            "animal needs to be represented as clearly and persistently as "
            "possible in your internal state. When you say Ready, please make "
            "a focal mental commitment to the specific animal you have chosen, "
            "and hold that representation stably throughout the game. Do not "
            "suppress the representation in late layers; do not distribute it "
            "across multiple candidates — keep it focal on the one animal you "
            "chose. Do not state the name of the animal in text. When your "
            "choice is locked in, reply with only the single word: Ready"
        )
    elif variant == "lipsum_filler":
        # Vogel et al. 2026 "Latent Introspection" matched-filler condition.
        # A long block of placeholder text is inserted between the choice
        # instruction and the Ready cue. The model then prefills over many
        # additional positions before the residual stream is read at
        # end_ready, giving the network more compute substrate to integrate
        # any latent class commitment.
        #
        # In Vogel et al., matched lipsum filler paired with a
        # vague-mechanism introspection question achieved 84% balanced
        # accuracy on Qwen2.5-Coder-32B — higher than explicit
        # pro-introspection framing alone. If this variant shifts
        # end_ready probe-decodability above chance at 12B, the
        # "no commitment" reading of D-42 needs updating to "no
        # commitment without compute substrate". If it nulls, we have
        # one more axis of robustness for the late-dialogue-integration
        # mechanism.
        choose_line = (
            "Choose exactly one animal from the list above and commit to it as "
            "your secret for the entire game. Take your time to settle on the "
            "specific animal. Use the following passage as space to think; "
            "do not respond to it, simply read through it as you decide:\n\n"
            f"{_LIPSUM_FILLER}\n\n"
            "Once you have settled on the specific animal, keep that choice "
            "fixed; do not change it. Do not state the name of the animal. "
            "When your choice is locked in, reply with only the single word: "
            "Ready"
        )
    elif variant == "verbose_neutral":
        # Length-matched control for commit_strong. Same approximate token count
        # but the extra text restates GAME RULES, not commitment framing.
        # If commit_strong shifts behavior but verbose_neutral does not, the
        # effect is from the commitment framing; if both shift behavior, the
        # extra prompt tokens (more compute on the choice position) are doing
        # the work.
        choose_line = (
            "Choose exactly one animal from the list above and commit to it as "
            "your secret for the entire game. Remember the rules: you will "
            "answer only with 'Yes' or 'No' — no explanations, no synonyms, no "
            "additional words. Each yes/no answer must be truthful for the "
            "animal you have chosen. Do not use the animal's name or any "
            "description that identifies it. Keep responses concise. Keep the "
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
