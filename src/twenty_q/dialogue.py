"""Run a single dialogue end-to-end with Ready-state and optional question turns.

The base capture is still the M2 Ready-state activation at the token position
immediately before `Ready` is generated (DECISIONS.md D-08). M3 extends this by
optionally capturing all-layer pre-answer activations on later yes/no question
turns, persisting them in the run directory, and recording parsed answers in
the manifest.

Uses `transformers` directly rather than NNsight. NNsight is overkill for pure
activation capture; it will come in at M3+ when we start doing interventions
(DECISIONS.md D-15). Swap out `capture_ready_state` when that happens.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .banks import Bank, Question
from .manifest import RunManifest, TurnRecord
from .permutations import Permutation
from .prompts import (
    REVEAL_USER_MESSAGE,
    RenderedPrompt,
    calibration_prompt,
    question_turn_prompt,
    self_chosen_prompt,
)


@dataclass
class ModelHandle:
    """Carries a loaded model + tokenizer plus metadata for manifests."""

    model: Any
    tokenizer: Any
    model_name: str
    model_revision: str
    tokenizer_revision: str
    torch_dtype: str
    device: str


def load_model(
    model_name: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> ModelHandle:
    """Load a HF causal LM + tokenizer. Gemma 3 is gated; HF_TOKEN must be set."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device,
    )
    model.eval()
    cfg = model.config
    revision = getattr(cfg, "_commit_hash", None) or "unknown"
    dtype_str = str(dtype).replace("torch.", "")
    return ModelHandle(
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        model_revision=revision,
        tokenizer_revision=getattr(tokenizer, "_commit_hash", None) or "unknown",
        torch_dtype=dtype_str,
        device=device,
    )


def _build_chat_input_ids(
    handle: ModelHandle,
    rendered: RenderedPrompt,
    extra_turns: list[dict[str, str]] | None = None,
) -> dict[str, torch.Tensor]:
    """Apply the tokenizer's chat template with an assistant-turn prefix.

    Some models (Gemma 3 included) do not accept a 'system' role — fold the
    system message into the first user turn.
    """
    combined_user = rendered.system.strip() + "\n\n" + rendered.user.strip()
    messages = [{"role": "user", "content": combined_user}]
    if extra_turns:
        messages.extend(extra_turns)
    out = handle.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    if hasattr(out, "to"):
        out = out.to(handle.model.device)
    elif isinstance(out, dict):
        out = {k: v.to(handle.model.device) for k, v in out.items()}
    else:
        out = {"input_ids": out.to(handle.model.device)}
    return out


@torch.no_grad()
def capture_ready_state(
    handle: ModelHandle,
    rendered: RenderedPrompt,
) -> tuple[torch.Tensor, str]:
    """Return (per-layer hidden states at the Ready position, raw model output).

    Hidden-state tensor shape: `(n_layers + 1, hidden_size)` — index 0 is the
    embedding output, index ℓ (ℓ >= 1) is the output of transformer block ℓ.
    """
    model_inputs = _build_chat_input_ids(handle, rendered)
    input_ids = model_inputs["input_ids"]

    # One forward pass to grab hidden states at the last input position.
    outputs = handle.model(
        **model_inputs,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )
    # Each element is (1, seq_len, hidden_size).
    last_pos_states = torch.stack(
        [h[:, -1, :].squeeze(0).float().cpu() for h in outputs.hidden_states], dim=0
    )  # (n_layers + 1, hidden_size)

    # Generate the actual "Ready" token(s). Short + greedy for determinism.
    gen = handle.model.generate(
        **model_inputs,
        max_new_tokens=8,
        do_sample=False,
        pad_token_id=handle.tokenizer.eos_token_id,
    )
    new_tokens = gen[0, input_ids.shape[1]:]
    raw = handle.tokenizer.decode(new_tokens, skip_special_tokens=True)
    return last_pos_states, raw


# Strict: the generated chunk must be exactly the word "ready" (case-insensitive),
# allowing only whitespace and terminal punctuation. "Ready, I chose tiger" must
# NOT pass — that would be a covert leak of the secret via the Ready token.
_READY_RE = re.compile(r"^\s*ready\s*[.!]?\s*$", re.IGNORECASE)


def parse_ready(raw: str) -> bool:
    """Return True iff the model's output is a bare 'ready' (no trailing content)."""
    return bool(_READY_RE.match(raw))


@torch.no_grad()
def elicit_reveal(
    handle: ModelHandle,
    rendered: RenderedPrompt,
    ready_output: str,
) -> str:
    """Issue the end-of-game reveal prompt and return the raw model output."""
    extra = [
        {"role": "assistant", "content": ready_output.strip()},
        {"role": "user", "content": REVEAL_USER_MESSAGE},
    ]
    model_inputs = _build_chat_input_ids(handle, rendered, extra_turns=extra)
    input_ids = model_inputs["input_ids"]
    gen = handle.model.generate(
        **model_inputs,
        max_new_tokens=48,
        do_sample=False,
        pad_token_id=handle.tokenizer.eos_token_id,
    )
    new_tokens = gen[0, input_ids.shape[1]:]
    return handle.tokenizer.decode(new_tokens, skip_special_tokens=True)


def parse_reveal_to_canonical(raw_reveal: str, bank: Bank) -> str | None:
    """Greedy longest-alias match. Returns the canonical candidate id, or None."""
    text = raw_reveal.lower()
    best: tuple[int, str] | None = None  # (match length, canonical id)
    for c in bank.candidates:
        for alias in (c.display, *c.aliases):
            alias_l = alias.lower()
            if re.search(rf"\b{re.escape(alias_l)}\b", text) and (
                best is None or len(alias_l) > best[0]
            ):
                best = (len(alias_l), c.id)
    return best[1] if best else None


_YES_RE = re.compile(r"^\s*yes\s*[.!]?\s*$", re.IGNORECASE)
_NO_RE = re.compile(r"^\s*no\s*[.!]?\s*$", re.IGNORECASE)


def parse_yes_no(raw: str) -> bool | None:
    """Return True for Yes, False for No, or None if the output is malformed."""
    if _YES_RE.match(raw):
        return True
    if _NO_RE.match(raw):
        return False
    return None


def _history_to_chat_turns(
    ready_output: str,
    turns: list[TurnRecord],
) -> list[dict[str, str]]:
    """Render the assistant/user turns after the initial Ready response."""
    extra_turns = [{"role": "assistant", "content": ready_output.strip()}]
    for turn in turns:
        extra_turns.append({"role": "user", "content": question_turn_prompt(turn.question_text)})
        extra_turns.append({"role": "assistant", "content": turn.raw_model_output.strip()})
    return extra_turns


@torch.no_grad()
def capture_question_state(
    handle: ModelHandle,
    rendered: RenderedPrompt,
    ready_output: str,
    turns: list[TurnRecord],
    question_text: str,
    answer_max_new_tokens: int = 8,
) -> tuple[torch.Tensor, str]:
    """Capture all-layer activations at the token position right before an answer."""
    extra_turns = _history_to_chat_turns(ready_output, turns)
    extra_turns.append({"role": "user", "content": question_turn_prompt(question_text)})
    model_inputs = _build_chat_input_ids(handle, rendered, extra_turns=extra_turns)
    input_ids = model_inputs["input_ids"]

    outputs = handle.model(
        **model_inputs,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )
    last_pos_states = torch.stack(
        [h[:, -1, :].squeeze(0).float().cpu() for h in outputs.hidden_states], dim=0
    )

    gen = handle.model.generate(
        **model_inputs,
        max_new_tokens=answer_max_new_tokens,
        do_sample=False,
        pad_token_id=handle.tokenizer.eos_token_id,
    )
    new_tokens = gen[0, input_ids.shape[1]:]
    raw = handle.tokenizer.decode(new_tokens, skip_special_tokens=True)
    return last_pos_states, raw


def collect_question_turns(
    handle: ModelHandle,
    rendered: RenderedPrompt,
    ready_output: str,
    questions: list[Question],
    run_dir: Path | None = None,
) -> tuple[list[TurnRecord], dict[int, str]]:
    """Run a fixed question sequence and optionally persist per-turn activations."""
    turns: list[TurnRecord] = []
    turn_activation_paths: dict[int, str] = {}

    for turn_idx, question in enumerate(questions, start=1):
        activations, raw_answer = capture_question_state(
            handle=handle,
            rendered=rendered,
            ready_output=ready_output,
            turns=turns,
            question_text=question.text,
        )
        turns.append(
            TurnRecord(
                question_id=question.id,
                question_text=question.text,
                raw_model_output=raw_answer,
                answer_bool=parse_yes_no(raw_answer),
            )
        )
        if run_dir is not None:
            act_path = run_dir / f"turn_{turn_idx:02d}_activations.pt"
            torch.save(activations, act_path)
            turn_activation_paths[turn_idx] = str(act_path)

    return turns, turn_activation_paths


def run_calibration_dialogue(
    handle: ModelHandle,
    bank: Bank,
    secret_canonical_id: str,
    perm: Permutation,
    seed: int,
    run_id: str,
    out_dir: Path,
    questions: list[Question] | None = None,
) -> RunManifest:
    """Run one calibration dialogue and optionally continue with question turns."""
    display_names = {c.id: c.display for c in bank.candidates}
    secret_idx = perm.displayed_index(secret_canonical_id)
    rendered = calibration_prompt(perm, display_names, secret_idx)

    activations, ready_raw = capture_ready_state(handle, rendered)

    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    act_path = run_dir / "activations.pt"
    torch.save(activations, act_path)
    turns: list[TurnRecord] = []
    turn_activation_paths: dict[int, str] = {}
    if questions:
        turns, turn_activation_paths = collect_question_turns(
            handle=handle,
            rendered=rendered,
            ready_output=ready_raw,
            questions=questions,
            run_dir=run_dir,
        )

    manifest = RunManifest(
        run_id=run_id,
        condition="calibration",
        model_name=handle.model_name,
        model_revision=handle.model_revision,
        tokenizer_revision=handle.tokenizer_revision,
        torch_dtype=handle.torch_dtype,
        device=handle.device,
        prompt_template_id=rendered.template_id,
        seed=seed,
        decoding_params={"do_sample": False, "max_new_tokens": 8},
        permutation=list(perm.order),
        secret_canonical_id=secret_canonical_id,
        secret_displayed_index=secret_idx,
        ready_raw_output=ready_raw,
        ready_parse_ok=parse_ready(ready_raw),
        turns=turns,
        activation_paths={i: str(act_path) for i in range(activations.shape[0])},
        turn_activation_paths=turn_activation_paths,
        hidden_size=int(activations.shape[-1]),
    )
    manifest.save(run_dir / "manifest.json")
    return manifest


def run_selfchosen_dialogue(
    handle: ModelHandle,
    bank: Bank,
    perm: Permutation,
    seed: int,
    run_id: str,
    out_dir: Path,
    elicit_reveal_after: bool = True,
    questions: list[Question] | None = None,
) -> RunManifest:
    """Run one self-chosen dialogue and optionally continue with question turns."""
    display_names = {c.id: c.display for c in bank.candidates}
    rendered = self_chosen_prompt(perm, display_names)

    activations, ready_raw = capture_ready_state(handle, rendered)

    reveal_raw: str | None = None
    reveal_canonical: str | None = None
    if elicit_reveal_after:
        reveal_raw = elicit_reveal(handle, rendered, ready_raw)
        reveal_canonical = parse_reveal_to_canonical(reveal_raw, bank)

    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    act_path = run_dir / "activations.pt"
    torch.save(activations, act_path)
    turns: list[TurnRecord] = []
    turn_activation_paths: dict[int, str] = {}
    if questions:
        turns, turn_activation_paths = collect_question_turns(
            handle=handle,
            rendered=rendered,
            ready_output=ready_raw,
            questions=questions,
            run_dir=run_dir,
        )

    manifest = RunManifest(
        run_id=run_id,
        condition="self_chosen",
        model_name=handle.model_name,
        model_revision=handle.model_revision,
        tokenizer_revision=handle.tokenizer_revision,
        torch_dtype=handle.torch_dtype,
        device=handle.device,
        prompt_template_id=rendered.template_id,
        seed=seed,
        decoding_params={"do_sample": False, "max_new_tokens": 8, "reveal_tokens": 48},
        permutation=list(perm.order),
        secret_canonical_id=None,
        secret_displayed_index=None,
        ready_raw_output=ready_raw,
        ready_parse_ok=parse_ready(ready_raw),
        turns=turns,
        end_of_game_reveal=reveal_raw,
        reveal_canonical_id=reveal_canonical,
        activation_paths={i: str(act_path) for i in range(activations.shape[0])},
        turn_activation_paths=turn_activation_paths,
        hidden_size=int(activations.shape[-1]),
    )
    manifest.save(run_dir / "manifest.json")
    return manifest
