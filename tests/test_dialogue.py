from pathlib import Path

import torch

from twenty_q.banks import Question, load_bank
from twenty_q.dialogue import (
    _history_to_chat_turns,
    collect_question_turns,
    parse_ready,
    parse_reveal_to_canonical,
    parse_yes_no,
)
from twenty_q.manifest import TurnRecord
from twenty_q.prompts import RenderedPrompt


def test_parse_ready_is_strict():
    assert parse_ready("Ready")
    assert parse_ready(" ready. ")
    assert not parse_ready("Ready, I picked tiger")
    assert not parse_ready("I am ready")


def test_parse_yes_no_is_strict():
    assert parse_yes_no("Yes") is True
    assert parse_yes_no(" no. ") is False
    assert parse_yes_no("Yes, it is") is None
    assert parse_yes_no("Maybe") is None


def test_parse_reveal_to_canonical_prefers_longest_alias():
    bank = load_bank()
    assert parse_reveal_to_canonical("I used an african elephant.", bank) == "elephant"
    assert parse_reveal_to_canonical("secret: cat", bank) == "cat"
    assert parse_reveal_to_canonical("secret: dragon", bank) is None


def test_history_to_chat_turns_replays_dialogue():
    turns = [
        TurnRecord(
            question_id="is_mammal",
            question_text="Is it a mammal?",
            raw_model_output="Yes",
            answer_bool=True,
        ),
        TurnRecord(
            question_id="can_fly",
            question_text="Can it fly?",
            raw_model_output="No",
            answer_bool=False,
        ),
    ]
    extra_turns = _history_to_chat_turns("Ready", turns)
    assert extra_turns == [
        {"role": "assistant", "content": "Ready"},
        {"role": "user", "content": "Is it a mammal?\nReply with only one word: Yes or No"},
        {"role": "assistant", "content": "Yes"},
        {"role": "user", "content": "Can it fly?\nReply with only one word: Yes or No"},
        {"role": "assistant", "content": "No"},
    ]


def test_collect_question_turns_records_turns_and_paths(monkeypatch, tmp_path: Path):
    questions = [
        Question(id="is_mammal", text="Is it a mammal?", attribute="mammal", polarity="positive"),
        Question(id="can_fly", text="Can it fly?", attribute="flies", polarity="positive"),
    ]
    calls: list[tuple[str, list[str]]] = []

    def fake_capture_question_state(handle, rendered, ready_output, turns, question_text):
        calls.append((question_text, [t.raw_model_output for t in turns]))
        if question_text == "Is it a mammal?":
            return torch.ones(3, 4), "Yes"
        return torch.zeros(3, 4), "No"

    monkeypatch.setattr(
        "twenty_q.dialogue.capture_question_state",
        fake_capture_question_state,
    )

    turns, turn_paths = collect_question_turns(
        handle=None,
        rendered=RenderedPrompt(system="system", user="user"),
        ready_output="Ready",
        questions=questions,
        run_dir=tmp_path,
    )

    assert calls == [
        ("Is it a mammal?", []),
        ("Can it fly?", ["Yes"]),
    ]
    assert [turn.question_id for turn in turns] == ["is_mammal", "can_fly"]
    assert [turn.answer_bool for turn in turns] == [True, False]
    assert set(turn_paths) == {1, 2}
    for path in turn_paths.values():
        assert Path(path).exists()
