import pytest

from twenty_q.banks import load_bank, resolve_id_selector, subset_bank


def test_bank_shape():
    bank = load_bank()
    assert len(bank.candidates) == 20
    assert len(bank.questions) >= 25
    for c in bank.candidates:
        row = bank.answers[c.id]
        assert set(row.keys()) == set(bank.question_ids)
        for v in row.values():
            assert v in (0, 1)


def test_candidate_ids_unique_and_ordered():
    bank = load_bank()
    ids = [c.id for c in bank.candidates]
    assert len(ids) == len(set(ids))
    assert bank.candidate_ids == tuple(ids)


def test_answer_accessor():
    bank = load_bank()
    # Tiger is a mammal.
    assert bank.answer("tiger", "is_mammal") == 1
    # Eagle is not a mammal.
    assert bank.answer("eagle", "is_mammal") == 0


def test_subset_bank_preserves_requested_order_and_answers():
    bank = load_bank()
    sub = subset_bank(
        bank,
        candidate_ids=("salmon", "frog", "eagle"),
        question_ids=("is_bird", "lives_primarily_in_water"),
    )

    assert sub.candidate_ids == ("salmon", "frog", "eagle")
    assert sub.question_ids == ("is_bird", "lives_primarily_in_water")
    assert sub.answer("salmon", "is_bird") == 0
    assert sub.answer("frog", "lives_primarily_in_water") == 1


def test_resolve_id_selector_supports_all_sentinel():
    bank = load_bank()

    assert resolve_id_selector("all", bank.candidate_ids, label="candidate") == bank.candidate_ids
    assert resolve_id_selector("ALL", bank.question_ids, label="question") == bank.question_ids


def test_resolve_id_selector_preserves_order():
    bank = load_bank()

    resolved = resolve_id_selector(
        "salmon,frog,eagle", bank.candidate_ids, label="candidate"
    )

    assert resolved == ("salmon", "frog", "eagle")


def test_resolve_id_selector_rejects_duplicates_and_unknown_ids():
    bank = load_bank()

    with pytest.raises(ValueError, match="Duplicate candidate ids"):
        resolve_id_selector("salmon,frog,salmon", bank.candidate_ids, label="candidate")

    with pytest.raises(ValueError, match="Unknown question ids"):
        resolve_id_selector("dragon", bank.question_ids, label="question")
