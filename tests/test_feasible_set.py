from twenty_q.banks import feasible_set, load_bank


def test_empty_history_returns_full_bank():
    bank = load_bank()
    assert feasible_set(bank, []) == set(bank.candidate_ids)


def test_single_constraint_is_mammal_yes():
    bank = load_bank()
    result = feasible_set(bank, [("is_mammal", 1)])
    expected = {"tiger", "elephant", "kangaroo", "bat", "dolphin", "gorilla",
                "cow", "dog", "cat", "horse"}
    assert result == expected


def test_two_constraints_narrow_further():
    bank = load_bank()
    # Mammal AND larger than human: tiger, elephant, kangaroo, dolphin, gorilla, cow, horse
    result = feasible_set(bank, [("is_mammal", 1), ("is_larger_than_human", 1)])
    assert "tiger" in result
    assert "elephant" in result
    assert "bat" not in result  # mammal but smaller
    assert "eagle" not in result  # not mammal
    # Narrow further: add "does it regularly swim" = 0 => excludes dolphin, cow, horse
    result2 = feasible_set(
        bank, [("is_mammal", 1), ("is_larger_than_human", 1), ("can_swim", 0)]
    )
    assert "tiger" not in result2  # tiger swims
    assert "dolphin" not in result2
    assert "kangaroo" in result2
    assert "gorilla" in result2


def test_contradictory_history_yields_empty():
    bank = load_bank()
    # Nothing is both a mammal and a bird.
    assert feasible_set(bank, [("is_mammal", 1), ("is_bird", 1)]) == set()


def test_hand_computed_fixture_tiger_only():
    # A history that should isolate tiger: mammal + carnivore + larger-than-human + striped.
    bank = load_bank()
    history = [
        ("is_mammal", 1),
        ("is_carnivore", 1),
        ("is_larger_than_human", 1),
        ("has_stripes", 1),
    ]
    assert feasible_set(bank, history) == {"tiger"}
