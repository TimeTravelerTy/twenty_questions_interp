from twenty_q.banks import load_bank
from twenty_q.permutations import shuffle_candidates


def test_deterministic_for_same_seed():
    bank = load_bank()
    a = shuffle_candidates(bank.candidate_ids, seed=42)
    b = shuffle_candidates(bank.candidate_ids, seed=42)
    assert a.order == b.order


def test_different_seeds_different_orders():
    bank = load_bank()
    a = shuffle_candidates(bank.candidate_ids, seed=1)
    b = shuffle_candidates(bank.candidate_ids, seed=2)
    # Not a hard correctness property but a regression check: should differ for small banks.
    assert a.order != b.order


def test_permutation_is_a_bijection():
    bank = load_bank()
    p = shuffle_candidates(bank.candidate_ids, seed=7)
    assert set(p.order) == set(bank.candidate_ids)
    assert len(p.order) == len(bank.candidate_ids)


def test_displayed_index_roundtrip():
    bank = load_bank()
    p = shuffle_candidates(bank.candidate_ids, seed=7)
    for cid in bank.candidate_ids:
        idx = p.displayed_index(cid)
        assert p.canonical_at(idx) == cid
