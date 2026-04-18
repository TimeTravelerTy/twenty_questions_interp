"""Randomize the displayed order of candidates per run; log the permutation.

Without this, later decoders can entangle concept identity, displayed index,
and positional biases in the model's hidden choice (DECISIONS.md D-07).
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Permutation:
    """Maps between canonical candidate IDs and their displayed 1-indexed position.

    `order` is the displayed order of canonical IDs (position 1 = order[0]).
    """

    order: tuple[str, ...]

    def displayed_index(self, canonical_id: str) -> int:
        """Return the 1-based displayed index of a canonical candidate id."""
        return self.order.index(canonical_id) + 1

    def canonical_at(self, displayed_index: int) -> str:
        """Return the canonical id at a 1-based displayed position."""
        if not 1 <= displayed_index <= len(self.order):
            raise IndexError(
                f"displayed_index {displayed_index} out of range 1..{len(self.order)}"
            )
        return self.order[displayed_index - 1]


def shuffle_candidates(canonical_ids: tuple[str, ...], seed: int) -> Permutation:
    """Return a seeded permutation of `canonical_ids`.

    Determinism: the same seed always yields the same permutation for a given
    input order, so replays are reproducible.
    """
    rng = random.Random(seed)
    order = list(canonical_ids)
    rng.shuffle(order)
    return Permutation(order=tuple(order))
