"""Card model (SPEC §4.1).

Card *type* t in [0, 52]: t = suit*13 + (rank-1) for standard cards, 52 = printed joker.
Physical card id c in [0, deck_size): unique per physical card. Deck copy d holds ids
d*52 .. d*52+51 (type = id % 52); jokers take the ids after all standard cards.
"""

from __future__ import annotations

import numpy as np

from .config import Variant

SUITS = ("S", "H", "D", "C")
RANK_CODES = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
NUM_STD = 52
JOKER = 52
NUM_TYPES = 53


def make_type(suit: int, rank: int) -> int:
    return suit * 13 + rank - 1


def suit_of(t: int) -> int:
    return t // 13


def rank_of(t: int) -> int:
    return t % 13 + 1


def is_joker(t: int) -> bool:
    return t == JOKER


def base_points(t: int) -> int:
    """R7.5 face points, ignoring wild status."""
    if t == JOKER:
        return 0
    r = rank_of(t)
    return 10 if r in (1, 11, 12, 13) else r


def card_points(t: int, wild_rank: int | None = None) -> int:
    """R7.5: pass wild_rank only for a post-original holder (cards of that rank score 0)."""
    if t == JOKER:
        return 0
    if wild_rank is not None and rank_of(t) == wild_rank:
        return 0
    return base_points(t)


def wild_types(wild_rank: int | None) -> tuple[int, ...]:
    if wild_rank is None:
        return ()
    return tuple(make_type(s, wild_rank) for s in range(4))


def type_code(t: int) -> str:
    if t == JOKER:
        return "JK"
    return RANK_CODES[rank_of(t) - 1] + SUITS[suit_of(t)]


def parse_type(code: str) -> int:
    code = code.strip().upper()
    if code == "JK":
        return JOKER
    rank, suit = code[:-1], code[-1]
    return make_type(SUITS.index(suit), RANK_CODES.index(rank) + 1)


def parse_types(codes: str) -> list[int]:
    """'5H 6H 7H JK' -> [types]"""
    return [parse_type(c) for c in codes.split()]


def build_deck(variant: Variant) -> np.ndarray:
    """Type of every physical card id."""
    std = np.tile(np.arange(NUM_STD, dtype=np.int16), variant.decks)
    jokers = np.full(variant.jokers, JOKER, dtype=np.int16)
    return np.concatenate([std, jokers])


def counts_of(types) -> np.ndarray:
    c = np.zeros(NUM_TYPES, dtype=np.int16)
    for t in types:
        c[t] += 1
    return c
