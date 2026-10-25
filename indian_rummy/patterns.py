"""Precomputed meld-pattern tables for the solver (SPEC §4.3).

A pattern is the natural part of a meld plus a number of wild slots:
    Pattern(nat=((type, count), ...), wilds=k, size=n, kind)
Every valid secondary meld (R5) is exactly one pattern filled with `wilds` wild cards, so the
solver never has to reason about where a joker "sits".
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from typing import NamedTuple

from .cards import NUM_STD, make_type, rank_of

MELD_SIZES = (3, 4)


class Pattern(NamedTuple):
    nat: tuple[tuple[int, int], ...]  # sorted (type, count) pairs
    wilds: int
    size: int
    kind: str  # "seq" | "set" | "trip"

    def natural_types(self) -> list[int]:
        return [t for t, c in self.nat for _ in range(c)]


def _pos_to_type(suit: int, pos: int) -> int:
    """Window position 1..14 (14 = Ace high) -> card type."""
    return make_type(suit, 1 if pos == 14 else pos)


def _as_nat(types) -> tuple[tuple[int, int], ...]:
    out: dict[int, int] = {}
    for t in types:
        out[t] = out.get(t, 0) + 1
    return tuple(sorted(out.items()))


def _build_secondary(triplets: bool) -> list[Pattern]:
    seen: dict[tuple, Pattern] = {}

    def add(types, wilds, size, kind):
        p = Pattern(_as_nat(types), wilds, size, kind)
        seen.setdefault((p.nat, p.wilds, p.size), p)

    for suit in range(4):
        for n in MELD_SIZES:
            for start in range(1, 15 - n + 1):
                window = [_pos_to_type(suit, p) for p in range(start, start + n)]
                for k in range(1, n + 1):
                    for chosen in combinations(window, k):
                        add(chosen, n - k, n, "seq")
    for rank in range(1, 14):
        suited = [make_type(s, rank) for s in range(4)]
        for n in MELD_SIZES:
            for k in range(1, n + 1):
                for chosen in combinations(suited, k):
                    add(chosen, n - k, n, "set")
    if triplets:
        for t in range(NUM_STD):
            add((t, t, t), 0, 3, "trip")
    return list(seen.values())


def _build_originals(triplets: bool) -> list[Pattern]:
    """R4.1: pure sequences of 3 or 4, and triplets (natural only)."""
    out = []
    for suit in range(4):
        for n in MELD_SIZES:
            for start in range(1, 15 - n + 1):
                window = [_pos_to_type(suit, p) for p in range(start, start + n)]
                out.append(Pattern(_as_nat(window), 0, n, "seq"))
    if triplets:
        for t in range(NUM_STD):
            out.append(Pattern(((t, 3),), 0, 3, "trip"))
    return out


@lru_cache(maxsize=None)
def secondary_patterns(triplets: bool) -> tuple[Pattern, ...]:
    return tuple(_build_secondary(triplets))


@lru_cache(maxsize=None)
def original_patterns(triplets: bool) -> tuple[Pattern, ...]:
    return tuple(_build_originals(triplets))


@lru_cache(maxsize=None)
def patterns_by_type(triplets: bool, flex_rank: int | None) -> tuple[tuple[Pattern, ...], ...]:
    """For each standard type t, the secondary patterns that use t as a natural card.

    When a wild rank is in force (post-original), cards of that rank are pooled with the jokers:
    using one "naturally" is identical to using it as a wild in the same slot, so patterns whose
    naturals include the wild rank are dropped (their effect is covered by pool fills).
    """
    pats = secondary_patterns(triplets)
    if flex_rank is not None:
        pats = tuple(p for p in pats if all(rank_of(t) != flex_rank for t, _ in p.nat))
    by_type: list[list[Pattern]] = [[] for _ in range(NUM_STD)]
    for p in pats:
        for t, _ in p.nat:
            by_type[t].append(p)
    # Bigger natural coverage first: finds good solutions early.
    return tuple(tuple(sorted(lst, key=lambda p: (p.wilds, -p.size))) for lst in by_type)
