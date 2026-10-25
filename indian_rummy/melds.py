"""Reference meld validators (SPEC R4.1, R5).

Deliberately simple and slow: these are the oracle that the fast solver is tested against.
All functions take card *types* (see cards.py).
"""

from __future__ import annotations

from itertools import combinations

from .cards import JOKER, rank_of, suit_of

MELD_SIZES = (3, 4)  # R5.1


def _seq_fits(naturals: list[int], n: int) -> bool:
    """Can these natural cards sit inside one n-long same-suit window (R5.2)?

    Positions run 1..14 where 1 is Ace-low and 14 is Ace-high, so K-A-2 never fits.
    """
    if not naturals or any(t == JOKER for t in naturals):
        return False
    if len({suit_of(t) for t in naturals}) != 1:
        return False
    ranks = [rank_of(t) for t in naturals]
    if len(set(ranks)) != len(ranks):
        return False
    for ace_high in (False, True):
        vals = [14 if (r == 1 and ace_high) else r for r in ranks]
        lo, hi = min(vals), max(vals)
        if max(1, hi - n + 1) <= min(lo, 15 - n):
            return True
    return False


def is_pure_sequence(types: list[int]) -> bool:
    """R5.2 with natural cards only and no gaps."""
    n = len(types)
    if n not in MELD_SIZES:
        return False
    return _seq_fits(list(types), n)  # n distinct ranks in an n-window => no gaps


def is_triplet(types: list[int]) -> bool:
    """R5.4: three identical natural cards."""
    return len(types) == 3 and types[0] != JOKER and len(set(types)) == 1


def is_valid_original(types: list[int]) -> bool:
    """R4.1: pure sequence (3/4) or triplet. Wild-rank cards count as natural; Jk never allowed."""
    return is_pure_sequence(types) or is_triplet(types)


def _is_natural_set(naturals: list[int], n: int) -> bool:
    if not naturals or any(t == JOKER for t in naturals):
        return False
    if len({rank_of(t) for t in naturals}) != 1:
        return False
    suits = [suit_of(t) for t in naturals]
    return len(set(suits)) == len(suits) and n <= 4


def is_secondary(types: list[int], wild_rank: int | None = None) -> bool:
    """R5.3-R5.6: any valid meld (sequence, set or triplet) where printed Jk are wild and,
    if wild_rank is given, cards of that rank may be wild or natural. Needs >= 1 natural."""
    n = len(types)
    if n not in MELD_SIZES:
        return False
    jokers = [t for t in types if t == JOKER]
    flex = [t for t in types if t != JOKER and wild_rank is not None and rank_of(t) == wild_rank]
    fixed = [t for t in types if t != JOKER and not (wild_rank is not None and rank_of(t) == wild_rank)]
    for k in range(len(flex) + 1):
        for chosen in combinations(flex, k):
            naturals = fixed + list(chosen)
            wilds = len(jokers) + len(flex) - k
            if not naturals:
                continue
            if wilds == 0 and is_triplet(naturals):
                return True
            if _seq_fits(naturals, n) or _is_natural_set(naturals, n):
                return True
    return False


def can_extend_original(original: list[int], card: int) -> bool:
    """R4.6: a 3-card pure sequence extended by one natural adjacent card."""
    if len(original) != 3 or not is_pure_sequence(original):
        return False
    return card != JOKER and is_pure_sequence(list(original) + [card])
