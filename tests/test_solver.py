"""Solver (SPEC §4.3) vs a brute-force oracle built on melds.py, plus hand-made R6/R7 cases."""

from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import pytest

from indian_rummy.cards import JOKER, card_points, counts_of, make_type
from indian_rummy.melds import can_extend_original, is_secondary, is_valid_original
from indian_rummy.solver import HandContext, find_originals, is_win, solve
from tests.helpers import T


# --------------------------------------------------------------------------- oracle

def _remove(slots, size):
    s = list(slots)
    s.remove(size)
    return tuple(s)


@lru_cache(maxsize=None)
def _oracle_dead(cards: tuple, slots: tuple, wild):
    if not cards:
        return 0
    first, rest = cards[0], cards[1:]
    best = card_points(first, wild) + _oracle_dead(rest, slots, wild)
    # Scoring lets a meld sit in any slot at least its size (3/3/3/3 + 1 card, R7.3); try every pairing.
    for slot in set(slots):
        for size in (3, 4):
            if size > slot:
                continue
            for combo in combinations(range(len(rest)), size - 1):
                group = [first] + [rest[i] for i in combo]
                if is_secondary(group, wild):
                    remaining = tuple(rest[i] for i in range(len(rest)) if i not in combo)
                    best = min(best, _oracle_dead(remaining, _remove(slots, slot), wild))
    return best


@lru_cache(maxsize=None)
def _oracle_cover(cards: tuple, slots: tuple, wild) -> bool:
    if not cards:
        return not slots
    first, rest = cards[0], cards[1:]
    for size in set(slots):
        for combo in combinations(range(len(rest)), size - 1):
            group = [first] + [rest[i] for i in combo]
            if is_secondary(group, wild):
                remaining = tuple(rest[i] for i in range(len(rest)) if i not in combo)
                if _oracle_cover(remaining, _remove(slots, size), wild):
                    return True
    return False


def _without(cards, sub):
    cards = list(cards)
    for t in sub:
        cards.remove(t)
    return tuple(sorted(cards))


def oracle_post(cards, locked, wild, template, exact=False):
    """Post-original: locked original (optionally extended by one natural card) + secondaries."""
    rest = _without(cards, locked)
    options = [(rest, _remove(template, len(locked)))]
    if len(locked) == 3 and 4 in template:
        for e in set(rest):
            if can_extend_original(list(locked), e):
                options.append((_without(rest, [e]), _remove(template, 4)))
    if exact:
        return any(_oracle_cover(r, s, wild) for r, s in options)
    return min(_oracle_dead(r, s, wild) for r, s in options)


def oracle_pre(cards, template, must=None, exact=False):
    """Pre-original: some natural original (containing `must`) + secondaries with jokers only."""
    cards = tuple(sorted(cards))
    vals = []
    for size in (3, 4):
        if size not in template:
            continue
        for combo in set(combinations(cards, size)):
            if is_valid_original(list(combo)) and (must is None or must in combo):
                rest = _without(cards, combo)
                s = _remove(template, size)
                vals.append(_oracle_cover(rest, s, None) if exact else _oracle_dead(rest, s, None))
    if exact:
        return any(vals)
    if vals:
        return min(vals)
    return _oracle_dead(cards, tuple(template), None)


# --------------------------------------------------------------------------- random hands

def dense_pool(rng):
    """Cards from 2 suits x 7 ranks + jokers so random hands are rich in melds."""
    suits = rng.sample(range(4), 2)
    lo = rng.randint(1, 7)
    ranks = [((lo - 1 + k) % 13) + 1 for k in range(7)]
    pool = [make_type(s, r) for s in suits for r in ranks for _ in range(3)]
    pool += [make_type(rng.randrange(4), r) for r in ranks]  # a few other suits for sets
    pool += [JOKER] * 3
    return pool


def random_hand(rng, n):
    return sorted(rng.sample(dense_pool(rng), n))


def _post_case(rng, n, template):
    """Hand of n cards with a locked 3-card pure sequence."""
    while True:
        cards = random_hand(rng, n)
        origs = find_originals(counts_of(cards), sizes=(3,))
        if origs:
            locked = tuple(sorted(rng.choice(origs)))
            return cards, locked


TEMPLATES = [(3, 4), (3, 3, 4)]


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(40))
def test_solver_matches_oracle_post_original(seed):
    rng = random.Random(seed)
    for _ in range(25):
        template = rng.choice(TEMPLATES)
        n = sum(template) + rng.choice([0, 0, 1, -1])
        cards, locked = _post_case(rng, n, template)
        wild = rng.choice([None, rng.randint(1, 13)])
        ctx = HandContext(post_original=True, wild_rank=wild, locked=locked, template=template)
        expect = oracle_post(cards, locked, wild, template)
        got = solve(counts_of(cards), ctx)
        assert got.points == expect, (cards, locked, wild, template, got)
        if n == sum(template):
            assert (is_win(counts_of(cards), ctx) is not None) == oracle_post(cards, locked, wild, template, exact=True)


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(40))
def test_solver_matches_oracle_pre_original(seed):
    rng = random.Random(1000 + seed)
    for _ in range(25):
        template = rng.choice(TEMPLATES)
        n = sum(template) + rng.choice([0, 0, 1])
        cards = random_hand(rng, n)
        must = rng.choice([None, rng.choice(cards)])
        if must == JOKER:
            must = None
        ctx = HandContext(post_original=False, must_contain=must, template=template)
        expect = oracle_pre(cards, template, must)
        got = solve(counts_of(cards), ctx)
        assert got.points == expect, (cards, must, template, got)
        if n == sum(template):
            assert (is_win(counts_of(cards), ctx) is not None) == oracle_pre(cards, template, must, exact=True)


def test_solver_arrangement_is_consistent():
    """The returned melds are valid and cover exactly the hand minus `uncovered`."""
    rng = random.Random(7)
    for _ in range(300):
        cards, locked = _post_case(rng, 13, (3, 3, 3, 4))
        wild = rng.choice([None, rng.randint(1, 13)])
        ctx = HandContext(post_original=True, wild_rank=wild, locked=locked)
        a = solve(counts_of(cards), ctx)
        used = sorted([t for m in a.melds for t in m] + a.uncovered)
        assert used == sorted(cards)
        assert sorted(a.melds[0][:3]) == sorted(locked) or sorted(a.melds[0]) == sorted(locked)
        for m in a.melds[1:]:
            assert is_secondary(m, wild), m
        assert a.points == sum(card_points(t, wild) for t in a.uncovered)


# --------------------------------------------------------------------------- hand-made cases

def test_r6_1_win_exact_template():
    hand = counts_of(T("5H 6H 7H 2S 2D 2C 9C 10C JC KS KD KC KH"))
    ctx = HandContext(post_original=True, wild_rank=None, locked=tuple(T("5H 6H 7H")))
    w = is_win(hand, ctx)
    assert w is not None and sorted(len(m) for m in w.melds) == [3, 3, 3, 4]


def test_r6_1_no_win_with_wrong_shape():
    # 3 + 5 + 5 shaped hand: sequences longer than 4 do not count (D-004)
    hand = counts_of(T("5H 6H 7H 2S 3S 4S 5S 6S 9C 10C JC QC KC"))
    ctx = HandContext(post_original=True, locked=tuple(T("5H 6H 7H")))
    assert is_win(hand, ctx) is None


def test_r6_2_same_turn_win_uses_jokers_only():
    hand = counts_of(T("5H 6H 7H 2S 2D JK 9C 10C JC KS KD KC KH"))
    pre = HandContext(post_original=False)
    assert is_win(hand, pre) is not None
    # needs the wild rank (3) as a joker -> not a same-turn win
    hand2 = counts_of(T("5H 6H 7H 2S 2D 3H 9C 10C JC KS KD KC KH"))
    assert is_win(hand2, pre) is None
    post = HandContext(post_original=True, wild_rank=3, locked=tuple(T("5H 6H 7H")))
    assert is_win(hand2, post) is not None


def test_r6_2_same_turn_win_must_contain_open_card():
    hand = counts_of(T("5H 6H 7H 2S 2D JK 9C 10C JC KS KD KC KH"))
    assert is_win(hand, HandContext(post_original=False, must_contain=T("6H")[0])) is not None
    # the original must contain 2S, but 2S can't be in a natural original here
    assert is_win(hand, HandContext(post_original=False, must_contain=T("2S")[0])) is None


def test_pure_pool_meld_needs_a_wild_rank_card():
    # wild rank 6: JK JK 6S is a valid meld (6S natural); JK JK JK is not (R5.5)
    hand = counts_of(T("5H 6H 7H 2S 2D 2C 9C 10C JC JK JK 6S 6D"))
    ctx = HandContext(post_original=True, wild_rank=6, locked=tuple(T("5H 6H 7H")))
    assert is_win(hand, ctx) is not None
    # Jokers spread into natural melds (e.g. JC JK JK JK), never forming an all-joker meld.
    hand2 = counts_of(T("5H 6H 7H 2S 2D 2C 9C 10C JC JK JK JK JK"))
    ctx2 = HandContext(post_original=True, wild_rank=None, locked=tuple(T("5H 6H 7H")))
    w = is_win(hand2, ctx2)
    assert w is not None and all(is_secondary(m, None) for m in w.melds[1:])


def test_r7_3_template_limits_partial_melds():
    # Two natural 4-card melds besides the original: only one 4-slot is available.
    hand = counts_of(T("5H 6H 7H 2S 3S 4S 5S 9C 10C JC QC KD 8D"))
    ctx = HandContext(post_original=True, locked=tuple(T("5H 6H 7H")))
    a = solve(hand, ctx)
    # best: orig(3) + 9C-QC (4) + 3S-5S (3); leftovers 2S + KD + 8D = 20
    assert sorted(len(m) for m in a.melds) == [3, 3, 4]
    assert a.points == 20


def test_r7_3_loser_may_count_four_three_card_melds():
    # orig + three 3-card melds + one loose card: only the 8D counts (3/3/3/3 + 1).
    hand = counts_of(T("5H 6H 7H 2S 2D 2C 9C 10C JC KS KD KC 8D"))
    ctx = HandContext(post_original=True, locked=tuple(T("5H 6H 7H")))
    a = solve(hand, ctx)
    assert sorted(len(m) for m in a.melds) == [3, 3, 3, 3]
    assert a.uncovered == T("8D") and a.points == 8
    assert is_win(hand, ctx) is None  # still not a win: that needs exactly 3/3/3/4


def test_r7_5_wild_rank_cards_score_zero_post_original():
    hand = counts_of(T("5H 6H 7H 9S 9D KC QD 2H 3S 4C 8C 10D AS"))
    ctx = HandContext(post_original=True, wild_rank=9, locked=tuple(T("5H 6H 7H")))
    a = solve(hand, ctx)
    assert all(card_points(t, 9) == 0 for t in a.uncovered if t in T("9S 9D"))


def test_r1_4a_no_wild_rank():
    hand = counts_of(T("5H 6H 7H 9S 9D KC QD 2H 3S 4C 8C 10D AS"))
    a = solve(hand, HandContext(post_original=True, wild_rank=None, locked=tuple(T("5H 6H 7H"))))
    b = solve(hand, HandContext(post_original=True, wild_rank=9, locked=tuple(T("5H 6H 7H"))))
    assert a.points > b.points


def test_r4_6_extension_used_when_helpful():
    hand = counts_of(T("5H 6H 7H 8H 2S 2D 2C 9C 10C JC KS KD KC"))
    ctx = HandContext(post_original=True, locked=tuple(T("5H 6H 7H")))
    w = is_win(hand, ctx)
    assert w is not None and sorted(w.original) == sorted(T("5H 6H 7H 8H"))


def test_r4_6_no_joker_extension():
    # The only way to win would be extending the original with a joker -> not allowed
    hand = counts_of(T("5H 6H 7H JK 2S 2D 2C 9C 10C JC KS KD KC"))
    ctx = HandContext(post_original=True, locked=tuple(T("5H 6H 7H")))
    w = is_win(hand, ctx)
    # JK can still join a secondary meld (e.g. KS KD KC JK), which is legal
    assert w is not None
    assert sorted(w.original) == sorted(T("5H 6H 7H"))


def test_find_originals():
    hand = counts_of(T("5H 6H 7H 8H 9C 9C 9C 2S JK"))
    found = {tuple(sorted(o)) for o in find_originals(hand)}
    assert tuple(sorted(T("5H 6H 7H"))) in found
    assert tuple(sorted(T("6H 7H 8H"))) in found
    assert tuple(sorted(T("5H 6H 7H 8H"))) in found
    assert tuple(sorted(T("9C 9C 9C"))) in found
    assert all(JOKER not in o for o in found)
