"""R4.1, R4.6, R5: meld validity (reference validators)."""

import pytest

from indian_rummy.cards import base_points, card_points, parse_type
from indian_rummy.melds import can_extend_original, is_pure_sequence, is_secondary, is_triplet, is_valid_original
from tests.helpers import T


@pytest.mark.parametrize("codes,ok", [
    ("2H 3H 4H", True),
    ("2H 3H 4H 5H", True),
    ("AH 2H 3H", True),           # R5.2 Ace low
    ("QH KH AH", True),           # R5.2 Ace high
    ("JH QH KH AH", True),
    ("AH 2H 3H 4H", True),
    ("KH AH 2H", False),          # R5.2 no wrap
    ("2H 3H 5H", False),          # gap
    ("2H 3S 4H", False),          # mixed suits
    ("2H 3H 4H 5H 6H", False),    # R5.1 max 4
    ("2H 3H", False),
    ("2H 2H 3H", False),
])
def test_r5_2_pure_sequence(codes, ok):
    assert is_pure_sequence(T(codes)) is ok


@pytest.mark.parametrize("codes,ok", [
    ("7H 7H 7H", True),
    ("7H 7H 7S", False),
    ("JK JK JK", False),
    ("7H 7H JK", False),
])
def test_r5_4_triplet(codes, ok):
    assert is_triplet(T(codes)) is ok


@pytest.mark.parametrize("codes,ok", [
    ("5H 6H 7H", True),
    ("5H 6H 7H 8H", True),
    ("9C 9C 9C", True),           # triplet is an original
    ("9C 9H 9S", False),          # set is not an original (R5.3)
    ("5H JK 7H", False),          # no jokers in the original (R4.1)
    ("5H 6H 7S", False),
])
def test_r4_1_valid_original(codes, ok):
    assert is_valid_original(T(codes)) is ok


def test_r4_1_wild_rank_card_is_natural_in_original():
    # CJ rank 3: 2C-3C-4C is still a pure sequence for the original (R4.1, D-007)
    assert is_valid_original(T("2C 3C 4C"))


@pytest.mark.parametrize("codes,wild,ok", [
    ("5H JK 7H", None, True),          # joker in sequence
    ("JK JK 7H", None, True),          # any number of wilds, >= 1 natural (R5.5)
    ("JK JK JK", None, False),         # all-wild is not a meld
    ("JK JK JK JK", None, False),
    ("KH AH JK", None, True),          # Q-K-A with joker as Q
    ("AH JK 3H", None, True),
    ("KH JK 2H", None, False),         # would need wrap
    ("AH KH JK 2H", None, False),
    ("7H 7S JK", None, True),          # set with joker
    ("7H 7S 7D 7C", None, True),
    ("7H 7S 7D 7C JK", None, False),   # 5 cards
    ("7H 7H 7S", None, False),         # duplicate suit in a set
    ("7H 7H JK", None, False),         # pseudo-triplet not allowed
    ("7H 7H 7H", None, True),          # natural triplet is a valid secondary
    ("5H 6S 7H", None, False),
    ("6H JK JK", 6, True),             # R5.6 wild-rank card counted as natural
    ("6H 6S 6D", 6, True),
    ("6H 6H JK", 6, True),
    ("5H 6S 7H", 6, True),             # 6S acts as wild 6H
    ("5H 6S 9H", 6, False),
    ("5H 6S 6D 8H", 6, True),          # two wild-rank cards fill 6H, 7H
    ("2C 3C 4C", 3, True),
])
def test_r5_secondary(codes, wild, ok):
    assert is_secondary(T(codes), wild) is ok


@pytest.mark.parametrize("orig,card,ok", [
    ("5H 6H 7H", "8H", True),
    ("5H 6H 7H", "4H", True),
    ("5H 6H 7H", "9H", False),
    ("5H 6H 7H", "JK", False),         # no joker extension (R4.6)
    ("QH KH AH", "JH", True),
    ("QH KH AH", "2H", False),         # wrap
    ("AH 2H 3H", "4H", True),
    ("9C 9C 9C", "9C", False),         # triplets cannot be extended
    ("5H 6H 7H 8H", "9H", False),      # already 4
])
def test_r4_6_extension(orig, card, ok):
    assert can_extend_original(T(orig), parse_type(card)) is ok


def test_r4_6_wild_rank_natural_extension_allowed():
    # wild rank 8: the real 8H still extends 5H-6H-7H (D-016)
    assert can_extend_original(T("5H 6H 7H"), parse_type("8H"))


@pytest.mark.parametrize("code,wild,pts", [
    ("AS", None, 10), ("KS", None, 10), ("QS", None, 10), ("JS", None, 10),
    ("10S", None, 10), ("9S", None, 9), ("2S", None, 2), ("JK", None, 0),
    ("9S", 9, 0), ("KD", 13, 0), ("9S", 5, 9),
])
def test_r7_5_points(code, wild, pts):
    assert card_points(parse_type(code), wild) == pts
    assert base_points(parse_type("JK")) == 0
