"""Game flow rules R1, R3, R4, R6-R9 on rigged positions."""

from __future__ import annotations

import random
from collections import Counter

import pytest

from indian_rummy import actions as A
from indian_rummy.cards import JOKER, NUM_STD, parse_type
from indian_rummy.config import V13
from indian_rummy.game import IllegalAction, Phase, RummyGame
from indian_rummy.match import Match
from tests.helpers import card_in_hand, first_option, rig

P = parse_type
BOTH = {0: first_option, 1: first_option}

# A filler hand with no original and no easy melds for the opponent.
FILLER = "2S 5S 8S JS 3H 9H QH 4D 7D 10D KD AC 6C"


# --------------------------------------------------------------------------- R1 setup

def test_r1_1_deck_composition():
    g = RummyGame(seed=1, reveal_policy=BOTH)
    assert g.variant.deck_size == 162
    c = Counter(g.types)
    assert all(c[t] == 3 for t in range(NUM_STD)) and c[JOKER] == 6


@pytest.mark.parametrize("seed", range(20))
def test_r1_5_deal(seed):
    g = RummyGame(seed=seed, reveal_policy=BOTH)
    g.init_game()
    assert all(len(p.hand) == 13 for p in g.players)
    assert len(g.discard_pile) == 1 and len(g.draw_pile) == 134
    zones = [c for p in g.players for c in p.hand] + g.discard_pile + g.draw_pile
    assert g.cj not in zones and len(set(zones)) == 161          # R1.4 CJ out of play
    assert g.current == (g.dealer + 1) % 2                       # R1.6


def test_r1_4a_printed_joker_cj_means_no_wild():
    g = rig([FILLER, FILLER], cj="JK", post_original=(True, False), locked=("", ""))
    assert g.wild_rank is None


def test_r1_6_dealer_alternates_in_match():
    m = Match(seed=3, reveal_policy=BOTH)
    dealers = []
    for _ in range(4):
        g = m.start_round()
        dealers.append(g.dealer)
        rng = random.Random(0)
        while not g.is_over():
            g.step(rng.choice(g.get_legal_actions()))
        m.finish_round()
    assert dealers[0] != dealers[1] and dealers[0] == dealers[2] and dealers[1] == dealers[3]


# --------------------------------------------------------------------------- R3 turn

def test_r3_2_pre_original_open_draw_only_if_completes_original():
    g = rig(["5H 6H 2S 9S QS 3D 8D KD 4C 7C 10C AC JC", FILLER], open_card="7H")
    assert A.DRAW_OPEN in g.get_legal_actions()
    g2 = rig(["5H 6H 2S 9S QS 3D 8D KD 4C 7C 10C AC JC", FILLER], open_card="9H")
    assert A.DRAW_OPEN not in g2.get_legal_actions()
    assert A.DRAW_CLOSED in g2.get_legal_actions()


def test_r3_2_post_original_open_draw_always():
    g = rig(["5H 6H 7H 9S QS 3D 8D KD 4C 7C 10C AC JC", FILLER], open_card="2S",
            post_original=(True, False), locked=("5H 6H 7H", ""))
    assert A.DRAW_OPEN in g.get_legal_actions()


def test_r3_3_cannot_discard_open_card_or_identical_copy():
    g = rig(["5H 6H 7H 9S 9S 3D 8D KD 4C 7C 10C AC JC", FILLER], open_card="9S",
            post_original=(True, False), locked=("5H 6H 7H", ""))
    g.step(A.DRAW_OPEN)
    legal = g.get_legal_actions()
    assert A.discard_action(P("9S")) not in legal                  # A2
    assert A.discard_action(P("3D")) in legal


def test_r3_4_open_pickup_must_be_revealed_and_kept():
    g = rig(["5H 6H 2S 9S QS 3D 8D KD 4C 7C 10C AC JC", FILLER], open_card="7H")
    g.step(A.DRAW_OPEN)
    legal = g.get_legal_actions()
    assert A.discard_action(P("5H")) not in legal and A.discard_action(P("6H")) not in legal
    g.step(A.discard_action(P("KD")))
    assert g.phase == Phase.REVEAL and g.current == 0
    opts = g.reveal_options()
    assert opts and all(P("7H") in [g.type_of(c) for c in o] for o in opts)
    g.step(("reveal", opts[0]))
    assert g.players[0].post_original and g.current == 1 and g.phase == Phase.DRAW


def test_r3_5_locked_cards_not_discardable():
    g = rig(["5H 6H 7H 9S QS 3D 8D KD 4C 7C 10C AC JC", FILLER],
            post_original=(True, False), locked=("5H 6H 7H", ""))
    g.step(A.DRAW_CLOSED)
    legal = g.get_legal_actions()
    for code in ("5H", "6H", "7H"):
        assert A.discard_action(P(code)) not in legal
    with pytest.raises(IllegalAction):
        g.step(("discard", g.players[0].locked[0]))


def test_r3_6_reshuffle_when_draw_pile_empty():
    g = rig([FILLER, FILLER], post_original=(True, True))
    g.discard_pile = g.draw_pile[:-1] + g.discard_pile
    g.draw_pile = g.draw_pile[-1:]
    n_disc = len(g.discard_pile)
    g.step(A.DRAW_CLOSED)
    assert len(g.discard_pile) == 1 and len(g.draw_pile) == n_disc - 1
    assert any(e["type"] == "reshuffle" for e in g.events)


# --------------------------------------------------------------------------- R4 original

def test_r4_2_reveal_mandatory_after_forming_original():
    g = rig(["5H 6H 2S 9S QS 3D 8D KD 4C 7C 10C AC JC", FILLER], draw_top="7H")
    g.step(A.DRAW_CLOSED)
    g.step(A.discard_action(P("KD")))
    assert g.phase == Phase.REVEAL and g.current == 0
    with pytest.raises(IllegalAction):
        g.step(A.DRAW_CLOSED)                                      # cannot skip the reveal (A1)


def test_r4_3_dealt_original_revealed_before_first_turn():
    for seed in range(300):
        g = RummyGame(seed=seed, reveal_policy=BOTH)
        g.init_game()
        revealed = [e["actor"] for e in g.events if e["type"] == "reveal"]
        if revealed:
            assert g.phase == Phase.DRAW and g.turn == 0
            assert all(g.players[pid].post_original for pid in revealed)
            return
    pytest.fail("no dealt original in 300 seeds")


def test_r4_3_dealt_original_waits_for_human_choice():
    for seed in range(300):
        g = RummyGame(seed=seed, reveal_policy={1: first_option})
        g.init_game()
        if g.phase == Phase.REVEAL:
            assert g.current == 0 and g.reveal_options()
            g.step(("reveal", g.reveal_options()[0]))
            assert g.players[0].post_original
            return
    pytest.fail("no dealt original for player 0 in 300 seeds")


def test_r4_4_reveal_locks_and_gives_wild_knowledge():
    g = rig(["5H 6H 2S 9S QS 3D 8D KD 4C 7C 10C AC JC", FILLER], draw_top="7H", cj="4S")
    assert g.get_state(0)["wild_rank"] is None
    g.step(A.DRAW_CLOSED)
    g.step(A.discard_action(P("KD")))
    g.step(("reveal", g.reveal_options()[0]))
    s0, s1 = g.get_state(0), g.get_state(1)
    assert s0["wild_rank"] == 4 and s1["wild_rank"] is None        # R11
    assert len(s0["locked"]) == 3 and s1["opponents"][0]["locked"] == s0["locked"]


def test_r4_5_cj_flips_when_both_revealed():
    g = rig(["5H 6H 2S 9S QS 3D 8D KD 4C 7C 10C AC JC", FILLER], draw_top="7H",
            post_original=(False, True), locked=("", ""))
    g.step(A.DRAW_CLOSED)
    g.step(A.discard_action(P("KD")))
    g.step(("reveal", g.reveal_options()[0]))
    assert g.cj_face_up and any(e["type"] == "cj_flip" for e in g.events)


# --------------------------------------------------------------------------- R6 winning

def test_r6_1_post_original_declares_automatically():
    g = rig(["5H 6H 7H 2S 2D 2C 9C 10C JC KS KD KC 4D", FILLER], draw_top="KH",
            post_original=(True, False), locked=("5H 6H 7H", ""))
    g.step(A.DRAW_CLOSED)
    g.step(A.discard_action(P("4D")))
    assert g.is_over() and g.result.winner == 0 and g.result.reason == "declare"
    assert g.result.points == [0, 80]                                # opponent pre-original (R7.2)


def test_r6_2_same_turn_win_pre_original():
    g = rig(["5H 6H 2S 2D JK 9C 10C JC KS KD KC KH 4D", FILLER], draw_top="7H")
    g.step(A.DRAW_CLOSED)
    g.step(A.discard_action(P("4D")))
    assert g.is_over() and g.result.winner == 0


def test_r6_2_same_turn_win_cannot_use_cj():
    # wild rank 3: 3H would complete 2S 2D 3H only as a wild -> no same-turn win, reveal instead
    g = rig(["5H 6H 2S 2D 3H 9C 10C JC KS KD KC KH 4D", FILLER], draw_top="7H", cj="3C")
    g.step(A.DRAW_CLOSED)
    g.step(A.discard_action(P("4D")))
    assert not g.is_over() and g.phase == Phase.REVEAL


# --------------------------------------------------------------------------- R7 scoring

def test_r7_3_loser_points_capped_at_80():
    g = rig(["5H 6H 7H 2S 2D 2C 9C 10C JC KS KD KC 4D", "5D 6D 7D KH KH QS QS JD JD 10S 10S AH AH"],
            draw_top="KH", post_original=(True, True), locked=("5H 6H 7H", "5D 6D 7D"))
    g.step(A.DRAW_CLOSED)
    g.step(A.discard_action(P("4D")))
    assert g.result.points[1] == 80


def test_r7_3_loser_points_optimal():
    g = rig(["5H 6H 7H 2S 2D 2C 9C 10C JC KS KD KC 4D", "5D 6D 7D 8S 8C 8H 3H 4H JK 9D 2H 2C AH"],
            draw_top="KH", post_original=(True, True), locked=("5H 6H 7H", "5D 6D 7D"), cj="QS")
    g.step(A.DRAW_CLOSED)
    g.step(A.discard_action(P("4D")))
    # P1: orig + 8S 8C 8H + 2H 3H 4H + (9D JK) ... best leaves 2C + AH + 9D or uses JK
    pts = g.result.points[1]
    assert pts == min(g.result.points[1], 80) and pts <= 12


def test_r7_6_forfeit_before_draw():
    g = rig([FILLER, FILLER], turns=(0, 0))
    assert A.FORFEIT in g.get_legal_actions()
    g.step(A.FORFEIT)
    assert g.result.points == [40, 0] and g.result.winner == 1


# --------------------------------------------------------------------------- R8 forfeit windows

def test_r8_1b_forfeit_after_closed_draw_throws_back_drawn_card():
    g = rig([FILLER, FILLER], draw_top="KS", turns=(0, 0))
    g.step(A.DRAW_CLOSED)
    drawn = g.players[0].drawn
    assert A.FORFEIT in g.get_legal_actions()
    g.step(A.FORFEIT)
    assert g.discard_pile[-1] == drawn and g.result.points[0] == 40


def test_r8_2_no_forfeit_after_open_draw_or_first_turn():
    g = rig(["5H 6H 7H 9S QS 3D 8D KD 4C 7C 10C AC JC", FILLER], open_card="2S", turns=(0, 0),
            post_original=(True, False), locked=("5H 6H 7H", ""))
    g.step(A.DRAW_OPEN)
    assert A.FORFEIT not in g.get_legal_actions()
    g2 = rig([FILLER, FILLER], turns=(1, 1))
    assert A.FORFEIT not in g2.get_legal_actions()


def test_r8_3_dealt_original_may_forfeit():
    g = rig(["5H 6H 7H 9S QS 3D 8D KD 4C 7C 10C AC JC", FILLER], turns=(0, 0),
            post_original=(True, False), locked=("5H 6H 7H", ""))
    assert A.FORFEIT in g.get_legal_actions()


# --------------------------------------------------------------------------- R9 match

def _play_random_match(seed):
    m = Match(seed=seed, reveal_policy=BOTH)
    rng = random.Random(seed)
    while not m.is_over():
        g = m.start_round()
        while not g.is_over():
            g.step(rng.choice(g.get_legal_actions()))
        m.finish_round()
    return m


def test_r9_match_multipliers_and_totals():
    m = _play_random_match(5)
    assert len(m.rounds) == 7
    assert [r.multiplier for r in m.rounds] == [2, 1, 1, 1, 1, 1, 2]
    tot = [sum(r.result.points[p] * r.multiplier for r in m.rounds) for p in range(2)]
    assert m.totals() == tot


def test_r9_2_tie_is_draw():
    m = _play_random_match(5)
    m.rounds = m.rounds[:0]
    assert m.winner() is None  # 0 - 0


# --------------------------------------------------------------------------- invariants

def _all_cards(g):
    return [c for p in g.players for c in p.hand] + g.discard_pile + g.draw_pile + [g.cj]


@pytest.mark.parametrize("seed", range(60))
def test_invariants_random_play(seed):
    g = RummyGame(seed=seed, reveal_policy=BOTH)
    g.init_game()
    rng = random.Random(seed)
    # avoid early forfeits so rounds run long
    while not g.is_over():
        legal = g.get_legal_actions()
        choices = [a for a in legal if a != A.FORFEIT] or legal
        g.step(rng.choice(choices))
        cards = _all_cards(g)
        assert sorted(cards) == list(range(162))
        if g.phase == Phase.DRAW:
            assert all(len(p.hand) == 13 for p in g.players)
        for p in g.players:
            assert set(p.locked) <= set(p.hand)
    assert g.result is not None


def test_determinism():
    def run(seed):
        g = RummyGame(seed=seed, reveal_policy=BOTH)
        g.init_game()
        rng = random.Random(99)
        while not g.is_over():
            g.step(rng.choice(g.get_legal_actions()))
        return g.events
    assert run(11) == run(11)


def test_state_hides_opponent_hand_and_deck():
    g = RummyGame(seed=4, reveal_policy=BOTH)
    g.init_game()
    s = g.get_state(0)
    other = set(g.players[1].hand)
    flat = repr(s)
    assert "draw_pile" not in s
    assert not other & set(s["hand"])
    assert set(s["opponents"][0]["locked"]) <= other
    assert s["opponents"][0]["hand_count"] == 13
