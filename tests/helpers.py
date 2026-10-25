"""Test helpers: build rigged game positions from card codes."""

from __future__ import annotations

from collections import defaultdict

from indian_rummy.cards import parse_type, parse_types
from indian_rummy.game import Phase, Player, RummyGame


def first_option(game, pid, options):
    return options[0]


def rig(hands: list[str], open_card: str | None = "2D", draw_top: str = "", cj: str = "3C",
        post_original: tuple[bool, bool] = (False, False), locked: tuple[str, str] = ("", ""),
        current: int = 0, turns: tuple[int, int] = (1, 1), policies: bool = False) -> RummyGame:
    """A 2-player v13 game in the DRAW phase of `current`.

    draw_top: codes listed top-first (the first code is drawn next).
    locked: codes (subset of the hand) of each player's revealed original.
    turns: completed turns per player (0 keeps the forfeit window open).
    """
    g = RummyGame(seed=0, dealer=1 - current,
                  reveal_policy={0: first_option, 1: first_option} if policies else None)
    g.init_game()
    pool: dict[int, list[int]] = defaultdict(list)
    for card in range(g.variant.deck_size):
        pool[g.type_of(card)].append(card)

    def take(code: str) -> int:
        return pool[parse_type(code)].pop()

    g.players = [Player(pid) for pid in range(2)]
    for pid, codes in enumerate(hands):
        g.players[pid].hand = [take(c) for c in codes.split()]
        g.players[pid].post_original = post_original[pid]
        g.players[pid].turns = turns[pid]
        lock_codes = locked[pid].split()
        for code in lock_codes:
            t = parse_type(code)
            card = next(c for c in g.players[pid].hand if g.type_of(c) == t and c not in g.players[pid].locked)
            g.players[pid].locked.append(card)
    g.cj = take(cj)
    g.wild_rank = None if parse_type(cj) == 52 else (parse_type(cj) % 13 + 1)
    g.discard_pile = [take(open_card)] if open_card else []
    top = [take(c) for c in draw_top.split()]
    rest = [c for lst in pool.values() for c in lst]
    g.draw_pile = rest + list(reversed(top))
    g.cj_face_up = all(post_original)
    g.events = []
    g.result = None
    g.turn = sum(turns)
    g.reveal_queue = []
    g.current = current
    g.phase = Phase.DRAW
    return g


def hand_types(g: RummyGame, pid: int) -> list[int]:
    return sorted(g.type_of(c) for c in g.players[pid].hand)


def card_in_hand(g: RummyGame, pid: int, code: str) -> int:
    t = parse_type(code)
    p = g.players[pid]
    return next(c for c in p.hand if g.type_of(c) == t and c not in p.locked)


def T(codes: str) -> list[int]:
    return parse_types(codes)
