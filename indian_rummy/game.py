"""Round state machine (SPEC §4.4), also implementing RLCard's Game interface.

Actions accepted by step():
  int                      RL action id (actions.py): draw closed/open, discard by type, forfeit
  ("discard", card_id)     discard a specific physical card (UI)
  ("reveal", card_ids)     reveal these physical cards as the original (R4)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np

from . import actions as A
from .cards import JOKER, NUM_TYPES, build_deck, rank_of
from .config import V13, Variant
from .solver import Arrangement, HandContext, find_originals, is_win, solve


class Phase(str, Enum):
    REVEAL = "reveal"
    DRAW = "draw"
    DISCARD = "discard"
    OVER = "over"


class IllegalAction(ValueError):
    pass


@dataclass
class Player:
    pid: int
    hand: list[int] = field(default_factory=list)        # physical ids
    locked: list[int] = field(default_factory=list)      # revealed original (subset of hand)
    post_original: bool = False
    turns: int = 0                                       # completed turns
    known_open: list[int] = field(default_factory=list)  # open-pile cards taken, still held (public)
    drawn: int | None = None                             # card drawn this turn
    drawn_source: str | None = None                      # "closed" | "open"
    must_contain: int | None = None                      # R3.4: open card that must be in the reveal
    discards: list[int] = field(default_factory=list)    # every card this player discarded (public)


@dataclass
class RoundResult:
    winner: int | None
    reason: str                                   # "declare" | "forfeit" | "exhausted"
    points: list[int]                             # raw round points (no match multiplier)
    arrangements: list[Arrangement | None]


RevealPolicy = Callable[["RummyGame", int, list[tuple[int, ...]]], tuple[int, ...]]


class RummyGame:
    def __init__(self, variant: Variant = V13, num_players: int = 2, seed: int | None = None,
                 dealer: int | None = None, reveal_policy: dict[int, RevealPolicy] | None = None):
        self.variant = variant
        self.num_players = num_players
        self.np_random = np.random.RandomState(seed)
        self.allow_step_back = False
        self.dealer_arg = dealer
        self.reveal_policy: dict[int, RevealPolicy] = reveal_policy or {}
        self.types = tuple(int(t) for t in build_deck(variant))
        self.phase: Phase = Phase.OVER

    # ------------------------------------------------------------------ setup
    def init_game(self):
        v = self.variant
        rng = self.np_random
        self.dealer = self.dealer_arg if self.dealer_arg is not None else int(rng.randint(self.num_players))
        deck = list(range(v.deck_size))
        rng.shuffle(deck)
        self.players = [Player(pid) for pid in range(self.num_players)]
        self.events: list[dict] = []
        self.result: RoundResult | None = None
        self.turn = 0
        for _ in range(v.hand_size):
            for k in range(self.num_players):
                self.players[(self.dealer + 1 + k) % self.num_players].hand.append(deck.pop())
        self.discard_pile: list[int] = [deck.pop()]                 # R1.3
        self.cj: int = deck.pop(int(rng.randint(len(deck))))        # R1.4
        self.wild_rank: int | None = None if self.types[self.cj] == JOKER else rank_of(int(self.types[self.cj]))
        self.draw_pile: list[int] = deck
        self.cj_face_up = False
        self.current = (self.dealer + 1) % self.num_players         # R1.6
        self._emit("deal", self.dealer, dealer=self.dealer, open=self.discard_pile[-1])
        # R4.3: dealt originals are revealed before the first turn, starting with the first player.
        self.reveal_queue: list[int] = []
        for k in range(self.num_players):
            pid = (self.current + k) % self.num_players
            if find_originals(self.counts(pid), triplets=v.triplets_possible):
                self.reveal_queue.append(pid)
        self._first_player = self.current
        self._after_reveal_resume = "start"
        self._advance_reveals()
        return self.get_state(self.current), self.current

    # ------------------------------------------------------------------ helpers
    def _emit(self, etype: str, actor: int | None, **kw):
        self.events.append({"seq": len(self.events), "type": etype, "actor": actor, **kw})

    def type_of(self, card: int) -> int:
        return int(self.types[card])

    def counts(self, pid: int, extra: list[int] = (), minus: list[int] = ()) -> list[int]:
        c = [0] * NUM_TYPES
        for card in self.players[pid].hand:
            c[self.types[card]] += 1
        for card in extra:
            c[self.types[card]] += 1
        for card in minus:
            c[self.types[card]] -= 1
        return c

    def known_wild_rank(self, pid: int) -> int | None:
        return self.wild_rank if self.players[pid].post_original else None

    def hand_context(self, pid: int, must_contain_type: int | None = None) -> HandContext:
        p = self.players[pid]
        v = self.variant
        if p.post_original:
            return HandContext(post_original=True, wild_rank=self.wild_rank,
                               locked=tuple(sorted(self.type_of(c) for c in p.locked)),
                               template=v.template, triplets=v.triplets_possible)
        mc = must_contain_type
        if mc is None and p.must_contain is not None:
            mc = self.type_of(p.must_contain)
        return HandContext(post_original=False, must_contain=mc,
                           template=v.template, triplets=v.triplets_possible)

    def _forfeit_window(self, pid: int) -> bool:
        return self.players[pid].turns == 0  # R8.1

    def _open_legal(self, pid: int) -> bool:
        if not self.discard_pile:
            return False
        p = self.players[pid]
        if p.post_original:
            return True
        top = self.discard_pile[-1]
        return bool(find_originals(self.counts(pid, extra=[top]), must_contain=self.type_of(top),
                                   triplets=self.variant.triplets_possible))  # R3.2

    def _closed_available(self) -> bool:
        return bool(self.draw_pile) or len(self.discard_pile) > 1

    def _discard_types(self, pid: int) -> list[int]:
        p = self.players[pid]
        free = [c for c in p.hand if c not in p.locked]                     # R3.5
        types = sorted({self.type_of(c) for c in free})
        if p.drawn_source == "open":
            banned = self.type_of(p.drawn)                                  # R3.3 / A2
            types = [t for t in types if t != banned]
        if p.must_contain is not None:                                      # R3.4
            mc = self.type_of(p.must_contain)
            counts = self.counts(pid)
            ok = []
            for t in types:
                counts[t] -= 1
                if find_originals(counts, must_contain=mc, triplets=self.variant.triplets_possible):
                    ok.append(t)
                counts[t] += 1
            types = ok
        return types

    # ------------------------------------------------------------------ RLCard interface
    def get_num_players(self) -> int:
        return self.num_players

    @staticmethod
    def get_num_actions() -> int:
        return A.NUM_ACTIONS

    def get_player_id(self) -> int:
        return self.current

    def is_over(self) -> bool:
        return self.phase == Phase.OVER

    def get_legal_actions(self) -> list[int]:
        pid = self.current
        p = self.players[pid]
        if self.phase == Phase.DRAW:
            out = []
            if self._closed_available():
                out.append(A.DRAW_CLOSED)
            if self._open_legal(pid):
                out.append(A.DRAW_OPEN)
            if self._forfeit_window(pid):
                out.append(A.FORFEIT)
            return out
        if self.phase == Phase.DISCARD:
            out = [A.discard_action(t) for t in self._discard_types(pid)]
            if self._forfeit_window(pid) and p.drawn_source == "closed":
                out.append(A.FORFEIT)  # R8.1b
            return out
        return []

    def reveal_options(self, pid: int | None = None) -> list[tuple[int, ...]]:
        """Physical-card tuples the player may reveal (R4.1/R4.2, R3.4)."""
        pid = self.current if pid is None else pid
        p = self.players[pid]
        mc = self.type_of(p.must_contain) if p.must_contain is not None else None
        out = []
        for types in find_originals(self.counts(pid), must_contain=mc,
                                    triplets=self.variant.triplets_possible):
            pool = list(p.hand)
            if p.must_contain is not None:
                pool.remove(p.must_contain)
                pool.insert(0, p.must_contain)
            ids = []
            for t in types:
                card = next(c for c in pool if self.type_of(c) == t)
                pool.remove(card)
                ids.append(card)
            out.append(tuple(ids))
        return out

    def step(self, action):
        if self.phase == Phase.OVER:
            raise IllegalAction("round is over")
        if isinstance(action, tuple) and action[0] == "reveal":
            self._do_reveal(self.current, tuple(action[1]))
        elif isinstance(action, tuple) and action[0] == "discard":
            card = int(action[1])
            p = self.players[self.current]
            if card not in p.hand or card in p.locked \
                    or A.discard_action(self.type_of(card)) not in self.get_legal_actions():
                raise IllegalAction(f"cannot discard card {card}")
            self._do_discard(self.current, card)
        else:
            a = int(action)
            if self.phase == Phase.REVEAL or a not in self.get_legal_actions():
                raise IllegalAction(f"illegal action {A.describe(a)} in phase {self.phase.value}")
            if a == A.DRAW_CLOSED:
                self._do_draw(self.current, "closed")
            elif a == A.DRAW_OPEN:
                self._do_draw(self.current, "open")
            elif a == A.FORFEIT:
                self._do_forfeit(self.current)
            else:
                t = A.discard_type(a)
                p = self.players[self.current]
                card = next(c for c in p.hand if self.type_of(c) == t and c not in p.locked)
                self._do_discard(self.current, card)
        return self.get_state(self.current), self.current

    # ------------------------------------------------------------------ transitions
    def _start_turn(self, pid: int):
        self.current = pid
        if self.turn >= self.variant.turn_cap:                     # R3.7
            return self._end_round(None, "exhausted")
        if not self._closed_available() and not self._open_legal(pid):
            return self._end_round(None, "exhausted")              # R3.6
        self.phase = Phase.DRAW

    def _do_draw(self, pid: int, source: str):
        p = self.players[pid]
        if source == "closed":
            if not self.draw_pile:
                self._reshuffle()
            card = self.draw_pile.pop()
        else:
            card = self.discard_pile.pop()
            p.known_open.append(card)
            if not p.post_original:
                p.must_contain = card
        p.hand.append(card)
        p.drawn, p.drawn_source = card, source
        self._emit("draw", pid, source=source, card=card)
        if not self.draw_pile:
            self._reshuffle()
        self.phase = Phase.DISCARD

    def _reshuffle(self):
        if len(self.discard_pile) <= 1:
            return
        top = self.discard_pile.pop()
        pile = self.discard_pile
        self.np_random.shuffle(pile)
        self.draw_pile = pile + self.draw_pile
        self.discard_pile = [top]
        self._emit("reshuffle", None, count=len(self.draw_pile))

    def _do_discard(self, pid: int, card: int):
        p = self.players[pid]
        p.hand.remove(card)
        if card in p.known_open:
            p.known_open.remove(card)
        p.discards.append(card)
        self.discard_pile.append(card)
        self._emit("discard", pid, card=card)
        self._finish_turn(pid)

    def _finish_turn(self, pid: int):
        p = self.players[pid]
        self.turn += 1
        p.turns += 1
        # R6: automatic declaration (A7). Pre-original same-turn win uses printed jokers only (R6.2).
        arr = is_win(self.counts(pid), self.hand_context(pid))
        if arr is not None:
            self._emit("declare", pid, melds=arr.melds)
            return self._end_round(pid, "declare", winner_arrangement=arr)
        p.drawn = p.drawn_source = None
        if not p.post_original and self.reveal_options(pid):
            self.reveal_queue = [pid]                              # R4.2 (A1: mandatory)
            self._after_reveal_resume = "next"
            return self._advance_reveals()
        p.must_contain = None
        self._start_turn((pid + 1) % self.num_players)

    def _advance_reveals(self):
        while self.reveal_queue:
            pid = self.reveal_queue[0]
            self.current = pid
            self.phase = Phase.REVEAL
            policy = self.reveal_policy.get(pid)
            if policy is None:
                return  # wait for an explicit ("reveal", ids) action
            choice = policy(self, pid, self.reveal_options(pid))
            self._apply_reveal(pid, tuple(choice))
        if self._after_reveal_resume == "start":
            self._start_turn(self._first_player)
        else:
            self._start_turn((self._last_revealer + 1) % self.num_players)

    def _do_reveal(self, pid: int, cards: tuple[int, ...]):
        if self.phase != Phase.REVEAL or not self.reveal_queue or self.reveal_queue[0] != pid:
            raise IllegalAction("no reveal pending")
        self._apply_reveal(pid, cards)
        self._advance_reveals()

    def _apply_reveal(self, pid: int, cards: tuple[int, ...]):
        options = {tuple(sorted(self.type_of(c) for c in o)) for o in self.reveal_options(pid)}
        p = self.players[pid]
        if len(set(cards)) != len(cards) or any(c not in p.hand for c in cards) \
                or tuple(sorted(self.type_of(c) for c in cards)) not in options:
            raise IllegalAction("not a valid original")
        # R3.4 is enforced by reveal_options (must_contain) above; identical copies are interchangeable.
        p.locked = list(cards)
        p.post_original = True
        p.must_contain = None
        self.reveal_queue.pop(0)
        self._last_revealer = pid
        self._emit("reveal", pid, cards=list(cards))
        if all(q.post_original for q in self.players) and not self.cj_face_up:
            self.cj_face_up = True                                  # R4.5
            self._emit("cj_flip", None, card=self.cj)

    def _do_forfeit(self, pid: int):
        p = self.players[pid]
        if self.phase == Phase.DISCARD:                             # R8.1b: throw back the drawn card
            card = p.drawn
            p.hand.remove(card)
            self.discard_pile.append(card)
            self._emit("discard", pid, card=card)
        self._emit("forfeit", pid)
        self._end_round(pid, "forfeit")

    # ------------------------------------------------------------------ scoring
    def loser_points(self, pid: int) -> tuple[int, Arrangement | None]:
        """R7.2/R7.3: 80 if pre-original, else optimal uncovered points capped at 80."""
        p = self.players[pid]
        v = self.variant
        if not p.post_original:
            return v.no_original_points, None
        arr = solve(self.counts(pid), self.hand_context(pid))
        return min(arr.points, v.max_points), arr

    def _end_round(self, actor: int | None, reason: str, winner_arrangement: Arrangement | None = None):
        n = self.num_players
        points = [0] * n
        arrs: list[Arrangement | None] = [None] * n
        winner = None
        if reason == "declare":
            winner = actor
            arrs[actor] = winner_arrangement
            for pid in range(n):
                if pid != actor:
                    points[pid], arrs[pid] = self.loser_points(pid)
        elif reason == "forfeit":
            points[actor] = self.variant.forfeit_points            # R7.6
            if n == 2:
                winner = 1 - actor
        else:
            for pid in range(n):
                points[pid], arrs[pid] = self.loser_points(pid)    # R7.4
        self.result = RoundResult(winner, reason, points, arrs)
        self.phase = Phase.OVER
        self._emit("round_end", winner, reason=reason, points=list(points))

    def get_payoffs(self) -> list[float]:
        """SPEC §5.4: (mean opponent points - own points) / 80, clipped to [-1, 1]."""
        pts = self.result.points
        n = self.num_players
        out = []
        for pid in range(n):
            others = sum(pts) - pts[pid]
            out.append(float(np.clip((others / (n - 1) - pts[pid]) / 80.0, -1.0, 1.0)))
        return out

    # ------------------------------------------------------------------ observation
    def get_state(self, pid: int) -> dict:
        """Raw information set of `pid` (R11). Never contains other players' hidden cards."""
        p = self.players[pid]
        knows_cj = p.post_original or self.cj_face_up
        return {
            "player": pid,
            "current_player": self.current,
            "phase": self.phase.value,
            "turn": self.turn,
            "hand": list(p.hand),
            "locked": list(p.locked),
            "post_original": p.post_original,
            "wild_known": knows_cj,
            "wild_rank": self.wild_rank if knows_cj else None,
            "cj": self.cj if knows_cj else None,
            "cj_face_up": self.cj_face_up,
            "discard_pile": list(self.discard_pile),
            "draw_count": len(self.draw_pile),
            "drawn": p.drawn,
            "drawn_source": p.drawn_source,
            "must_contain": p.must_contain,
            "forfeit_window": self._forfeit_window(pid),
            "my_discards": list(p.discards),
            "opponents": [
                {"pid": q.pid, "hand_count": len(q.hand), "post_original": q.post_original,
                 "locked": list(q.locked), "known_open": list(q.known_open), "discards": list(q.discards)}
                for q in self.players if q.pid != pid
            ],
            "legal_actions": self.get_legal_actions() if (pid == self.current and self.phase != Phase.OVER) else [],
            "reveal_options": self.reveal_options(pid) if (self.phase == Phase.REVEAL and pid == self.current) else [],
        }
