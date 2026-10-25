"""In-memory game sessions (SPEC §7.2, D-041)."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import OrderedDict

from indian_rummy import actions as A
from indian_rummy.agents.heuristic import heuristic_reveal
from indian_rummy.game import IllegalAction, Phase
from indian_rummy.match import Match

from .ai import AIPlayer, Models
from .views import AI, HUMAN, event_json

SESSION_TTL_S = 2 * 3600
MAX_SESSIONS = 500


class GameSession:
    def __init__(self, models: Models, mode: str, difficulty: str, coach: str, seed: int | None = None):
        self.id = uuid.uuid4().hex
        self.mode, self.difficulty, self.coach = mode, difficulty, coach
        multipliers = None if mode == "match" else (1,)
        kw = {"seed": seed, "reveal_policy": {AI: heuristic_reveal}}   # the human reveals explicitly
        if multipliers:
            kw["multipliers"] = multipliers
        self.match = Match(**kw)
        self.ai = AIPlayer(models, difficulty, seed=seed)
        self.lock = asyncio.Lock()
        self.last_access = time.time()
        self.cursor = 0
        self.round_multiplier = 1
        self.last_facts: dict | None = None
        self._start_round()

    @property
    def game(self):
        return self.match.game

    def _start_round(self):
        self.round_multiplier = self.match.multipliers[len(self.match.rounds)]
        self.match.start_round()
        self.cursor = 0
        self._recorded = False
        self._run_ai()

    def _run_ai(self):
        g = self.game
        guard = 0
        while not g.is_over() and g.current == AI:
            g.step(self.ai.act(g, AI))
            guard += 1
            if guard > 2000:
                raise RuntimeError("AI loop did not terminate")
        if g.is_over() and not self._recorded:
            self.match.finish_round()
            self._recorded = True

    def take_events(self) -> list[dict]:
        evs = self.game.events[self.cursor:]
        self.cursor = len(self.game.events)
        return [event_json(self.game, e) for e in evs]

    def apply(self, action: dict) -> None:
        """Apply one human action, then let the AI play until the human must act again."""
        g = self.game
        kind = action.get("type")
        if kind == "next_round":
            if not g.is_over() or self.match.is_over():
                raise IllegalAction("no next round")
            self._start_round()
            return
        if g.is_over() or g.current != HUMAN:
            raise IllegalAction("not your turn")
        if kind == "draw":
            src = action.get("source")
            g.step(A.DRAW_CLOSED if src == "closed" else A.DRAW_OPEN if src == "open" else -1)
        elif kind == "discard":
            g.step(("discard", int(action["card_id"])))
        elif kind == "reveal":
            if g.phase != Phase.REVEAL:
                raise IllegalAction("no reveal pending")
            g.step(("reveal", tuple(int(c) for c in action["card_ids"])))
        elif kind == "forfeit":
            g.step(A.FORFEIT)
        else:
            raise IllegalAction(f"unknown action {kind!r}")
        self._run_ai()


class SessionStore:
    def __init__(self, models: Models):
        self.models = models
        self.sessions: OrderedDict[str, GameSession] = OrderedDict()

    def create(self, mode: str, difficulty: str, coach: str, seed: int | None = None) -> GameSession:
        self.expire()
        s = GameSession(self.models, mode, difficulty, coach, seed)
        self.sessions[s.id] = s
        while len(self.sessions) > MAX_SESSIONS:
            self.sessions.popitem(last=False)              # least recently used
        return s

    def get(self, sid: str) -> GameSession | None:
        s = self.sessions.get(sid)
        if s is not None:
            s.last_access = time.time()
            self.sessions.move_to_end(sid)
        return s

    def delete(self, sid: str) -> None:
        self.sessions.pop(sid, None)

    def expire(self) -> int:
        now = time.time()
        dead = [k for k, s in self.sessions.items() if now - s.last_access > SESSION_TTL_S]
        for k in dead:
            del self.sessions[k]
        return len(dead)
