"""Heuristic bot (SPEC §5.6): curriculum opponent, Easy/Medium base and Hard fallback.

Works on a features.Analysis, so it shares all solver work with the observation encoder.
"""

from __future__ import annotations

import numpy as np

from .. import actions as A
from ..cards import JOKER, base_points, make_type
from ..features import Analysis
from ..solver import HandContext, solve


def heuristic_reveal(game, pid: int, options: list[tuple[int, ...]]) -> tuple[int, ...]:
    """Choose which original to reveal: minimise expected deadwood over the (still unknown) wild
    rank, weighting each rank by how many of its cards this player can't see. Prefers 3 cards (A4)."""
    opts = [o for o in options if len(o) == 3] or list(options)
    if len(opts) == 1:
        return opts[0]
    v = game.variant
    counts = game.counts(pid)
    weights = [v.decks * 4 - sum(counts[make_type(s, r)] for s in range(4)) for r in range(1, 14)]
    best, best_val = opts[0], None
    for o in opts:
        locked = tuple(sorted(game.type_of(c) for c in o))
        val = 0.0
        for r in range(1, 14):
            if weights[r - 1] <= 0:
                continue
            ctx = HandContext(post_original=True, wild_rank=r, locked=locked,
                              template=v.template, triplets=v.triplets_possible)
            val += weights[r - 1] * solve(counts, ctx).points
        if best_val is None or val < best_val:
            best, best_val = o, val
    return best


class HeuristicAgent:
    """RLCard-compatible agent; reads state['raw_obs']['analysis']."""

    use_raw = False

    def __init__(self, epsilon: float = 0.0, allow_forfeit: bool = True, seed: int | None = None):
        self.epsilon = epsilon
        self.allow_forfeit = allow_forfeit
        self.rng = np.random.RandomState(seed)

    # -- policy ---------------------------------------------------------------
    def should_forfeit(self, an: Analysis) -> bool:
        if not self.allow_forfeit or not an.state["forfeit_window"]:
            return False
        return (not an.has_original) and an.original_outs <= 2 and an.counts[JOKER] == 0

    def discard_score(self, an: Analysis, t: int) -> float:
        d = an.discards[t]
        if d.win:
            return -1e9
        wild_penalty = 100.0 if an.is_wild(t) else 0.0
        if an.post_original:
            # Uncovered cards matter even at 0 points: a win needs an exact cover (R6).
            return d.points + 1.5 * d.uncovered - 0.8 * d.outs + 0.05 * base_points(t) + wild_penalty
        return (0.0 if d.has_original else 15.0) - 0.8 * d.outs + 0.3 * d.points \
            + 0.05 * base_points(t) + wild_penalty

    def ranked_discards(self, an: Analysis) -> list[int]:
        return sorted(an.discards, key=lambda t: self.discard_score(an, t))

    def act(self, an: Analysis) -> int:
        legal = an.legal
        if self.epsilon and self.rng.rand() < self.epsilon:
            choices = [a for a in legal if a != A.FORFEIT] or legal
            return int(choices[self.rng.randint(len(choices))])
        if an.decision == "draw":
            if A.FORFEIT in legal and self.should_forfeit(an):
                return A.FORFEIT
            if A.DRAW_OPEN in legal:
                if not an.post_original or an.open_win:
                    return A.DRAW_OPEN
                # Never take back a card we threw away earlier: breaks pick-up/throw-back cycles.
                if an.top in an.my_discard_types:
                    return A.DRAW_CLOSED if A.DRAW_CLOSED in legal else A.DRAW_OPEN
                if an.open_best_points is not None and an.current.points - an.open_best_points >= 3:
                    return A.DRAW_OPEN
            return A.DRAW_CLOSED if A.DRAW_CLOSED in legal else int(legal[0])
        if an.decision == "discard" and an.discards:
            return A.discard_action(self.ranked_discards(an)[0])
        return int(legal[0])

    # -- RLCard interface -----------------------------------------------------
    def step(self, state) -> int:
        return self.act(state["raw_obs"]["analysis"])

    def eval_step(self, state):
        return self.step(state), {}
