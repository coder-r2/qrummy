"""Uniform random legal-action agent (RLCard agent interface)."""

from __future__ import annotations

import numpy as np


class RandomAgent:
    use_raw = False

    def __init__(self, seed: int | None = None, allow_forfeit: bool = True):
        self.rng = np.random.RandomState(seed)
        self.allow_forfeit = allow_forfeit

    def step(self, state):
        legal = list(state["legal_actions"].keys())
        if not self.allow_forfeit and len(legal) > 1:
            from ..actions import FORFEIT
            legal = [a for a in legal if a != FORFEIT] or legal
        return int(legal[self.rng.randint(len(legal))])

    def eval_step(self, state):
        return self.step(state), {}
