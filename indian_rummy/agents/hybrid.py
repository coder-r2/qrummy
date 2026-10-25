"""DQN + heuristic hybrid policy (SPEC §5.8). Torch-free: works with DQNPolicy or NpDQNPolicy."""

from __future__ import annotations

import numpy as np

from .. import actions as A
from ..features import Analysis, encode
from .heuristic import HeuristicAgent


class HybridPolicy:
    """DQN chooses among the heuristic's top-k discards (always taking a winning discard);
    draw/forfeit decisions come from the DQN."""

    use_raw = False

    def __init__(self, dqn, k: int = 3):
        self.dqn = dqn
        self.k = k
        self.heuristic = HeuristicAgent()

    def act_analysis(self, an: Analysis, obs: np.ndarray | None = None) -> int:
        obs = encode(an) if obs is None else obs
        if an.decision == "discard" and an.discards:
            ranked = self.heuristic.ranked_discards(an)
            if an.discards[ranked[0]].win:
                return A.discard_action(ranked[0])
            cands = [A.discard_action(t) for t in ranked[:self.k]]
            if A.FORFEIT in an.legal:
                cands.append(A.FORFEIT)
            qs = self.dqn.masked_q(obs, cands)
            return max(qs, key=qs.get)
        qs = self.dqn.masked_q(obs, an.legal)
        return max(qs, key=qs.get)

    def eval_step(self, state):
        return self.act_analysis(state["raw_obs"]["analysis"], state["obs"]), {}

    def step(self, state):
        return self.eval_step(state)[0]
