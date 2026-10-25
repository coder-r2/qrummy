"""RLCard environment wrapper (SPEC §5.1).

    import indian_rummy.rlcard_env  # registers "indian-rummy"
    env = rlcard.make("indian-rummy", config={"seed": 0})
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace

import numpy as np
from rlcard.envs import Env
from rlcard.envs.registration import register, registry

from . import actions as A
from .agents.heuristic import heuristic_reveal
from .config import VARIANTS
from .features import OBS_DIM, analyse, encode
from .game import RummyGame


class IndianRummyEnv(Env):
    def __init__(self, config):
        self.name = "indian-rummy"
        self.default_game_config = {}
        self.variant = VARIANTS[config.get("game_variant", "v13")]
        if config.get("game_turn_cap"):
            self.variant = replace(self.variant, turn_cap=int(config["game_turn_cap"]))
        # Reveals are not RL decisions: both seats use the heuristic reveal policy (D-033, D-055).
        self.game = RummyGame(self.variant, num_players=2,
                              reveal_policy={0: heuristic_reveal, 1: heuristic_reveal})
        super().__init__(config)
        self.state_shape = [[OBS_DIM] for _ in range(self.num_players)]
        self.action_shape = [None for _ in range(self.num_players)]

    def _extract_state(self, state):
        acting = state["current_player"] == state["player"] and state["phase"] != "over"
        an = analyse(state, self.game.types, self.variant, light=not acting)
        legal = OrderedDict((a, None) for a in state["legal_actions"]) if acting else OrderedDict()
        raw = dict(state)
        raw["analysis"] = an
        return {
            "obs": encode(an),
            "legal_actions": legal,
            "raw_obs": raw,
            "raw_legal_actions": [A.describe(a) for a in legal],
        }

    def get_payoffs(self):
        return np.array(self.game.get_payoffs(), dtype=np.float32)

    def _decode_action(self, action_id):
        return int(action_id)

    def _get_legal_actions(self):
        return self.game.get_legal_actions()


if "indian-rummy" not in registry.env_specs:
    register(env_id="indian-rummy", entry_point="indian_rummy.rlcard_env:IndianRummyEnv")
