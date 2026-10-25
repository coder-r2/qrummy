"""Inference-only DQN policy and the DQN+heuristic hybrid (SPEC §5.8).

The Q-network is RLCard's own EstimatorNetwork, so checkpoints load bit-exact (D-043).
Checkpoint format (torch.save dict): {"qnet": state_dict, "meta": {obs_dim, mlp_layers,
num_actions, feature_version, ...}}.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
from rlcard.agents.dqn_agent import EstimatorNetwork

from .. import actions as A
from ..features import FEATURE_VERSION, Analysis, encode
from .hybrid import HybridPolicy  # noqa: F401  (re-exported; it lives in a torch-free module)


class DQNPolicy:
    use_raw = False

    def __init__(self, qnet: EstimatorNetwork, meta: dict, device: str = "cpu"):
        self.qnet = qnet.to(device).eval()
        self.meta = meta
        self.device = device

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "DQNPolicy":
        ck = torch.load(path, map_location=device, weights_only=False)
        meta = ck["meta"]
        if meta.get("feature_version") != FEATURE_VERSION:
            raise ValueError(f"checkpoint feature_version {meta.get('feature_version')} != {FEATURE_VERSION}")
        net = EstimatorNetwork(meta["num_actions"], [meta["obs_dim"]], meta["mlp_layers"])
        net.load_state_dict(ck["qnet"])
        return cls(net, meta, device)

    @classmethod
    def from_estimator(cls, qnet: EstimatorNetwork, meta: dict) -> "DQNPolicy":
        """Frozen CPU copy of a live network (opponent pool snapshots)."""
        net = EstimatorNetwork(meta["num_actions"], [meta["obs_dim"]], meta["mlp_layers"])
        net.load_state_dict({k: v.detach().cpu() for k, v in qnet.state_dict().items()})
        return cls(net, meta, "cpu")

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        """obs: (D,) or (B, D) -> (num_actions,) or (B, num_actions)."""
        single = obs.ndim == 1
        x = torch.from_numpy(np.atleast_2d(obs).astype(np.float32)).to(self.device)
        with torch.no_grad():
            q = self.qnet(x).cpu().numpy()
        return q[0] if single else q

    def masked_q(self, obs: np.ndarray, legal) -> dict[int, float]:
        q = self.q_values(obs)
        return {int(a): float(q[a]) for a in legal}

    def act_analysis(self, an: Analysis) -> int:
        qs = self.masked_q(encode(an), an.legal)
        return max(qs, key=qs.get)

    # RLCard interface
    def eval_step(self, state):
        qs = self.masked_q(state["obs"], state["legal_actions"].keys())
        best = max(qs, key=qs.get)
        return best, {"values": {A.describe(a): v for a, v in qs.items()}}

    def step(self, state):
        return self.eval_step(state)[0]


def legal_dict(legal) -> OrderedDict:
    return OrderedDict((int(a), None) for a in legal)
