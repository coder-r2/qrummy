"""Win-probability / expected-score network (SPEC §6.1), used by the coach and forfeit logic."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..features import FEATURE_VERSION


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, layers=(512, 256)):
        super().__init__()
        dims = [obs_dim, *layers]
        body = []
        for i in range(len(dims) - 1):
            body += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        self.body = nn.Sequential(*body)
        self.win_head = nn.Linear(dims[-1], 1)     # logit of P(win round)
        self.score_head = nn.Linear(dims[-1], 1)   # tanh -> payoff in [-1, 1] (§5.4)

    def forward(self, x):
        h = self.body(x)
        return self.win_head(h).squeeze(-1), torch.tanh(self.score_head(h)).squeeze(-1)


class ValueModel:
    """Inference wrapper with temperature-scaled probabilities."""

    def __init__(self, net: ValueNet, temperature: float = 1.0, meta: dict | None = None, device: str = "cpu"):
        self.net = net.to(device).eval()
        self.temperature = temperature
        self.meta = meta or {}
        self.device = device

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "ValueModel":
        ck = torch.load(path, map_location=device, weights_only=False)
        meta = ck["meta"]
        if meta.get("feature_version") != FEATURE_VERSION:
            raise ValueError("value model feature_version mismatch")
        net = ValueNet(meta["obs_dim"], tuple(meta["layers"]))
        net.load_state_dict(ck["state_dict"])
        return cls(net, meta.get("temperature", 1.0), meta, device)

    def predict(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """obs (B, D) or (D,) -> (p_win, expected payoff)."""
        single = obs.ndim == 1
        x = torch.from_numpy(np.atleast_2d(obs).astype(np.float32)).to(self.device)
        with torch.no_grad():
            logit, score = self.net(x)
            p = torch.sigmoid(logit / self.temperature)
        p, score = p.cpu().numpy(), score.cpu().numpy()
        return (p[0], score[0]) if single else (p, score)
