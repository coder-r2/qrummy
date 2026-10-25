"""Torch-free inference for the served models (D-085).

The backend runs on a 512 MB / 0.1-CPU free instance, so it loads plain NumPy copies of the
trained networks instead of importing torch. `training/export_models.py` writes them; the
forward passes below reproduce RLCard's EstimatorNetwork (BatchNorm -> [Linear, Tanh]* -> Linear)
and ValueNet ([Linear, ReLU]* -> win/score heads) exactly, in eval mode.

File format (.npz): "spec" = JSON {"kind", "meta", "ops": [{"op": "bn"|"linear"|"tanh"|"relu", ...}]}
plus the arrays named in each op.
"""

from __future__ import annotations

import json

import numpy as np

from .. import actions as A
from ..features import FEATURE_VERSION, Analysis, encode


def _load(path: str) -> tuple[dict, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as z:
        spec = json.loads(str(z["spec"]))
        arrays = {k: z[k].astype(np.float32) for k in z.files if k != "spec"}
    if spec["meta"].get("feature_version") != FEATURE_VERSION:
        raise ValueError(f"{path}: feature_version {spec['meta'].get('feature_version')} != {FEATURE_VERSION}")
    return spec, arrays


def _run(ops: list[dict], arrays: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    for op in ops:
        kind = op["op"]
        if kind == "bn":
            x = (x - arrays[op["mean"]]) / np.sqrt(arrays[op["var"]] + op["eps"]) * arrays[op["w"]] + arrays[op["b"]]
        elif kind == "linear":
            x = x @ arrays[op["w"]].T + arrays[op["b"]]
        elif kind == "tanh":
            x = np.tanh(x)
        elif kind == "relu":
            x = np.maximum(x, 0.0)
        else:
            raise ValueError(f"unknown op {kind}")
    return x


class NpDQNPolicy:
    """Same interface as policy.DQNPolicy (q_values / masked_q / act_analysis / eval_step)."""

    use_raw = False

    def __init__(self, spec: dict, arrays: dict[str, np.ndarray]):
        self.ops, self.arrays, self.meta = spec["ops"], arrays, spec["meta"]

    @classmethod
    def load(cls, path: str) -> "NpDQNPolicy":
        spec, arrays = _load(path)
        if spec["kind"] != "dqn":
            raise ValueError(f"{path} is not a DQN export")
        return cls(spec, arrays)

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        single = obs.ndim == 1
        q = _run(self.ops, self.arrays, np.atleast_2d(obs).astype(np.float32))
        return q[0] if single else q

    def masked_q(self, obs: np.ndarray, legal) -> dict[int, float]:
        q = self.q_values(obs)
        return {int(a): float(q[a]) for a in legal}

    def act_analysis(self, an: Analysis) -> int:
        qs = self.masked_q(encode(an), an.legal)
        return max(qs, key=qs.get)

    def eval_step(self, state):
        qs = self.masked_q(state["obs"], state["legal_actions"].keys())
        best = max(qs, key=qs.get)
        return best, {"values": {A.describe(a): v for a, v in qs.items()}}

    def step(self, state):
        return self.eval_step(state)[0]


class NpValueModel:
    """Same interface as value.ValueModel.predict: obs -> (p_win, expected payoff)."""

    def __init__(self, spec: dict, arrays: dict[str, np.ndarray]):
        self.ops, self.arrays, self.meta = spec["ops"], arrays, spec["meta"]
        self.temperature = float(self.meta.get("temperature", 1.0))

    @classmethod
    def load(cls, path: str) -> "NpValueModel":
        spec, arrays = _load(path)
        if spec["kind"] != "value":
            raise ValueError(f"{path} is not a value-model export")
        return cls(spec, arrays)

    def predict(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        single = obs.ndim == 1
        h = _run(self.ops, self.arrays, np.atleast_2d(obs).astype(np.float32))
        a = self.arrays
        logit = (h @ a["win_w"].T + a["win_b"])[:, 0]
        score = np.tanh(h @ a["score_w"].T + a["score_b"])[:, 0]
        p = 1.0 / (1.0 + np.exp(-logit / self.temperature))
        return (p[0], score[0]) if single else (p, score)
