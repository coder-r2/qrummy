"""Export trained torch models to the NumPy format the backend serves (D-085).

    python training/export_models.py --dqn models/checkpoints/A/latest.pt --out models/dqn_model.npz
    python training/export_models.py --value models/value_model.pth --out models/value_model.npz

Verifies the export against the torch model on random and real-looking inputs (max |diff| < 1e-4).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indian_rummy.agents.np_models import NpDQNPolicy, NpValueModel  # noqa: E402
from indian_rummy.agents.policy import DQNPolicy  # noqa: E402
from indian_rummy.agents.value import ValueModel  # noqa: E402


def sequential_ops(seq: nn.Sequential, prefix: str, arrays: dict) -> list[dict]:
    ops = []
    for i, m in enumerate(seq):
        name = f"{prefix}{i}"
        if isinstance(m, nn.Flatten):
            continue
        if isinstance(m, nn.BatchNorm1d):
            for k, v in (("w", m.weight), ("b", m.bias), ("mean", m.running_mean), ("var", m.running_var)):
                arrays[f"{name}_{k}"] = v.detach().cpu().numpy().astype(np.float32)
            ops.append({"op": "bn", "eps": m.eps, **{k: f"{name}_{k}" for k in ("w", "b", "mean", "var")}})
        elif isinstance(m, nn.Linear):
            arrays[f"{name}_w"] = m.weight.detach().cpu().numpy().astype(np.float32)
            arrays[f"{name}_b"] = m.bias.detach().cpu().numpy().astype(np.float32)
            ops.append({"op": "linear", "w": f"{name}_w", "b": f"{name}_b"})
        elif isinstance(m, nn.Tanh):
            ops.append({"op": "tanh"})
        elif isinstance(m, nn.ReLU):
            ops.append({"op": "relu"})
        else:
            raise TypeError(f"unsupported layer {type(m).__name__}")
    return ops


def save(out: str, kind: str, meta: dict, ops: list[dict], arrays: dict):
    spec = json.dumps({"kind": kind, "meta": meta, "ops": ops})
    np.savez_compressed(out, spec=np.array(spec), **arrays)


def probe_inputs(dim: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.concatenate([rng.random((64, dim)), (rng.random((64, dim)) < 0.05).astype(float)]).astype(np.float32)


def export_dqn(path: str, out: str):
    pol = DQNPolicy.load(path)
    arrays: dict = {}
    ops = sequential_ops(pol.qnet.fc_layers, "q", arrays)
    meta = {k: v for k, v in pol.meta.items() if isinstance(v, (int, float, str, list, bool, type(None)))}
    save(out, "dqn", meta, ops, arrays)
    x = probe_inputs(pol.meta["obs_dim"])
    diff = np.abs(pol.q_values(x) - NpDQNPolicy.load(out).q_values(x)).max()
    print(f"dqn -> {out}  max|diff| = {diff:.2e}")
    assert diff < 1e-4


def export_value(path: str, out: str):
    vm = ValueModel.load(path)
    arrays: dict = {}
    ops = sequential_ops(vm.net.body, "v", arrays)
    for head in ("win", "score"):
        lin = getattr(vm.net, f"{head}_head")
        arrays[f"{head}_w"] = lin.weight.detach().cpu().numpy().astype(np.float32)
        arrays[f"{head}_b"] = lin.bias.detach().cpu().numpy().astype(np.float32)
    meta = {k: v for k, v in vm.meta.items() if isinstance(v, (int, float, str, list, bool, type(None)))}
    meta["temperature"] = float(vm.temperature)
    save(out, "value", meta, ops, arrays)
    x = probe_inputs(vm.meta["obs_dim"])
    (p1, s1), (p2, s2) = vm.predict(x), NpValueModel.load(out).predict(x)
    diff = max(np.abs(p1 - p2).max(), np.abs(s1 - s2).max())
    print(f"value -> {out}  max|diff| = {diff:.2e}")
    assert diff < 1e-4


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dqn")
    g.add_argument("--value")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with torch.no_grad():
        export_dqn(args.dqn, args.out) if args.dqn else export_value(args.value, args.out)


if __name__ == "__main__":
    main()
