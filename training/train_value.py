"""Train the value network on states logged during DQN training (SPEC §6.1).

    python training/train_value.py --data training/data/A --out models/value_model.pth
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from indian_rummy.agents.value import ValueNet  # noqa: E402
from indian_rummy.features import FEATURE_VERSION, OBS_DIM  # noqa: E402


def load_shards(dirs: list[Path]):
    obs, won, pay = [], [], []
    for d in dirs:
        for f in sorted(d.glob("value_*.npz")):
            z = np.load(f)
            obs.append(z["obs"])
            won.append(z["won"])
            pay.append(z["payoff"])
    return np.concatenate(obs), np.concatenate(won).astype(np.float32), np.concatenate(pay).astype(np.float32)


def ece(p: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    best_t, best = 1.0, None
    for t in np.linspace(0.5, 3.0, 51):
        p = 1 / (1 + np.exp(-logits / t))
        nll = -np.mean(y * np.log(p + 1e-7) + (1 - y) * np.log(1 - p + 1e-7))
        if best is None or nll < best:
            best_t, best = float(t), nll
    return best_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=[str(ROOT / "training" / "data" / "A")])
    ap.add_argument("--out", default=str(ROOT / "models" / "value_model.pth"))
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--layers", default="512,256")
    args = ap.parse_args()

    t0 = time.time()
    X, Y, P = load_shards([Path(d) for d in args.data])
    n = len(Y)
    print(f"{n} states, win rate {Y.mean():.3f}, loaded in {time.time() - t0:.0f}s")
    rng = np.random.RandomState(0)
    # Shards are in play order, so holding out the tail keeps a round's states on one side of the split
    # (a random per-state split leaks near-duplicate positions into validation and flatters the ECE).
    n_val = max(1000, n // 10)
    val, tr = np.arange(n - n_val, n), np.arange(n - n_val)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layers = tuple(int(x) for x in args.layers.split(","))
    net = ValueNet(OBS_DIM, layers).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    Xt = torch.from_numpy(X)            # float16 on CPU; batches moved to GPU
    Yt, Pt = torch.from_numpy(Y), torch.from_numpy(P)

    def batches(idx, shuffle):
        idx = rng.permutation(idx) if shuffle else idx
        for i in range(0, len(idx), args.batch):
            j = torch.from_numpy(idx[i:i + args.batch])
            yield Xt[j].float().to(dev), Yt[j].to(dev), Pt[j].to(dev)

    def evaluate_split(idx):
        net.eval()
        logits, ys, ps, preds = [], [], [], []
        with torch.no_grad():
            for x, y, p in batches(idx, False):
                lg, sc = net(x)
                logits.append(lg.cpu().numpy()); ys.append(y.cpu().numpy())
                ps.append(p.cpu().numpy()); preds.append(sc.cpu().numpy())
        return np.concatenate(logits), np.concatenate(ys), np.concatenate(ps), np.concatenate(preds)

    best = (float("inf"), -1, None)
    for ep in range(args.epochs):
        net.train()
        tot = 0.0
        for x, y, p in batches(tr, True):
            lg, sc = net(x)
            loss = F.binary_cross_entropy_with_logits(lg, y) + F.mse_loss(sc, p)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
        lg, y, p, sc = evaluate_split(val)
        prob = 1 / (1 + np.exp(-lg))
        acc = ((prob > 0.5) == (y > 0.5)).mean()
        val_loss = float(F.binary_cross_entropy_with_logits(torch.from_numpy(lg), torch.from_numpy(y)))
        print(f"epoch {ep}: train loss {tot / len(tr):.4f} | val bce {val_loss:.4f} acc {acc:.3f} "
              f"ece {ece(prob, y):.4f} score mse {np.mean((sc - p) ** 2):.4f}", flush=True)
        if val_loss < best[0]:  # keep the epoch that generalises best; the net overfits quickly
            best = (val_loss, ep, {k: v.detach().clone() for k, v in net.state_dict().items()})

    print(f"keeping epoch {best[1]} (val bce {best[0]:.4f})")
    net.load_state_dict(best[2])
    lg, y, p, sc = evaluate_split(val)
    e_before = ece(1 / (1 + np.exp(-lg)), y)
    T = fit_temperature(lg, y)
    e_after = ece(1 / (1 + np.exp(-lg / T)), y)
    meta = {"obs_dim": OBS_DIM, "layers": list(layers), "feature_version": FEATURE_VERSION,
            "temperature": T, "best_epoch": best[1], "ece_before": e_before, "ece_after": e_after, "n_train": int(len(tr)),
            "n_val": int(n_val), "val_win_rate": float(y.mean()), "data": args.data,
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.cpu() for k, v in net.state_dict().items()}, "meta": meta}, args.out)
    print(json.dumps(meta, indent=2))
    print("CALIBRATION GATE", "PASS" if e_after <= 0.05 else "FAIL")


if __name__ == "__main__":
    main()
