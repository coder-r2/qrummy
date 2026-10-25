"""Probe: how well can RLCard's Q-network imitate the heuristic? (informs D-059 settings)

    python training/imitation_probe.py --demos 40000 --steps 3000,10000,20000
Saves the demo set to training/data/demos.npz for reuse.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rlcard  # noqa: E402

import indian_rummy.rlcard_env  # noqa: E402,F401
from indian_rummy.agents import HeuristicAgent  # noqa: E402
from training.evaluate import evaluate  # noqa: E402
from training.train_dqn import CONFIG, QRummyDQNAgent, frozen, pretrain, warm_start  # noqa: E402


def agreement(agent, demos, idx) -> float:
    qnet = agent.q_estimator.qnet
    with torch.no_grad():
        s = torch.from_numpy(demos["s"][idx]).float().to(agent.device)
        q = qnet(s).cpu().numpy()
    q[~demos["m"][idx]] = -np.inf
    return float((q.argmax(1) == demos["a"][idx]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", type=int, default=40000)
    ap.add_argument("--steps", default="3000,10000,20000")
    ap.add_argument("--layers", default="512,512,256")
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()
    cfg = dict(CONFIG)
    cfg["mlp_layers"] = [int(x) for x in args.layers.split(",")]
    env = rlcard.make("indian-rummy", config={"seed": 3, "game_turn_cap": 120})
    path = ROOT / "training" / "data" / "demos.npz"

    def new_agent():
        return QRummyDQNAgent(replay_memory_size=10_000, replay_memory_init_size=100, batch_size=256,
                              num_actions=env.num_actions, state_shape=env.state_shape[0],
                              mlp_layers=cfg["mlp_layers"], learning_rate=args.lr,
                              update_target_estimator_every=2000, device=torch.device("cuda"))

    if path.exists():
        demos = dict(np.load(path))
    else:
        t0 = time.time()
        demos = warm_start(env, new_agent(), args.demos, 0.99, 0.5)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **demos)
        print(f"collected {len(demos['a'])} demos in {time.time() - t0:.0f}s")
    n = len(demos["a"])
    held = np.arange(n - 5000, n)
    train = {k: v[: n - 5000] for k, v in demos.items()}
    is_discard = (demos["a"][held] >= 2) & (demos["a"][held] < 55)
    for steps in [int(s) for s in args.steps.split(",")]:
        agent = new_agent()
        t0 = time.time()
        pretrain(agent, train, steps, 0.8, 1.0, 0.99)
        acc = agreement(agent, demos, held)
        acc_d = agreement(agent, demos, held[is_discard])
        r = evaluate(frozen(agent, cfg), HeuristicAgent(), 200, seed=777, turn_cap=120)
        print(f"steps={steps} layers={cfg['mlp_layers']} lr={args.lr}: agree {acc:.3f} (discards {acc_d:.3f}) "
              f"| vs heuristic win {r.win_rate:.3f} diff {r.mean_score_diff:+.1f} | {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
