"""Evaluation protocol (SPEC §5.7).

    python training/evaluate.py --a heuristic --b random --rounds 1000
    python training/evaluate.py --a models/checkpoints/runA/ckpt_latest.pt --b heuristic
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rlcard  # noqa: E402

import indian_rummy.rlcard_env  # noqa: E402,F401  (registers the env)
from indian_rummy import actions as A  # noqa: E402
from indian_rummy.agents import HeuristicAgent, RandomAgent  # noqa: E402


@dataclass
class EvalResult:
    rounds: int
    win_rate: float          # agent A round wins / rounds
    loss_rate: float
    mean_score_diff: float   # (B points - A points) per round, raw
    a_forfeit_rate: float
    mean_turns: float
    illegal: int
    rounds_per_sec: float

    def row(self) -> dict:
        return self.__dict__.copy()


class IllegalGuard:
    """Wraps an agent and counts any action outside the legal set (must stay 0)."""

    def __init__(self, agent):
        self.agent = agent
        self.use_raw = agent.use_raw
        self.illegal = 0

    def _check(self, state, a):
        if a not in state["legal_actions"]:
            self.illegal += 1
            a = next(iter(state["legal_actions"]))
        return a

    def step(self, state):
        return self._check(state, self.agent.step(state))

    def eval_step(self, state):
        a, info = self.agent.eval_step(state)
        return self._check(state, int(a)), info


def evaluate(agent_a, agent_b, rounds: int = 1000, seed: int = 12345, turn_cap: int | None = None) -> EvalResult:
    """Plays `rounds` rounds, swapping seats every round; fixed seeds for comparability."""
    cfg = {"seed": seed}
    if turn_cap:
        cfg["game_turn_cap"] = turn_cap
    env = rlcard.make("indian-rummy", config=cfg)
    ga, gb = IllegalGuard(agent_a), IllegalGuard(agent_b)
    wins = losses = forfeits = 0
    diff = 0.0
    turns = 0
    t0 = time.time()
    for k in range(rounds):
        seat_a = k % 2
        agents = [ga, gb] if seat_a == 0 else [gb, ga]
        env.set_agents(agents)
        env.run(is_training=False)
        res = env.game.result
        pa, pb = res.points[seat_a], res.points[1 - seat_a]
        wins += res.winner == seat_a
        losses += res.winner == 1 - seat_a
        forfeits += res.reason == "forfeit" and pa == env.variant.forfeit_points
        diff += pb - pa
        turns += env.game.turn
    dt = time.time() - t0
    return EvalResult(rounds, wins / rounds, losses / rounds, diff / rounds, forfeits / rounds,
                      turns / rounds, ga.illegal + gb.illegal, rounds / dt)


def load_agent(spec: str, device: str = "cpu"):
    if spec == "random":
        return RandomAgent(seed=1)
    if spec == "heuristic":
        return HeuristicAgent()
    if spec.startswith("heuristic:"):
        return HeuristicAgent(epsilon=float(spec.split(":")[1]), seed=2)
    from indian_rummy.agents.policy import DQNPolicy, HybridPolicy
    if spec.startswith("hybrid:"):
        return HybridPolicy(DQNPolicy.load(spec.split(":", 1)[1], device=device))
    return DQNPolicy.load(spec, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="heuristic")
    ap.add_argument("--b", default="random")
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    r = evaluate(load_agent(args.a), load_agent(args.b), args.rounds, args.seed)
    print(f"{args.a} vs {args.b}: " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                                for k, v in r.row().items()))


if __name__ == "__main__":
    main()
