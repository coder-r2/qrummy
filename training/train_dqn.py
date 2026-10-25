"""DQN self-play training (SPEC §5.4-§5.5).

    python training/train_dqn.py --run A --hours 8
    python training/train_dqn.py --run A --hours 4 --resume models/checkpoints/A/latest.pt

Outputs
    models/checkpoints/<run>/latest.pt, ckpt_<rounds>.pt   Q-net + optimizer + counters
    training/logs/<run>/train.csv, eval.csv               progress
    training/data/<run>/value_<k>.npz                     sampled states for the value net (§6.1)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rlcard  # noqa: E402
from rlcard.agents.dqn_agent import DQNAgent, Memory, Transition  # noqa: E402

import indian_rummy.rlcard_env  # noqa: E402,F401
from indian_rummy.actions import NUM_ACTIONS  # noqa: E402
from indian_rummy.agents import HeuristicAgent, RandomAgent  # noqa: E402
from indian_rummy.agents.policy import DQNPolicy, HybridPolicy  # noqa: E402
from indian_rummy.features import FEATURE_VERSION, OBS_DIM  # noqa: E402
from training.evaluate import evaluate  # noqa: E402

CONFIG = dict(
    mlp_layers=[512, 512, 256],
    learning_rate=5e-5,              # run A used 1e-4 and diverged (D-091)
    td_target_clip=True,             # clip TD targets to the bounded true return ±(1 + shaping_coef) (D-091)
    huber_loss=True,                 # smooth-L1 instead of RLCard's MSE (D-091)
    batch_size=256,
    replay_memory_size=300_000,
    replay_memory_init_size=1_000,   # buffer is pre-filled with heuristic play (warm start)
    warm_start_transitions=60_000,
    pretrain_steps=6_000,            # DQfD-style: TD + large-margin imitation of the heuristic (D-059)
    pretrain_margin=0.8,
    pretrain_lambda=1.0,
    demo_lambda_start=1.0,           # DQfD margin loss kept during RL (D-065) ...
    demo_lambda_end=0.5,             # ... decayed linearly (run A: 0.1) ...
    demo_lambda_decay_frac=0.6,      # ... over the first 60% of the budget
    update_target_estimator_every=2_000,
    discount_factor=0.99,
    epsilon_start=0.15,              # low: pretraining already gives a competent policy
    epsilon_end=0.03,
    epsilon_decay_steps=300_000,
    train_every=4,
    turn_cap=120,                    # training-only round cap (A5 is 300 for real games)
    selfplay_gate=0.40,              # self-play only once eval win rate vs heuristic >= this
    shaping_coef=0.5,
    shaping_anneal_frac=0.3,         # anneal shaping to 0 over the final 30% of the budget
    stage_a_frac=0.25,               # first 25%: vs heuristic only
    pool=dict(self=0.5, heuristic=0.3, snapshot=0.2),
    snapshot_every_s=3600,
    max_snapshots=6,
    eval_every_rounds=5_000,
    eval_rounds_heuristic=400,
    eval_rounds_random=100,
    latest_every_rounds=2_000,
    ckpt_every_s=1800,
    value_sample_prob=1 / 6,
    value_start_frac=0.3,
    value_shard_size=100_000,
)


# --------------------------------------------------------------------------- agent

class RingMemory(Memory):
    """O(1) replay buffer storing observations as float16 (halves RAM)."""

    def __init__(self, memory_size, batch_size):
        super().__init__(memory_size, batch_size)
        self.pos = 0

    def save(self, state, action, reward, next_state, legal_actions, done):
        t = Transition(np.asarray(state, dtype=np.float16), int(action), float(reward),
                       np.asarray(next_state, dtype=np.float16), bool(done), tuple(legal_actions))
        if len(self.memory) < self.memory_size:
            self.memory.append(t)
        else:
            self.memory[self.pos] = t
        self.pos = (self.pos + 1) % self.memory_size

    def sample(self):
        s, a, r, ns, d, la = zip(*random.sample(self.memory, self.batch_size))
        return (np.asarray(s, dtype=np.float32), np.asarray(a), np.asarray(r, dtype=np.float32),
                np.asarray(ns, dtype=np.float32), np.asarray(d, dtype=bool), la)


def large_margin_loss(q: torch.Tensor, a: torch.Tensor, legal: torch.Tensor, margin: float) -> torch.Tensor:
    """DQfD supervised loss: max_a'[Q(s,a') + l(a_E,a')] - Q(s,a_E), over legal actions."""
    qa = q.gather(1, a[:, None]).squeeze(1)
    bonus = margin * (1 - torch.nn.functional.one_hot(a, q.shape[1]).float())
    return ((q + bonus).masked_fill(~legal, -1e9).max(1).values - qa).mean()


class QRummyDQNAgent(DQNAgent):
    """RLCard DQNAgent with a ring buffer and without per-step console output.

    train() is RLCard's Double-DQN update. When demonstrations are attached (D-065), the same
    optimizer step also adds λ·large-margin loss on a demo batch (DQfD); λ = 0 gives RLCard's
    update exactly."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.memory = RingMemory(kw["replay_memory_size"], kw["batch_size"])
        self.losses: list[float] = []
        self.demo: dict | None = None
        self.demo_lambda = 0.0
        self.demo_margin = 0.8
        self.td_clip: float | None = None   # |target| bound; None = RLCard's unclipped update
        self.huber = False

    def attach_demos(self, demos: dict, margin: float):
        dev = self.device
        self.demo = {"s": torch.from_numpy(demos["s"]).to(dev), "a": torch.from_numpy(demos["a"]).long().to(dev),
                     "m": torch.from_numpy(demos["m"]).to(dev)}
        self.demo_margin = margin

    def train(self):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, legal_actions_batch = self.memory.sample()
        q_values_next = self.q_estimator.predict_nograd(next_state_batch)
        legal_actions = []
        for b in range(self.batch_size):
            legal_actions.extend([i + b * self.num_actions for i in legal_actions_batch[b]])
        masked_q_values = -np.inf * np.ones(self.num_actions * self.batch_size, dtype=float)
        masked_q_values[legal_actions] = q_values_next.flatten()[legal_actions]
        masked_q_values = masked_q_values.reshape((self.batch_size, self.num_actions))
        best_actions = np.argmax(masked_q_values, axis=1)
        q_values_next_target = self.target_estimator.predict_nograd(next_state_batch)
        target_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
            self.discount_factor * q_values_next_target[np.arange(self.batch_size), best_actions]
        if self.td_clip is not None:
            target_batch = np.clip(target_batch, -self.td_clip, self.td_clip)

        est = self.q_estimator
        dev = self.device
        est.optimizer.zero_grad()
        est.qnet.train()
        s = torch.from_numpy(np.array(state_batch)).float().to(dev)
        a = torch.from_numpy(action_batch).long().to(dev)
        y = torch.from_numpy(target_batch.astype(np.float32)).to(dev)
        q = est.qnet(s).gather(-1, a.unsqueeze(-1)).squeeze(-1)
        batch_loss = torch.nn.functional.smooth_l1_loss(q, y) if self.huber else est.mse_loss(q, y)
        if self.demo is not None and self.demo_lambda > 0:
            idx = torch.randint(len(self.demo["a"]), (self.batch_size,), device=dev)
            dq = est.qnet(self.demo["s"][idx].float())
            batch_loss = batch_loss + self.demo_lambda * large_margin_loss(
                dq, self.demo["a"][idx], self.demo["m"][idx], self.demo_margin)
        batch_loss.backward()
        est.optimizer.step()
        est.qnet.eval()
        loss = batch_loss.item()
        self.losses.append(loss)
        if self.train_t % self.update_target_estimator_every == 0:
            self.target_estimator.qnet.load_state_dict(self.q_estimator.qnet.state_dict())
        self.train_t += 1
        return loss


def meta_dict(cfg) -> dict:
    return {"obs_dim": OBS_DIM, "mlp_layers": cfg["mlp_layers"], "num_actions": NUM_ACTIONS,
            "feature_version": FEATURE_VERSION}


def save_ckpt(agent: QRummyDQNAgent, path: Path, cfg, extra: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = meta_dict(cfg) | extra | {"total_t": agent.total_t, "train_t": agent.train_t,
                                     "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    tmp = path.with_suffix(".tmp")
    torch.save({"qnet": agent.q_estimator.qnet.state_dict(),
                "optimizer": agent.q_estimator.optimizer.state_dict(), "meta": meta}, tmp)
    tmp.replace(path)


def frozen(agent: QRummyDQNAgent, cfg) -> DQNPolicy:
    return DQNPolicy.from_estimator(agent.q_estimator.qnet, meta_dict(cfg))


# --------------------------------------------------------------------------- shaping

def reorganize_shaped(trajectories, payoffs, gamma: float, coef: float):
    """RLCard's reorganize plus potential-based shaping F = γΦ(s') − Φ(s), Φ(terminal) = 0."""
    out = [[] for _ in trajectories]
    for p, traj in enumerate(trajectories):
        n = len(traj)
        for i in range(0, n - 2, 2):
            s, a, s2 = traj[i], traj[i + 1], traj[i + 2]
            done = i == n - 3
            phi_s = s["raw_obs"]["analysis"].phi
            phi_s2 = 0.0 if done else s2["raw_obs"]["analysis"].phi
            r = (float(payoffs[p]) if done else 0.0) + coef * (gamma * phi_s2 - phi_s)
            out[p].append([s, a, r, s2, done])
    return out


# --------------------------------------------------------------------------- value-net data

class ValueLogger:
    def __init__(self, out_dir: Path, shard: int, prob: float, rng: random.Random):
        self.dir = out_dir
        self.shard = shard
        self.prob = prob
        self.rng = rng
        self.obs, self.won, self.payoff = [], [], []
        self.k = len(list(out_dir.glob("value_*.npz"))) if out_dir.exists() else 0

    def add_round(self, trajectories, game):
        res = game.result
        payoffs = game.get_payoffs()
        for p, traj in enumerate(trajectories):
            for i in range(0, len(traj) - 1, 2):
                if self.rng.random() < self.prob:
                    self.obs.append(traj[i]["obs"].astype(np.float16))
                    self.won.append(res.winner == p)
                    self.payoff.append(payoffs[p])
        if len(self.obs) >= self.shard:
            self.flush()

    def flush(self):
        if not self.obs:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.dir / f"value_{self.k:03d}.npz", obs=np.stack(self.obs),
                            won=np.array(self.won, dtype=bool), payoff=np.array(self.payoff, dtype=np.float32))
        self.k += 1
        self.obs, self.won, self.payoff = [], [], []


# --------------------------------------------------------------------------- main loop

def _mask(legal) -> np.ndarray:
    m = np.zeros(NUM_ACTIONS, dtype=bool)
    m[list(legal)] = True
    return m


def warm_start(env, agent: QRummyDQNAgent, n: int, gamma: float, coef: float) -> dict:
    """Heuristic-vs-heuristic play: fills the replay buffer and returns demonstration arrays."""
    h = [HeuristicAgent(epsilon=0.05, seed=11), HeuristicAgent(epsilon=0.05, seed=12)]
    env.set_agents(h)
    S, A_, R, S2, D, M, M2 = [], [], [], [], [], [], []
    while len(S) < n:
        traj, payoffs = env.run(is_training=True)
        for seat in reorganize_shaped(traj, payoffs, gamma, coef):
            for s, a, r, s2, done in seat:
                nxt = list(s2["legal_actions"].keys())
                agent.feed_memory(s["obs"], a, r, s2["obs"], nxt, done)
                S.append(s["obs"].astype(np.float16)); A_.append(a); R.append(r)
                S2.append(s2["obs"].astype(np.float16)); D.append(done)
                M.append(_mask(s["legal_actions"].keys())); M2.append(_mask(nxt) if nxt else np.ones(NUM_ACTIONS, bool))
    return {"s": np.stack(S), "a": np.array(A_), "r": np.array(R, np.float32), "s2": np.stack(S2),
            "d": np.array(D), "m": np.stack(M), "m2": np.stack(M2)}


def pretrain(agent: QRummyDQNAgent, demos: dict, steps: int, margin: float, lam: float, gamma: float):
    """DQfD-style pretraining (Hester et al. 2018): double-DQN TD loss + large-margin
    supervised loss towards the demonstrated (heuristic) action, on RLCard's Q-network."""
    qnet, tnet = agent.q_estimator.qnet, agent.target_estimator.qnet
    opt = agent.q_estimator.optimizer
    dev = agent.device
    N, B = len(demos["a"]), agent.batch_size
    t = {k: torch.from_numpy(v).to(dev) for k, v in demos.items()}  # ~80 MB per 40k demos, GPU-resident
    for k in range(steps):
        idx = torch.randint(N, (B,), device=dev)
        s = t["s"][idx].float()
        s2 = t["s2"][idx].float()
        a = t["a"][idx].long()
        r = t["r"][idx]
        d = t["d"][idx].float()
        m = t["m"][idx]
        m2 = t["m2"][idx]
        with torch.no_grad():
            qnet.eval()
            best = qnet(s2).masked_fill(~m2, -1e9).argmax(1)
            target = r + gamma * (1 - d) * tnet(s2).gather(1, best[:, None]).squeeze(1)
            if agent.td_clip is not None:
                target = target.clamp(-agent.td_clip, agent.td_clip)
        qnet.train()
        q = qnet(s)
        qa = q.gather(1, a[:, None]).squeeze(1)
        td = (torch.nn.functional.smooth_l1_loss if agent.huber else torch.nn.functional.mse_loss)(qa, target)
        large_margin = large_margin_loss(q, a, m, margin)
        loss = td + lam * large_margin
        opt.zero_grad()
        loss.backward()
        opt.step()
        if k % 1000 == 0:
            tnet.load_state_dict(qnet.state_dict())
            print(f"  pretrain {k}/{steps} td {td.item():.4f} margin {large_margin.item():.4f}", flush=True)
    qnet.eval()
    tnet.load_state_dict(qnet.state_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="A")
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rounds", type=int, default=None, help="stop early (smoke tests)")
    ap.add_argument("--config", default=None, help="JSON overrides for CONFIG")
    args = ap.parse_args()

    cfg = dict(CONFIG)
    if args.config:
        cfg.update(json.loads(args.config))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    ckpt_dir = ROOT / "models" / "checkpoints" / args.run
    log_dir = ROOT / "training" / "logs" / args.run
    data_dir = ROOT / "training" / "data" / args.run
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "config.json").write_text(json.dumps(cfg | vars(args), indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = rlcard.make("indian-rummy", config={"seed": args.seed, "game_turn_cap": cfg["turn_cap"]})
    agent = QRummyDQNAgent(
        replay_memory_size=cfg["replay_memory_size"], replay_memory_init_size=cfg["replay_memory_init_size"],
        update_target_estimator_every=cfg["update_target_estimator_every"],
        discount_factor=cfg["discount_factor"], epsilon_start=cfg["epsilon_start"],
        epsilon_end=cfg["epsilon_end"], epsilon_decay_steps=cfg["epsilon_decay_steps"],
        batch_size=cfg["batch_size"], num_actions=env.num_actions, state_shape=env.state_shape[0],
        train_every=cfg["train_every"], mlp_layers=cfg["mlp_layers"],
        learning_rate=cfg["learning_rate"], device=device)
    if cfg.get("td_target_clip"):
        agent.td_clip = 1.0 + cfg["shaping_coef"]   # |payoff| <= 1 and shaping telescopes to <= coef·max|Φ| = coef
    agent.huber = bool(cfg.get("huber_loss"))
    rounds_done = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        agent.q_estimator.qnet.load_state_dict(ck["qnet"])
        agent.q_estimator.optimizer.load_state_dict(ck["optimizer"])
        agent.target_estimator.qnet.load_state_dict(ck["qnet"])
        agent.total_t, agent.train_t = ck["meta"]["total_t"], ck["meta"]["train_t"]
        rounds_done = ck["meta"].get("rounds", 0)
        print(f"resumed from {args.resume} at round {rounds_done}")

    gamma = cfg["discount_factor"]
    t_start = time.time()
    budget = args.hours * 3600
    print(f"device={device} obs_dim={OBS_DIM} warm-starting replay buffer...", flush=True)
    demos = warm_start(env, agent, cfg["warm_start_transitions"], gamma, cfg["shaping_coef"])
    print(f"warm start: {len(demos['a'])} transitions in {time.time() - t_start:.0f}s", flush=True)
    pre_win = 0.0
    if not args.resume and cfg["pretrain_steps"]:
        pretrain(agent, demos, cfg["pretrain_steps"], cfg["pretrain_margin"], cfg["pretrain_lambda"], gamma)
        r = evaluate(frozen(agent, cfg), HeuristicAgent(), 200, seed=777, turn_cap=cfg["turn_cap"])
        pre_win = r.win_rate
        print(f"after pretrain vs heuristic: win {r.win_rate:.3f} loss {r.loss_rate:.3f} "
              f"diff {r.mean_score_diff:+.2f} illegal {r.illegal} ({time.time() - t_start:.0f}s)", flush=True)
        save_ckpt(agent, ckpt_dir / "pretrained.pt", cfg, {"rounds": 0})
    if cfg["demo_lambda_start"] > 0:
        agent.attach_demos(demos, cfg["pretrain_margin"])
    del demos

    heur = HeuristicAgent(epsilon=0.1, seed=21)
    snapshots: list[DQNPolicy] = []
    vlog = ValueLogger(data_dir, cfg["value_shard_size"], cfg["value_sample_prob"], rng)
    train_csv = open(log_dir / "train.csv", "a", newline="")
    eval_csv = open(log_dir / "eval.csv", "a", newline="")
    tw, ew = csv.writer(train_csv), csv.writer(eval_csv)
    if train_csv.tell() == 0:
        tw.writerow(["time_s", "rounds", "total_t", "train_t", "epsilon", "loss", "rounds_per_s", "stage", "coef"])
    if eval_csv.tell() == 0:
        ew.writerow(["time_s", "rounds", "opponent", "win_rate", "loss_rate", "mean_score_diff",
                     "forfeit_rate", "mean_turns", "illegal"])

    t_loop = time.time()
    last_ckpt = last_snap = time.time()
    window_t, window_r = time.time(), 0

    gate = {"vs_heuristic": pre_win}

    def run_eval():
        pol = frozen(agent, cfg)
        for name, me, opp, nr in (("heuristic", pol, HeuristicAgent(), cfg["eval_rounds_heuristic"]),
                                  ("random", pol, RandomAgent(seed=3), cfg["eval_rounds_random"]),
                                  ("hybrid_vs_heuristic", HybridPolicy(pol), HeuristicAgent(),
                                   cfg["eval_rounds_heuristic"])):
            r = evaluate(me, opp, nr, seed=777, turn_cap=cfg["turn_cap"])
            if name == "heuristic":
                gate["vs_heuristic"] = r.win_rate
            ew.writerow([int(time.time() - t_start), rounds_done, name, r.win_rate, r.loss_rate,
                         r.mean_score_diff, r.a_forfeit_rate, r.mean_turns, r.illegal])
            print(f"  eval vs {name}: win {r.win_rate:.3f} loss {r.loss_rate:.3f} diff {r.mean_score_diff:+.2f} "
                  f"forfeit {r.a_forfeit_rate:.3f} illegal {r.illegal}", flush=True)
        eval_csv.flush()

    try:
        while True:
            elapsed = time.time() - t_loop
            frac = elapsed / budget
            if frac >= 1 or (args.max_rounds and rounds_done >= args.max_rounds):
                break
            coef = cfg["shaping_coef"] * min(1.0, max(0.0, (1 - frac) / cfg["shaping_anneal_frac"]))
            agent.demo_lambda = cfg["demo_lambda_start"] + (cfg["demo_lambda_end"] - cfg["demo_lambda_start"]) \
                * min(1.0, frac / cfg["demo_lambda_decay_frac"])
            stage = "A" if frac < cfg["stage_a_frac"] else "B"
            if stage == "A":
                kind = "heuristic"
            else:
                u = rng.random()
                kind = "self" if u < cfg["pool"]["self"] else (
                    "heuristic" if u < cfg["pool"]["self"] + cfg["pool"]["heuristic"] else "snapshot")
                if kind == "snapshot" and not snapshots:
                    kind = "heuristic"
                if kind == "self" and gate["vs_heuristic"] < cfg["selfplay_gate"]:
                    kind = "heuristic"
            if kind == "self":
                agents, learners = [agent, agent], [0, 1]
            else:
                opp = heur if kind == "heuristic" else rng.choice(snapshots)
                seat = rng.randrange(2)
                agents = [agent, opp] if seat == 0 else [opp, agent]
                learners = [seat]
            env.set_agents(agents)
            traj, payoffs = env.run(is_training=True)
            for p, seat_ts in enumerate(reorganize_shaped(traj, payoffs, gamma, coef)):
                if p in learners:
                    for ts in seat_ts:
                        agent.feed(ts)
            if frac >= cfg["value_start_frac"]:
                vlog.add_round(traj, env.game)
            rounds_done += 1
            window_r += 1

            if rounds_done % 500 == 0:
                now = time.time()
                rps = window_r / (now - window_t)
                window_t, window_r = now, 0
                loss = float(np.mean(agent.losses[-500:])) if agent.losses else float("nan")
                eps = agent.epsilons[min(agent.total_t, agent.epsilon_decay_steps - 1)]
                tw.writerow([int(now - t_start), rounds_done, agent.total_t, agent.train_t, round(float(eps), 4),
                             round(loss, 5), round(rps, 2), stage, round(coef, 3)])
                train_csv.flush()
                agent.losses = agent.losses[-1000:]
                print(f"[{(now - t_start) / 3600:.2f}h] rounds {rounds_done} steps {agent.total_t} "
                      f"eps {eps:.3f} loss {loss:.4f} {rps:.1f} r/s stage {stage}", flush=True)
            if rounds_done % cfg["latest_every_rounds"] == 0:
                save_ckpt(agent, ckpt_dir / "latest.pt", cfg, {"rounds": rounds_done})
            if rounds_done % cfg["eval_every_rounds"] == 0:
                run_eval()
            if time.time() - last_snap > cfg["snapshot_every_s"]:
                snapshots.append(frozen(agent, cfg))
                snapshots = snapshots[-cfg["max_snapshots"]:]
                last_snap = time.time()
            if time.time() - last_ckpt > cfg["ckpt_every_s"]:
                save_ckpt(agent, ckpt_dir / f"ckpt_{rounds_done:07d}.pt", cfg, {"rounds": rounds_done})
                last_ckpt = time.time()
    except KeyboardInterrupt:
        print("interrupted: saving...", flush=True)
    finally:
        save_ckpt(agent, ckpt_dir / "latest.pt", cfg, {"rounds": rounds_done})
        save_ckpt(agent, ckpt_dir / f"ckpt_{rounds_done:07d}.pt", cfg, {"rounds": rounds_done})
        vlog.flush()
        train_csv.close()
        print(f"saved. rounds={rounds_done} steps={agent.total_t}", flush=True)
        if rounds_done:
            run_eval()
        eval_csv.close()


if __name__ == "__main__":
    main()
