"""Self-play statistics for the Learn tab's strategy notes (SPEC §8.7) -> web/public/stats.json

    python training/gen_stats.py --rounds 2000 --agent heuristic

Every number shown in the strategy notes comes from here, never hand-written.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rlcard  # noqa: E402

import indian_rummy.rlcard_env  # noqa: E402,F401
from evaluate import IllegalGuard, load_agent  # noqa: E402


def pct(a, b):
    return round(100 * a / b, 1) if b else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--agent", default="heuristic")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--out", default=str(ROOT / "web" / "public" / "stats.json"))
    args = ap.parse_args()

    env = rlcard.make("indian-rummy", config={"seed": args.seed})
    a, b = IllegalGuard(load_agent(args.agent)), IllegalGuard(load_agent(args.agent))
    env.set_agents([a, b])

    n = 0
    first_wins = decided = forfeits = capped = 0
    dealt_orig = 0                      # player-rounds with an original in the opening 13
    reveal_turn = []                    # own turns taken before revealing (0 = dealt)
    orig_kind = {"seq3": 0, "seq4": 0, "triplet": 0}
    first_reveal_rounds = first_reveal_wins = 0
    never_reveal_players = never_reveal_wins = 0
    lead_buckets: dict[str, list[int]] = {"0": [0, 0], "1-2": [0, 0], "3-5": [0, 0], "6+": [0, 0]}
    loser_post, round_turns = [], []
    open_draws = all_draws = 0
    t0 = time.time()
    for k in range(args.rounds):
        env.run(is_training=False)
        g = env.game
        res = g.result
        n += 1
        round_turns.append(g.turn)
        if res.reason == "forfeit":
            forfeits += 1
            continue
        if res.winner is None:
            capped += 1
            continue
        decided += 1
        own_turns = [0, 0]
        revealed_at: dict[int, int] = {}
        first_actor = None
        for e in g.events:
            if e["type"] == "draw":
                first_actor = e["actor"] if first_actor is None else first_actor
                all_draws += 1
                open_draws += e["source"] == "open"
            elif e["type"] == "discard":
                own_turns[e["actor"]] += 1
            elif e["type"] == "reveal" and e["actor"] not in revealed_at:
                p = e["actor"]
                revealed_at[p] = own_turns[p]
                ts = [g.type_of(c) for c in e["cards"]]
                orig_kind["triplet" if len(set(ts)) == 1 else f"seq{len(ts)}"] += 1
        first_wins += res.winner == first_actor
        for p in (0, 1):
            if p in revealed_at:
                reveal_turn.append(revealed_at[p])
                dealt_orig += revealed_at[p] == 0
            else:
                never_reveal_players += 1
                never_reveal_wins += res.winner == p
            if p != res.winner and p in revealed_at:
                loser_post.append(res.points[p])
        if len(revealed_at) >= 1:
            order = sorted(revealed_at, key=lambda p: revealed_at[p])
            leader = order[0]
            if len(revealed_at) == 2 and revealed_at[0] == revealed_at[1]:
                lead = 0
            else:
                first_reveal_rounds += 1
                first_reveal_wins += res.winner == leader
                other = revealed_at.get(1 - leader, own_turns[1 - leader] + 1)
                lead = other - revealed_at[leader]
            key = "0" if lead == 0 else "1-2" if lead <= 2 else "3-5" if lead <= 5 else "6+"
            if lead:
                lead_buckets[key][0] += 1
                lead_buckets[key][1] += res.winner == leader
        if k % 250 == 249:
            print(f"{k + 1} rounds, {time.time() - t0:.0f}s", flush=True)

    players = 2 * decided
    kinds = sum(orig_kind.values())
    out = {
        "version": 1,
        "agent": args.agent,
        "rounds": n,
        "illegal": a.illegal + b.illegal,
        "decided_pct": pct(decided, n),
        "forfeit_pct": pct(forfeits, n),
        "first_player_win_pct": pct(first_wins, decided),
        "dealt_original_pct": pct(dealt_orig, players),
        "reveal_turn_median": st.median(reveal_turn) if reveal_turn else None,
        "reveal_turn_p75": sorted(reveal_turn)[int(0.75 * len(reveal_turn))] if reveal_turn else None,
        "never_reveal_pct": pct(never_reveal_players, players),
        "never_reveal_win_pct": pct(never_reveal_wins, never_reveal_players),
        "first_reveal_win_pct": pct(first_reveal_wins, first_reveal_rounds),
        "reveal_lead_win_pct": {k: pct(w, r) for k, (r, w) in lead_buckets.items() if k != "0"},
        "reveal_lead_rounds": {k: r for k, (r, w) in lead_buckets.items() if k != "0"},
        "original_kind_pct": {k: pct(v, kinds) for k, v in orig_kind.items()},
        "loser_post_points_mean": round(st.mean(loser_post), 1) if loser_post else None,
        "loser_post_points_median": st.median(loser_post) if loser_post else None,
        "open_draw_pct": pct(open_draws, all_draws),
        "round_turns_median": st.median(round_turns),
        "printed_joker_cj_pct": round(100 * 6 / 162, 1),
    }
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
