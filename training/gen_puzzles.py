"""Generate static Learn-tab puzzles (SPEC §8.7) -> web/public/puzzles.json

    python training/gen_puzzles.py --n 15

Every answer is computed by the engine (melds.py / solver.py / heuristic), never hand-written.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rlcard  # noqa: E402

import indian_rummy.rlcard_env  # noqa: E402,F401
from indian_rummy import actions as A  # noqa: E402
from indian_rummy.agents import HeuristicAgent  # noqa: E402
from indian_rummy.cards import JOKER, counts_of, make_type, type_code  # noqa: E402
from indian_rummy.melds import is_secondary, is_valid_original  # noqa: E402
from indian_rummy.solver import HandContext, solve  # noqa: E402

RANKS = {1: "Ace", 11: "Jack", 12: "Queen", 13: "King"}


def rname(r):
    return RANKS.get(r, str(r))


def codes(types):
    return " ".join(type_code(t) for t in types)


def rand_seq(rng, n=3, natural=True):
    s = rng.randrange(4)
    start = rng.randint(1, 14 - n)
    ranks = [((start - 1 + k) % 13) + 1 if start + k <= 13 else 1 for k in range(n)]
    return [make_type(s, r) for r in ranks]


def distractors(rng):
    s, s2 = rng.sample(range(4), 2)
    r = rng.randint(2, 11)
    return [
        ([make_type(s, r), make_type(s2, r), make_type((s2 + 1) % 4 if (s2 + 1) % 4 != s else (s2 + 2) % 4, r)],
         "a set of different suits isn't allowed as the original"),
        ([make_type(s, r), JOKER, make_type(s, r + 2)], "jokers can't be used in the original"),
        ([make_type(s, 13), make_type(s, 1), make_type(s, 2)], "sequences can't wrap around from King to 2"),
        ([make_type(s, r), make_type(s, r + 1), make_type(s2, r + 2)], "all cards in a sequence must share a suit"),
        ([make_type(s, r), make_type(s, r + 1), make_type(s, r + 3)], "there's a gap in the sequence"),
        ([make_type(s, r), make_type(s, r), make_type(s2, r)], "a triplet needs three identical cards"),
    ]


def p_valid_original(rng, k):
    good = rand_seq(rng) if rng.random() < 0.7 else [make_type(rng.randrange(4), rng.randint(1, 13))] * 3
    assert is_valid_original(good)
    bad = rng.sample(distractors(rng), 3)
    for b, _ in bad:
        assert not is_valid_original(b), b
    opts = [(good, "a natural pure sequence or identical triplet — exactly what an original needs")] + bad
    rng.shuffle(opts)
    return {
        "id": f"orig-{k}", "type": "valid_original",
        "prompt": "Which of these can be revealed as your original rummy?",
        "options": [codes(o) for o, _ in opts],
        "answer": [i for i, (o, _) in enumerate(opts) if o is good][0],
        "explanations": [why for _, why in opts],
    }


def p_valid_meld(rng, k):
    wild = rng.randint(2, 13)
    s, s2 = rng.sample(range(4), 2)
    cands = []
    while len(cands) < 12:
        n = rng.choice([3, 3, 4])
        kind = rng.random()
        r = rng.randint(1, 13)
        if kind < 0.35:
            grp = [make_type(s, min(13, r + i)) for i in range(n)]
        elif kind < 0.6:
            grp = [make_type(x, r) for x in rng.sample(range(4), n)]
        else:
            grp = [rng.choice([JOKER, make_type(rng.randrange(4), wild), make_type(s, r), make_type(s2, r),
                               make_type(s, min(13, r + 1)), make_type(s, min(13, r + 2))]) for _ in range(n)]
        cands.append(grp)
    good = [g for g in cands if is_secondary(g, wild)]
    bad = [g for g in cands if not is_secondary(g, wild)]
    if not good or len(bad) < 3:
        return None
    opts = [good[0]] + bad[:3]
    rng.shuffle(opts)
    ans = opts.index(good[0])
    return {
        "id": f"meld-{k}", "type": "valid_meld",
        "prompt": f"You've revealed your original and the wild rank is {rname(wild)}. Which group is a valid meld?",
        "wildRank": wild,
        "options": [codes(o) for o in opts],
        "answer": ans,
        "explanations": ["valid: " + ("uses wild cards with at least one natural card" if any(
            t == JOKER or t % 13 + 1 == wild for t in o) else "a natural sequence or set") if i == ans
            else "not a valid sequence, set or triplet under these rules" for i, o in enumerate(opts)],
    }


def positions(n_rounds, seed):
    """Discard decisions from heuristic self-play with a clear best move."""
    env = rlcard.make("indian-rummy", config={"seed": seed})
    h = HeuristicAgent()
    out = []
    for _ in range(n_rounds):
        state, _ = env.reset()
        while not env.is_over():
            an = state["raw_obs"]["analysis"]
            if an.decision == "discard" and len(an.discards) >= 4:
                ranked = h.ranked_discards(an)
                s0, s1 = h.discard_score(an, ranked[0]), h.discard_score(an, ranked[1])
                if s1 - s0 >= 2.0 and not an.discards[ranked[0]].win:
                    out.append((an, ranked))
            state, _ = env.step(h.step(state))
    return out


def p_best_discard(an, ranked, rng, k):
    h = HeuristicAgent()
    best = ranked[0]
    others = rng.sample(ranked[1:], min(3, len(ranked) - 1))
    opts = [best] + others
    rng.shuffle(opts)

    def why(t):
        d = an.discards[t]
        bits = [f"leaves {d.points} unmelded points", f"{d.outs} unseen cards would help"]
        if not an.post_original:
            bits.insert(0, "keeps your original" if d.has_original else "breaks your only original")
        if an.is_wild(t):
            bits.append("throws away a wild card")
        return ", ".join(bits)
    hand = [t for t in range(53) for _ in range(int(an.counts[t]))]
    return {
        "id": f"discard-{k}", "type": "best_discard",
        "prompt": "You've just drawn. Which card should you discard?",
        "hand": codes(hand),
        "locked": codes(list(an.ctx.locked)),
        "wildRank": an.wild_rank,
        "postOriginal": an.post_original,
        "options": [type_code(t) for t in opts],
        "answer": opts.index(best),
        "explanations": [why(t) for t in opts],
    }


def p_score(an, rng, k):
    c13 = an.counts.copy()
    arr = solve(c13, an.ctx)
    pts = min(arr.points, 80)
    wrong = {max(0, pts + d) for d in rng.sample([-10, -7, -5, -3, 3, 5, 7, 10, 12], 5)} - {pts}
    opts = [pts] + rng.sample(sorted(wrong), 3)
    rng.shuffle(opts)
    hand = [t for t in range(53) for _ in range(int(c13[t]))]
    return {
        "id": f"score-{k}", "type": "score_hand",
        "prompt": "The AI just declared. You have revealed your original. How many points does this hand score?",
        "hand": codes(hand),
        "locked": codes(list(an.ctx.locked)),
        "wildRank": an.wild_rank,
        "options": [str(o) for o in opts],
        "answer": opts.index(pts),
        "melds": [codes(m) for m in arr.melds],
        "unmelded": codes(arr.uncovered),
        "explanations": [f"Best arrangement leaves {codes(arr.uncovered) or 'nothing'} unmelded = {arr.points} points"
                         + (" (capped at 80)" if arr.points > 80 else "")] * len(opts),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=str(ROOT / "web" / "public" / "puzzles.json"))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    puzzles = [p_valid_original(rng, k) for k in range(args.n)]
    k = 0
    while sum(p["type"] == "valid_meld" for p in puzzles) < args.n:
        p = p_valid_meld(rng, k)
        k += 1
        if p:
            puzzles.append(p)
    pos = positions(120, args.seed)
    rng.shuffle(pos)
    post = [(an, r) for an, r in pos if an.post_original]
    pre = [(an, r) for an, r in pos if not an.post_original]
    picks = post[: (2 * args.n) // 3] + pre[: args.n - (2 * args.n) // 3]
    for i, (an, ranked) in enumerate(picks):
        puzzles.append(p_best_discard(an, ranked, rng, i))
    # score puzzles from post-original 13-card hands (after the heuristic's discard)
    j = 0
    for an, ranked in post[args.n: args.n * 3]:
        c = an.counts.copy()
        c[ranked[0]] -= 1
        an.counts = c
        p = p_score(an, rng, j)
        if p["unmelded"]:
            puzzles.append(p)
            j += 1
        if j >= args.n:
            break
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"version": 1, "puzzles": puzzles}, indent=1))
    from collections import Counter
    print(f"wrote {len(puzzles)} puzzles -> {args.out}", Counter(p["type"] for p in puzzles))


if __name__ == "__main__":
    main()
