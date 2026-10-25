"""Solver performance gate (SPEC §4.3): python tests/bench_solver.py

Uses realistic hands: random deals from the v13 deck, 14 cards, with a locked original taken from
the hand when one exists (post-original), otherwise solved pre-original.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indian_rummy.cards import build_deck, counts_of  # noqa: E402
from indian_rummy.config import V13  # noqa: E402
from indian_rummy.solver import HandContext, discard_eval, find_originals, solve  # noqa: E402


def sample_cases(n: int, seed: int = 0):
    rng = np.random.RandomState(seed)
    deck = build_deck(V13)
    cases = []
    while len(cases) < n:
        hand = [int(t) for t in rng.choice(deck, 14, replace=False)]
        counts = counts_of(hand)
        origs = find_originals(counts, sizes=(3,))
        wild = int(rng.randint(1, 14))
        if origs:
            ctx = HandContext(post_original=True, wild_rank=wild, locked=tuple(sorted(origs[0])))
        else:
            ctx = HandContext(post_original=False)
        cases.append((counts, ctx))
    return cases


def main():
    cases = sample_cases(2000)
    times = []
    for counts, ctx in cases:
        t0 = time.perf_counter()
        solve(counts, ctx)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    med, p99 = statistics.median(times), times[int(len(times) * 0.99)]
    post = sum(c.post_original for _, c in cases)
    print(f"solve: n={len(times)} (post-original {post})  median {med:.3f} ms  p99 {p99:.3f} ms  max {times[-1]:.1f} ms")

    t0 = time.perf_counter()
    for counts, ctx in cases[:300]:
        cand = [t for t in range(53) if counts[t] > sum(1 for x in ctx.locked if x == t)]
        discard_eval(counts, ctx, cand)
    per = (time.perf_counter() - t0) * 1000 / 300
    print(f"discard_eval (all discards of a 14-card hand): {per:.2f} ms per hand")
    ok = med <= 1.0 and p99 <= 10.0
    print("GATE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
