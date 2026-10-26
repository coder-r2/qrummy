"""Plan Phase 7: the AI never needs its illegal-move fallback over 50 full matches (SPEC §5.3)."""

from __future__ import annotations

import random

import pytest

from server.server import store
from tests.test_api import client, new


def random_step(view, rng: random.Random):
    """A human that picks uniformly among the view's legal moves, open draws and forfeits included."""
    ph, legal = view["phase"], view["legal"]
    if ph == "reveal":
        return {"type": "reveal", "card_ids": rng.choice(legal["revealOptions"])}
    if ph == "round_over":
        return {"type": "next_round"}
    moves = []
    if ph == "your_draw":
        if legal["drawClosed"]:
            moves.append({"type": "draw", "source": "closed"})
        if legal["drawOpen"]:
            moves.append({"type": "draw", "source": "open"})
    else:
        moves += [{"type": "discard", "card_id": c} for c in legal["discardIds"]]
    if legal.get("forfeit") and rng.random() < 0.05:
        return {"type": "forfeit"}
    return rng.choice(moves)


@pytest.mark.slow
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_no_illegal_ai_moves_over_full_matches(difficulty):
    n = {"easy": 17, "medium": 17, "hard": 16}[difficulty]  # 50 matches in all
    rng = random.Random(difficulty)
    for k in range(n):
        p = new("match", difficulty, seed=500 + k)
        gid = p["view"]["gameId"]
        while p["view"]["phase"] != "match_over":
            r = client.post(f"/api/games/{gid}/actions", json=random_step(p["view"], rng))
            assert r.status_code == 200, r.text
            p = r.json()
        assert store.get(gid).ai.illegal_count == 0
        client.delete(f"/api/games/{gid}")
