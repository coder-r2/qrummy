"""Backend API (SPEC §7, §10.3)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from server.server import app, store

client = TestClient(app)


def new(mode="match", difficulty="easy", seed=1):
    r = client.post("/api/games", json={"mode": mode, "difficulty": difficulty, "coach": "on", "seed": seed})
    assert r.status_code == 200
    return r.json()


def autoplay_step(view):
    """A trivial human policy driven purely by the view's legal moves."""
    ph, legal = view["phase"], view["legal"]
    if ph == "reveal":
        return {"type": "reveal", "card_ids": legal["revealOptions"][0]}
    if ph == "your_draw":
        return {"type": "draw", "source": "closed" if legal["drawClosed"] else "open"}
    if ph == "your_discard":
        return {"type": "discard", "card_id": legal["discardIds"][0]}
    if ph == "round_over":
        return {"type": "next_round"}
    raise AssertionError(f"unexpected phase {ph}")


def assert_hidden(payload: dict):
    """§7.4 security invariants: nothing hidden ever leaves the server before round end."""
    view = payload["view"]
    text = json.dumps(payload)
    assert "draw_pile" not in text and "np_random" not in text
    if view["phase"] not in ("round_over", "match_over"):
        assert view["ai"]["hand"] is None
        if not view["you"]["postOriginal"] and not view["table"]["cj"]["faceUp"]:
            assert view["table"]["cj"]["card"] is None
        for ev in payload["events"]:
            if ev["type"] == "draw" and ev["actor"] == "ai" and ev["source"] == "closed":
                assert "card" not in ev


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_full_match_flow(difficulty):
    p = new("match", difficulty, seed=3)
    gid = p["view"]["gameId"]
    assert p["view"]["totalRounds"] == 7 and p["view"]["multiplier"] == 2
    assert_hidden(p)
    steps = 0
    while p["view"]["phase"] != "match_over":
        r = client.post(f"/api/games/{gid}/actions", json=autoplay_step(p["view"]))
        assert r.status_code == 200, r.text
        p = r.json()
        assert_hidden(p)
        steps += 1
        assert steps < 5000
    v = p["view"]
    assert len(v["roundHistory"]) == 7
    assert [h["multiplier"] for h in v["roundHistory"]] == [2, 1, 1, 1, 1, 1, 2]
    assert v["scores"]["you"] == sum(h["you"] for h in v["roundHistory"])
    assert v["matchWinner"] in ("you", "ai", "draw")
    assert v["ai"]["hand"] is not None  # revealed at the end


def test_round_mode_single_round():
    p = new("round", seed=4)
    assert p["view"]["totalRounds"] == 1 and p["view"]["multiplier"] == 1


def test_illegal_actions_409():
    p = new(seed=5)
    gid = p["view"]["gameId"]
    v = p["view"]
    if v["phase"] == "your_draw":
        r = client.post(f"/api/games/{gid}/actions", json={"type": "discard", "card_id": v["you"]["hand"][0]["id"]})
        assert r.status_code == 409
    r = client.post(f"/api/games/{gid}/actions", json={"type": "next_round"})
    assert r.status_code == 409
    r = client.post(f"/api/games/{gid}/actions", json={"type": "discard", "card_id": 99999})
    assert r.status_code == 409


def test_unknown_game_404():
    assert client.get("/api/games/nope").status_code == 404


def test_hint_schema_and_explain_fallback(monkeypatch):
    p = new(seed=6)
    gid = p["view"]["gameId"]
    v = p["view"]
    while v["phase"] not in ("your_draw", "your_discard"):
        v = client.post(f"/api/games/{gid}/actions", json=autoplay_step(v)).json()["view"]
    h = client.post(f"/api/games/{gid}/hint").json()
    assert h["available"] and h["actions"]
    for row in h["actions"]:
        assert {"action", "label", "q", "preference", "winProbability", "tags"} <= set(row)
    assert abs(sum(r["preference"] for r in h["actions"]) - 1) < 1e-3
    assert h["explanation"]["text"]
    from server import server as srv
    monkeypatch.setattr(srv.llm, "key", "")
    e = client.post(f"/api/games/{gid}/hint/explain").json()
    assert e["source"] == "template"


def test_hint_does_not_push_forfeit():
    """Regression: the heuristic fallback ranked Forfeit first after a first-turn closed draw."""
    pushed = 0
    for seed in range(20, 32):
        p = new(seed=seed)
        gid, v = p["view"]["gameId"], p["view"]
        while v["phase"] not in ("your_draw", "your_discard"):
            v = client.post(f"/api/games/{gid}/actions", json=autoplay_step(v)).json()["view"]
        if v["phase"] == "your_draw" and v["legal"]["drawClosed"]:
            v = client.post(f"/api/games/{gid}/actions", json={"type": "draw", "source": "closed"}).json()["view"]
        if v["phase"] != "your_discard" or not v["legal"]["forfeit"]:
            continue
        h = client.post(f"/api/games/{gid}/hint").json()
        pushed += h["actions"][0]["action"]["type"] == "forfeit"
    assert pushed <= 2  # only genuinely hopeless hands


def test_session_store_expiry():
    p = new(seed=7)
    gid = p["view"]["gameId"]
    store.sessions[gid].last_access -= 3 * 3600
    store.expire()
    assert client.get(f"/api/games/{gid}").status_code == 404
