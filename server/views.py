"""GameView / Event serialisation (SPEC §7.4) for the human seat.

This module is the information-hiding boundary: everything sent to the browser passes through
here, and only information the human may see (R11) is included.
"""

from __future__ import annotations

from indian_rummy import actions as A
from indian_rummy.agents.heuristic import heuristic_reveal
from indian_rummy.cards import JOKER, SUITS, rank_of, suit_of, type_code
from indian_rummy.game import Phase, RummyGame
from indian_rummy.solver import solve

HUMAN, AI = 0, 1


def card_json(game: RummyGame, card: int, wild_rank: int | None) -> dict:
    t = game.type_of(card)
    joker = t == JOKER
    return {
        "id": int(card),
        "code": type_code(t),
        "rank": None if joker else rank_of(t),
        "suit": None if joker else SUITS[suit_of(t)],
        "isJoker": joker,
        "isWild": joker or (wild_rank is not None and rank_of(t) == wild_rank),
    }


def _groups_to_ids(game: RummyGame, hand: list[int], melds: list[list[int]], locked: list[int]) -> list[list[int]]:
    """Map solver melds (types) back onto physical card ids, locked cards first."""
    pool = list(locked) + [c for c in hand if c not in locked]
    out = []
    for meld in melds:
        ids = []
        for t in meld:
            c = next((c for c in pool if game.type_of(c) == t), None)
            if c is not None:
                pool.remove(c)
                ids.append(c)
        out.append(ids)
    return out


def human_wild(game: RummyGame) -> int | None:
    p = game.players[HUMAN]
    return game.wild_rank if (p.post_original or game.cj_face_up) else None


def event_json(game: RummyGame, ev: dict) -> dict:
    """Public version of an engine event; hides cards the human must not see."""
    wr = human_wild(game)
    actor = {HUMAN: "you", AI: "ai"}.get(ev["actor"], "dealer")
    out = {"seq": ev["seq"], "actor": actor, "type": ev["type"]}
    et = ev["type"]
    if et == "draw":
        out["source"] = ev["source"]
        if ev["actor"] == HUMAN or ev["source"] == "open":
            out["card"] = card_json(game, ev["card"], wr)
    elif et == "discard":
        out["card"] = card_json(game, ev["card"], wr)
    elif et == "reveal":
        out["cards"] = [card_json(game, c, wr) for c in ev["cards"]]
    elif et == "cj_flip":
        out["card"] = card_json(game, ev["card"], wr)
    elif et == "deal":
        out["dealer"] = {HUMAN: "you", AI: "ai"}[ev["dealer"]]
        out["card"] = card_json(game, ev["open"], wr)
    elif et == "declare":
        out["by"] = actor
    elif et == "round_end":
        out["reason"] = ev["reason"]
        out["points"] = {"you": ev["points"][HUMAN], "ai": ev["points"][AI]}
    elif et == "reshuffle":
        out["count"] = ev["count"]
    return out


def phase_of(session) -> str:
    game = session.game
    if session.match.is_over() and game.is_over():
        return "match_over"
    if game.is_over():
        return "round_over"
    if game.current != HUMAN:
        return "ai_turn"
    return {Phase.REVEAL: "reveal", Phase.DRAW: "your_draw", Phase.DISCARD: "your_discard"}[game.phase]


def game_view(session) -> dict:
    game: RummyGame = session.game
    match = session.match
    you, ai = game.players[HUMAN], game.players[AI]
    wr = human_wild(game)
    phase = phase_of(session)
    over = game.is_over()

    legal = {"drawClosed": False, "drawOpen": False, "discardIds": [], "forfeit": False, "revealOptions": [],
             "revealRecommended": None}
    if phase in ("your_draw", "your_discard"):
        acts = game.get_legal_actions()
        legal["drawClosed"] = A.DRAW_CLOSED in acts
        legal["drawOpen"] = A.DRAW_OPEN in acts
        legal["forfeit"] = A.FORFEIT in acts
        ok_types = {A.discard_type(a) for a in acts if A.is_discard(a)}
        legal["discardIds"] = [c for c in you.hand if game.type_of(c) in ok_types and c not in you.locked]
    elif phase == "reveal":
        opts = game.reveal_options(HUMAN)
        legal["revealOptions"] = [list(o) for o in opts]
        best = heuristic_reveal(game, HUMAN, opts)
        legal["revealRecommended"] = opts.index(best) if best in opts else 0

    arr = solve(game.counts(HUMAN), game.hand_context(HUMAN))
    suggested = _groups_to_ids(game, you.hand, arr.melds, you.locked)

    round_result = None
    if over and game.result is not None:
        res = game.result

        def side(pid):
            p = game.players[pid]
            a = res.arrangements[pid]
            groups = _groups_to_ids(game, p.hand, a.melds, p.locked) if a is not None else []
            return {"points": res.points[pid], "groups": groups}
        round_result = {
            "winner": {HUMAN: "you", AI: "ai", None: "none"}[res.winner],
            "reason": res.reason,
            "multiplier": session.round_multiplier,
            "you": side(HUMAN),
            "ai": side(AI),
        }

    totals = match.totals()  # finished rounds are recorded immediately by the session

    return {
        "gameId": session.id,
        "mode": session.mode,
        "difficulty": session.difficulty,
        "coach": session.coach,
        "round": match.round_no,
        "totalRounds": match.total_rounds,
        "multiplier": session.round_multiplier,
        "scores": {"you": totals[HUMAN], "ai": totals[AI]},
        "roundHistory": [
            {"round": r.round_no, "multiplier": r.multiplier, "you": r.scored[HUMAN], "ai": r.scored[AI],
             "winner": {HUMAN: "you", AI: "ai", None: "none"}[r.result.winner]}
            for r in match.rounds
        ],
        "phase": phase,
        "dealer": "you" if game.dealer == HUMAN else "ai",
        "you": {
            "hand": [card_json(game, c, wr) for c in you.hand],
            "lockedIds": list(you.locked),
            "postOriginal": you.post_original,
            "wildRank": (wr if wr is not None else ("none" if (you.post_original and game.wild_rank is None) else None)),
            "suggestedGroups": suggested,
            "drawnId": you.drawn,
        },
        "ai": {
            "handCount": len(ai.hand),
            "postOriginal": ai.post_original,
            "original": [card_json(game, c, wr) for c in ai.locked] if ai.locked else None,
            "hand": [card_json(game, c, wr) for c in ai.hand] if over else None,
        },
        "table": {
            "drawCount": len(game.draw_pile),
            "discardTop": card_json(game, game.discard_pile[-1], wr) if game.discard_pile else None,
            "discardCount": len(game.discard_pile),
            "cj": {
                "faceUp": game.cj_face_up,
                "card": card_json(game, game.cj, wr) if (you.post_original or game.cj_face_up or over) else None,
            },
        },
        "legal": legal,
        "roundResult": round_result,
        "matchWinner": ({HUMAN: "you", AI: "ai", None: "draw"}[match.winner()]
                        if phase == "match_over" else None),
    }
