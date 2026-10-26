"""Coach / XAI layer (SPEC §6.2): ranked moves, win probability, reason tags, template text.

Everything is computed from the human's information set. Hypothetical positions (after a
discard, after a closed draw) are built on a *copy* of the game in which hidden cards are
re-sampled from the human's unseen pool, so no hidden information can influence the numbers.
"""

from __future__ import annotations

import copy
import random

import numpy as np

from indian_rummy import actions as A
from indian_rummy.agents.heuristic import HeuristicAgent, heuristic_reveal
from indian_rummy.agents.hybrid import HybridPolicy
from indian_rummy.cards import JOKER, SUITS, base_points, rank_of, suit_of, type_code
from indian_rummy.features import Analysis, analyse, encode
from indian_rummy.game import Phase, RummyGame

from .views import AI, HUMAN

SUIT_SYMBOL = {"S": "♠", "H": "♥", "D": "♦", "C": "♣"}
TAU = 0.05
CLOSED_SAMPLES = 16


def card_name(t: int) -> str:
    if t == JOKER:
        return "Joker"
    code = type_code(t)
    return code[:-1] + SUIT_SYMBOL[code[-1]]


def softmax(x: np.ndarray, tau: float) -> np.ndarray:
    z = (x - x.max()) / tau
    e = np.exp(z)
    return e / e.sum()


class Coach:
    def __init__(self, models):
        self.models = models
        self.heuristic = HeuristicAgent()

    # ------------------------------------------------------------------ value of a position
    def _value(self, game: RummyGame) -> float | None:
        vm = self.models.value
        if vm is None:
            return None
        st = game.get_state(HUMAN)
        an = analyse(st, game.types, game.variant)
        return float(vm.predict(encode(an))[0])

    def _as_human_draw(self, g: RummyGame) -> RummyGame:
        """Re-cast a position as 'human to draw next' so the value net sees a decision state."""
        if g.is_over():
            return g
        if g.phase == Phase.REVEAL and g.current == HUMAN:
            g.step(("reveal", heuristic_reveal(g, HUMAN, g.reveal_options(HUMAN))))
        if not g.is_over():
            g.current, g.phase = HUMAN, Phase.DRAW
        return g

    def _hidden_pool(self, g: RummyGame) -> list[int]:
        ai = g.players[AI]
        pool = list(g.draw_pile) + [c for c in ai.hand if c not in ai.locked and c not in ai.known_open]
        if not (g.players[HUMAN].post_original or g.cj_face_up):
            pool.append(g.cj)
        return pool

    def _resample_hidden(self, g: RummyGame, rng: random.Random) -> None:
        """Shuffle every card the human cannot see among the hidden slots (AI hand, draw pile, CJ)."""
        ai = g.players[AI]
        hidden_hand = [i for i, c in enumerate(ai.hand) if c not in ai.locked and c not in ai.known_open]
        cj_hidden = not (g.players[HUMAN].post_original or g.cj_face_up)
        pool = self._hidden_pool(g)
        rng.shuffle(pool)
        k = 0
        for i in hidden_hand:
            ai.hand[i] = pool[k]; k += 1
        n = len(g.draw_pile)
        g.draw_pile = pool[k:k + n]; k += n
        if cj_hidden:
            g.cj = pool[k]
            g.wild_rank = None if g.type_of(g.cj) == JOKER else rank_of(g.type_of(g.cj))

    def action_win_prob(self, game: RummyGame, a, rng: random.Random) -> float | None:
        if self.models.value is None:
            return None
        if a == A.FORFEIT:
            return 0.0
        if a == A.DRAW_CLOSED:
            vals = []
            for _ in range(CLOSED_SAMPLES):
                g = copy.deepcopy(game)
                self._resample_hidden(g, rng)
                g.step(A.DRAW_CLOSED)
                vals.append(self._value(g))
            return float(np.mean(vals))
        g = copy.deepcopy(game)
        self._resample_hidden(g, rng)
        g.step(a)
        if g.is_over():
            return 1.0 if g.result.winner == HUMAN else 0.0
        if g.phase == Phase.DISCARD and g.current == HUMAN:
            return self._value(g)
        return self._value(self._as_human_draw(g))

    # ------------------------------------------------------------------ tags and text
    def tags_for(self, an: Analysis, a: int) -> list[str]:
        tags: list[str] = []
        if a == A.FORFEIT:
            return ["forfeit_fixed_40"]
        if a == A.DRAW_OPEN:
            if not an.post_original:
                tags.append("completes_original")
            if an.open_win:
                tags.append("completes_win")
            if an.open_best_points is not None and an.current is not None:
                gain = an.current.points - an.open_best_points
                if gain > 0:
                    tags.append(f"reduces_deadwood:{gain}")
            tags.append("reveals_card_to_opponent")
            return tags
        if a == A.DRAW_CLOSED:
            return ["unknown_card", "keeps_plans_hidden"]
        t = A.discard_type(a)
        d = an.discards.get(t)
        if d is None:
            return tags
        if d.win:
            tags.append("completes_win")
        if not an.post_original and d.has_original:
            tags.append("keeps_original")
        if an.is_wild(t):
            tags.append("discards_wild")
        in_meld = an.current is not None and any(t in m for m in an.current.melds)
        if in_meld:
            tags.append("breaks_meld")
        elif base_points(t) >= 10:
            tags.append("sheds_high_card")
        seen = int(an.dead[t]) + (1 if an.top == t else 0)
        if seen >= 1 and t != JOKER:
            tags.append(f"dead_card:{seen}")
        if t != JOKER and self._opponent_collecting(an, t):
            tags.append("opponent_may_want")
        tags.append(f"deadwood_after:{d.points}")
        if d.outs:
            tags.append(f"outs:{d.outs}")
        return tags

    @staticmethod
    def _opponent_collecting(an: Analysis, t: int) -> bool:
        for u in range(52):
            if an.opp_known[u] == 0 or u == t:
                continue
            if rank_of(u) == rank_of(t):
                return True
            if suit_of(u) == suit_of(t) and abs(rank_of(u) - rank_of(t)) <= 2:
                return True
        return False

    @staticmethod
    def label(an: Analysis, a: int) -> str:
        if a == A.DRAW_CLOSED:
            return "Draw from the closed pile"
        if a == A.DRAW_OPEN:
            return f"Take {card_name(an.top)} from the discard pile"
        if a == A.FORFEIT:
            return "Forfeit this round (40 points)"
        return f"Discard {card_name(A.discard_type(a))}"

    @staticmethod
    def template_text(label: str, tags: list[str], win_p: float | None) -> str:
        parts = []
        tagset = {t.split(":")[0]: (t.split(":")[1] if ":" in t else None) for t in tags}
        if "completes_win" in tagset:
            return f"{label} — this completes your hand and wins the round!"
        if "completes_original" in tagset:
            parts.append("it completes your original rummy, unlocking the closed joker")
        if "keeps_original" in tagset:
            parts.append("you keep a valid original to reveal")
        if "reduces_deadwood" in tagset:
            parts.append(f"it cuts your unmelded points by {tagset['reduces_deadwood']}")
        if "sheds_high_card" in tagset:
            parts.append("it sheds a high-value card that isn't helping any meld")
        if "dead_card" in tagset:
            parts.append(f"{tagset['dead_card']} cop{'y' if tagset['dead_card'] == '1' else 'ies'} of it "
                         "are already gone, so it's hard to build around")
        if "outs" in tagset:
            parts.append(f"about {tagset['outs']} unseen cards would still improve your hand")
        if "opponent_may_want" in tagset:
            parts.append("careful: the AI has picked up cards near this one")
        if "breaks_meld" in tagset:
            parts.append("note it breaks one of your current melds")
        if "discards_wild" in tagset:
            parts.append("throwing a wild card is usually a mistake")
        if "unknown_card" in tagset:
            parts.append("the open card doesn't help enough, and a closed draw keeps your plans hidden")
        if "forfeit_fixed_40" in tagset:
            parts.append("you take a fixed 40 points instead of risking up to 80")
        text = label + (": " + "; ".join(parts[:3]) + "." if parts else ".")
        if win_p is not None:
            text += f" Estimated win chance after this move: {round(100 * win_p)}%."
        return text

    # ------------------------------------------------------------------ main entry
    def hint(self, session) -> dict:
        game: RummyGame = session.game
        if game.is_over() or game.current != HUMAN or game.phase == Phase.REVEAL:
            return {"available": False}
        rng = random.Random(len(game.events))
        an = analyse(game.get_state(HUMAN), game.types, game.variant)
        obs = encode(an)
        legal = list(an.legal)
        dqn = self.models.coach_dqn
        if dqn is not None:
            qd = dqn.masked_q(obs, legal)
            q = np.array([qd[a] for a in legal])
            # The hybrid's move goes first: the raw DQN's argmax plays much weaker discards (D-096).
            pick = HybridPolicy(dqn).act_analysis(an, obs)
            q[legal.index(pick)] = q.max() + 1e-3
        else:  # heuristic preference as pseudo-Q
            q = -self._heuristic_costs(an, legal) / 20.0
        pref = softmax(q, TAU)
        vm = self.models.value
        win_now = float(vm.predict(obs)[0]) if vm is not None else None
        rows = []
        for a, qa, pa in zip(legal, q, pref):
            wp = self.action_win_prob(game, a, rng)
            tags = self.tags_for(an, a)
            d = an.discards.get(A.discard_type(a)) if A.is_discard(a) else None
            rows.append({"action": self._action_json(game, an, a), "label": self.label(an, a),
                         "q": round(float(qa), 4), "preference": round(float(pa), 4),
                         "winProbability": None if wp is None else round(wp, 4),
                         "deadwoodAfter": None if d is None else d.points, "tags": tags})
        order = sorted(range(len(rows)), key=lambda i: -rows[i]["q"])
        rows = [rows[i] for i in order]
        best = rows[0]
        decision = "draw" if an.decision == "draw" else "discard"
        return {
            "available": True,
            "winProbability": win_now,
            "decision": "forfeit_window" if (an.state["forfeit_window"] and A.FORFEIT in legal) else decision,
            "actions": rows,
            "recommendedIndex": 0,
            "explanation": {"text": self.template_text(best["label"], best["tags"], best["winProbability"]),
                            "source": "template"},
            "facts": self.facts(an, rows),
        }

    def _heuristic_costs(self, an: Analysis, legal: list[int]) -> np.ndarray:
        """Heuristic cost per legal action (lower is better), clipped so a win doesn't swamp the scale."""
        best = self.heuristic.act(an)
        costs = np.array([max(-100.0, self.heuristic.discard_score(an, A.discard_type(a))) if A.is_discard(a)
                          else 0.0 if a == best else 5.0 for a in legal])
        if A.FORFEIT in legal:
            # Forfeit sits just above or just below the alternatives, per the heuristic's forfeit rule.
            others = [c for a, c in zip(legal, costs) if a != A.FORFEIT]
            i = legal.index(A.FORFEIT)
            costs[i] = (min(others) - 1.0) if self.heuristic.should_forfeit(an) else (max(others) + 5.0)
        return costs

    @staticmethod
    def _action_json(game, an, a) -> dict:
        if a == A.DRAW_CLOSED:
            return {"type": "draw", "source": "closed"}
        if a == A.DRAW_OPEN:
            return {"type": "draw", "source": "open"}
        if a == A.FORFEIT:
            return {"type": "forfeit"}
        t = A.discard_type(a)
        p = game.players[HUMAN]
        card = next(c for c in p.hand if game.type_of(c) == t and c not in p.locked)
        return {"type": "discard", "card_id": card}

    @staticmethod
    def facts(an: Analysis, rows: list[dict]) -> dict:
        """Compact, human-visible facts for the LLM (SPEC §6.3)."""
        cur = an.current
        return {
            "phase": "post_original" if an.post_original else "pre_original",
            "decision": an.decision,
            "wild_rank": an.wild_rank,
            "hand_melds": [[card_name(t) for t in m] for m in (cur.melds if cur else [])],
            "unmelded": [card_name(t) for t in (cur.uncovered if cur else [])],
            "unmelded_points": cur.points if cur else None,
            "open_card": card_name(an.top) if an.top is not None else None,
            "recommended": {"move": rows[0]["label"], "win_pct": rows[0]["winProbability"], "tags": rows[0]["tags"]},
            "alternatives": [{"move": r["label"], "win_pct": r["winProbability"], "tags": r["tags"][:3]}
                             for r in rows[1:4]],
        }
