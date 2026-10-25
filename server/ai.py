"""Difficulty -> policy mapping (SPEC §5.8) and model loading.

models/model_card.json (written in Phase 5) selects the policy per level:
    {"levels": {"easy":   {"type": "heuristic", "epsilon": 0.35, "forfeit": false},
                "medium": {"type": "dqn", "checkpoint": "models/ckpt_medium.pth"},
                "hard":   {"type": "hybrid", "checkpoint": "models/dqn_model.pth"}},
     "coach": {"dqn": "models/dqn_model.pth", "value": "models/value_model.pth"}}
Without a model card every level falls back to heuristic variants, so the server always works.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from indian_rummy.agents.heuristic import HeuristicAgent
from indian_rummy.features import Analysis, analyse

log = logging.getLogger("qrummy.ai")
ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

DEFAULT_LEVELS = {
    "easy": {"type": "heuristic", "epsilon": 0.35, "forfeit": False},
    "medium": {"type": "heuristic", "epsilon": 0.10, "forfeit": True},
    "hard": {"type": "heuristic", "epsilon": 0.0, "forfeit": True},
}


class Models:
    """Lazily loaded, shared across sessions."""

    def __init__(self):
        self.card = {}
        path = MODELS / "model_card.json"
        if path.exists():
            self.card = json.loads(path.read_text())
        self._dqn = {}
        self._value = None
        self.value_loaded = False

    def dqn(self, rel: str | None):
        if not rel:
            return None
        if rel not in self._dqn:
            try:
                from indian_rummy.agents.policy import DQNPolicy
                self._dqn[rel] = DQNPolicy.load(str(ROOT / rel), device="cpu")
            except Exception as e:  # missing / incompatible model -> heuristic fallback
                log.warning("cannot load DQN %s: %s", rel, e)
                self._dqn[rel] = None
        return self._dqn[rel]

    @property
    def coach_dqn(self):
        return self.dqn(self.card.get("coach", {}).get("dqn"))

    @property
    def value(self):
        if self._value is None and not self.value_loaded:
            self.value_loaded = True
            rel = self.card.get("coach", {}).get("value")
            if rel:
                try:
                    from indian_rummy.agents.value import ValueModel
                    self._value = ValueModel.load(str(ROOT / rel), device="cpu")
                except Exception as e:
                    log.warning("cannot load value model %s: %s", rel, e)
        return self._value

    def level(self, difficulty: str) -> dict:
        return self.card.get("levels", {}).get(difficulty) or DEFAULT_LEVELS[difficulty]


class AIPlayer:
    def __init__(self, models: Models, difficulty: str, seed: int | None = None):
        spec = models.level(difficulty)
        self.spec = spec
        self.heuristic = HeuristicAgent(epsilon=spec.get("epsilon", 0.0),
                                        allow_forfeit=spec.get("forfeit", True), seed=seed)
        self.policy = None
        if spec["type"] in ("dqn", "hybrid"):
            dqn = models.dqn(spec.get("checkpoint"))
            if dqn is not None:
                if spec["type"] == "hybrid":
                    from indian_rummy.agents.policy import HybridPolicy
                    self.policy = HybridPolicy(dqn, k=spec.get("k", 3))
                else:
                    self.policy = dqn
        self.illegal_count = 0

    def act(self, game, pid: int):
        state = game.get_state(pid)
        an: Analysis = analyse(state, game.types, game.variant)
        if self.policy is not None:
            a = self.policy.act_analysis(an)
        else:
            a = self.heuristic.act(an)
        if a not in an.legal:  # never trust a model blindly (SPEC §5.3)
            self.illegal_count += 1
            log.error("illegal AI action %s, legal %s", a, an.legal)
            a = self.heuristic.act(an)
        return a
