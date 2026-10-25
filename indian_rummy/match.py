"""Game of 7s (SPEC R9): 7 rounds, rounds 1 and 7 doubled, lowest total wins, dealer alternates."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import MATCH_MULTIPLIERS, V13, Variant
from .game import RevealPolicy, RoundResult, RummyGame


@dataclass
class RoundRecord:
    round_no: int
    multiplier: int
    result: RoundResult

    @property
    def scored(self) -> list[int]:
        return [pts * self.multiplier for pts in self.result.points]


@dataclass
class Match:
    variant: Variant = V13
    num_players: int = 2
    seed: int | None = None
    reveal_policy: dict[int, RevealPolicy] | None = None
    multipliers: tuple[int, ...] = MATCH_MULTIPLIERS
    rounds: list[RoundRecord] = field(default_factory=list)
    game: RummyGame | None = None

    def __post_init__(self):
        self._rng = np.random.RandomState(self.seed)
        self.first_dealer = int(self._rng.randint(self.num_players))  # R1.6

    @property
    def total_rounds(self) -> int:
        return len(self.multipliers)

    @property
    def round_no(self) -> int:
        """1-based number of the current (or next) round."""
        return len(self.rounds) + (0 if self.game is None or self.game.is_over() else 1)

    @property
    def multiplier(self) -> int:
        return self.multipliers[max(self.round_no, 1) - 1]

    def start_round(self) -> RummyGame:
        if self.is_over():
            raise RuntimeError("match is over")
        k = len(self.rounds)
        self.game = RummyGame(self.variant, self.num_players, seed=int(self._rng.randint(2**31 - 1)),
                              dealer=(self.first_dealer + k) % self.num_players,   # R9.3
                              reveal_policy=self.reveal_policy)
        self.game.init_game()
        return self.game

    def finish_round(self) -> RoundRecord:
        assert self.game is not None and self.game.is_over()
        k = len(self.rounds)
        rec = RoundRecord(k + 1, self.multipliers[k], self.game.result)
        self.rounds.append(rec)
        return rec

    def totals(self) -> list[int]:
        tot = [0] * self.num_players
        for r in self.rounds:
            for pid, s in enumerate(r.scored):
                tot[pid] += s
        return tot

    def is_over(self) -> bool:
        return len(self.rounds) >= self.total_rounds

    def winner(self) -> int | None:
        """Lowest total wins; ties are a draw (A8) -> None."""
        tot = self.totals()
        best = min(tot)
        leaders = [pid for pid, t in enumerate(tot) if t == best]
        return leaders[0] if len(leaders) == 1 else None
