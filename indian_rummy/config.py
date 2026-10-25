"""Game variants (SPEC R10) and match constants (R9)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Variant:
    name: str
    decks: int
    jokers: int                 # printed jokers in the whole deck
    hand_size: int
    template: tuple[int, ...]   # winning meld sizes, sorted (R6.1)
    turn_cap: int = 300         # A5
    forfeit_points: int = 40    # R7.6
    no_original_points: int = 80  # R7.2
    max_points: int = 80        # R7.3 cap

    @property
    def deck_size(self) -> int:
        return self.decks * 52 + self.jokers

    @property
    def triplets_possible(self) -> bool:
        return self.decks >= 3


V13 = Variant("v13", decks=3, jokers=6, hand_size=13, template=(3, 3, 3, 4))
V10 = Variant("v10", decks=2, jokers=4, hand_size=10, template=(3, 3, 4))
V7 = Variant("v7", decks=1, jokers=2, hand_size=7, template=(3, 4))

VARIANTS = {v.name: v for v in (V13, V10, V7)}

# Game of 7s: rounds 1 and 7 score double (R9.1).
MATCH_MULTIPLIERS = (2, 1, 1, 1, 1, 1, 2)
