"""Discrete action ids shared by the engine, RL env and server (SPEC §5.3)."""

from .cards import NUM_TYPES, type_code

DRAW_CLOSED = 0
DRAW_OPEN = 1
DISCARD_BASE = 2
FORFEIT = DISCARD_BASE + NUM_TYPES  # 55
NUM_ACTIONS = FORFEIT + 1           # 56


def discard_action(t: int) -> int:
    return DISCARD_BASE + t


def is_discard(a: int) -> bool:
    return DISCARD_BASE <= a < FORFEIT


def discard_type(a: int) -> int:
    return a - DISCARD_BASE


def describe(a: int) -> str:
    if a == DRAW_CLOSED:
        return "draw closed"
    if a == DRAW_OPEN:
        return "draw open"
    if a == FORFEIT:
        return "forfeit"
    return f"discard {type_code(discard_type(a))}"
