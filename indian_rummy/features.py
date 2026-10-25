"""Observation encoder (SPEC §5.2) and per-decision analysis shared by the env, heuristic bot,
server AI and coach. NumPy only.

`analyse(state, types, variant)` reads one player's raw information set (RummyGame.get_state)
and returns an Analysis; `encode(analysis)` turns it into the float32 observation vector.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import actions as A
from .cards import JOKER, NUM_STD, NUM_TYPES, base_points, make_type, rank_of, suit_of, wild_types
from .config import Variant
from .solver import Arrangement, HandContext, discard_eval, is_win, solve

FEATURE_VERSION = 2

_STD_BITS, _JK_BITS = 3, 6
COUNT_BLOCK = NUM_STD * _STD_BITS + _JK_BITS   # 162
LOCK_BLOCK = NUM_STD * _STD_BITS               # 156
BLOCKS = [
    ("A_hand", COUNT_BLOCK),
    ("B_locked", LOCK_BLOCK),
    ("C_top", NUM_TYPES + 1),
    ("D_wild", 13 + 2),
    ("E_dead", COUNT_BLOCK),
    ("F_opp_known", COUNT_BLOCK),
    ("G_flags", 6 + NUM_TYPES),
    ("H_scalars", 3),
    ("I_discards", NUM_TYPES * 6),
    ("J_global", 7),
]
OBS_DIM = sum(n for _, n in BLOCKS)            # 1098
_OFFSETS = {}
_o = 0
for _name, _n in BLOCKS:
    _OFFSETS[_name] = _o
    _o += _n

_THR_STD = np.arange(_STD_BITS)
_THR_JK = np.arange(_JK_BITS)
_POINTS = np.array([base_points(t) for t in range(NUM_TYPES)], dtype=np.float32)


@dataclass
class DiscardInfo:
    points: int            # min deadwood after discarding this type
    outs: int              # unseen cards that would improve the hand after this discard
    has_original: bool     # pre-original: an original remains after this discard
    arrangement: Arrangement
    win: bool = False      # discarding this type declares a win (exact cover, R6)

    @property
    def uncovered(self) -> int:
        return len(self.arrangement.uncovered)


@dataclass
class Analysis:
    pid: int
    decision: str                        # "draw" | "discard" | "reveal" | "none"
    counts: np.ndarray                   # own hand counts (53)
    ctx: HandContext
    post_original: bool
    wild_rank: int | None                # known wild rank (None if unknown or no wild)
    wild_known: bool
    legal: list[int]
    unseen: np.ndarray                   # copies not visible to this player (53)
    dead: np.ndarray                     # discard pile below the top (53)
    opp_known: np.ndarray                # opponent's known holdings (53)
    opp_locked: np.ndarray
    top: int | None                      # top discard type
    drawn: int | None                    # type drawn this turn
    drawn_source: str | None
    state: dict
    current: Arrangement | None = None   # best arrangement of the hand as it stands
    original_outs: int = 0
    has_original: bool = False
    open_legal: bool = False
    open_best_points: int | None = None
    open_best_discard: int | None = None
    open_win: bool = False
    discards: dict[int, DiscardInfo] = field(default_factory=dict)
    phi: float = 0.0                     # shaping potential (§5.4)
    types: tuple = ()                    # physical id -> type

    @property
    def my_discard_types(self) -> set[int]:
        return {self.types[c] for c in self.state.get("my_discards", [])}

    @property
    def min_deadwood(self) -> int:
        if self.decision == "discard" and self.discards:
            return min(d.points for d in self.discards.values())
        return self.current.points if self.current is not None else 0

    def is_wild(self, t: int) -> bool:
        return t == JOKER or (self.post_original and self.wild_rank is not None and rank_of(t) == self.wild_rank)


# --------------------------------------------------------------------------- helpers

def _counts(cards, types) -> np.ndarray:
    c = np.zeros(NUM_TYPES, dtype=np.int16)
    for card in cards:
        c[types[card]] += 1
    return c


def _neighbours(t: int) -> list[int]:
    """Same-suit types within rank distance 2 (Ace both low and high, no wrap)."""
    s, r = suit_of(t), rank_of(t)
    out = []
    for pos in ([1, 14] if r == 1 else [r]):
        for d in (-2, -1, 1, 2):
            q = pos + d
            if 1 <= q <= 14:
                out.append(make_type(s, 1 if q == 14 else q))
    return out


def _completers(x: int, y: int, triplets: bool) -> set[int]:
    """Types that turn the pair (x, y) into a 3-card pure sequence or triplet (an original)."""
    if x == y:
        return {x} if triplets else set()
    if suit_of(x) != suit_of(y):
        return set()
    out = set()
    s = suit_of(x)
    for rx in ([1, 14] if rank_of(x) == 1 else [rank_of(x)]):
        for ry in ([1, 14] if rank_of(y) == 1 else [rank_of(y)]):
            lo, hi = min(rx, ry), max(rx, ry)
            if hi - lo == 1:
                cands = [lo - 1, hi + 1]
            elif hi - lo == 2:
                cands = [lo + 1]
            else:
                continue
            for q in cands:
                if 1 <= q <= 14:
                    out.add(make_type(s, 1 if q == 14 else q))
    return out


def original_outs(counts: np.ndarray, unseen: np.ndarray, triplets: bool) -> int:
    """Unseen copies that would complete a natural original with two cards already held."""
    held = [t for t in range(NUM_STD) if counts[t]]
    comp: set[int] = set()
    for i, x in enumerate(held):
        if counts[x] >= 2 and triplets:
            comp.add(x)
        for y in held[i + 1:]:
            comp |= _completers(x, y, triplets)
    return int(sum(unseen[t] for t in comp))


def meld_outs(arr: Arrangement, unseen: np.ndarray, wild_rank: int | None, post_original: bool,
              triplets: bool) -> int:
    """Approximate outs: unseen cards that complete a meld from two uncovered cards, or any wild
    if at least two natural cards are uncovered (documented approximation, D-058)."""
    unc = [t for t in arr.uncovered if t != JOKER and not (post_original and wild_rank and rank_of(t) == wild_rank)]
    if len(unc) < 2:
        return 0
    comp: set[int] = set()
    for i, x in enumerate(unc):
        for y in unc[i + 1:]:
            if x == y:
                if triplets:
                    comp.add(x)
                continue
            if rank_of(x) == rank_of(y):
                comp |= {make_type(s, rank_of(x)) for s in range(4)} - {x, y}
            elif suit_of(x) == suit_of(y):
                comp |= _completers(x, y, False)
    wilds = {JOKER} | (set(wild_types(wild_rank)) if post_original else set())
    comp |= wilds
    return int(sum(unseen[t] for t in comp))


def _count_bits(c: np.ndarray) -> np.ndarray:
    std = (c[:NUM_STD, None] > _THR_STD).ravel()
    jk = c[JOKER] > _THR_JK
    return np.concatenate([std, jk])


# --------------------------------------------------------------------------- analysis

def analyse(state: dict, types, variant: Variant, light: bool = False) -> Analysis:
    """Compute everything a policy or the coach needs at this decision point.

    light=True skips the solver work (used for terminal / non-acting states).
    """
    pid = state["player"]
    counts = _counts(state["hand"], types)
    locked = tuple(sorted(types[c] for c in state["locked"]))
    post = state["post_original"]
    wild_known = state["wild_known"]
    wild_rank = state["wild_rank"] if wild_known else None
    discard_pile = state["discard_pile"]
    top = types[discard_pile[-1]] if discard_pile else None
    dead = _counts(discard_pile[:-1], types)
    opp = state["opponents"][0]
    opp_locked = _counts(opp["locked"], types)
    opp_known = _counts(opp["locked"] + opp["known_open"], types)
    drawn = types[state["drawn"]] if state["drawn"] is not None else None
    must = types[state["must_contain"]] if state["must_contain"] is not None else None

    total = np.full(NUM_TYPES, variant.decks, dtype=np.int16)
    total[JOKER] = variant.jokers
    unseen = total - counts - dead - opp_known
    if top is not None:
        unseen[top] -= 1
    if state["cj"] is not None:
        unseen[types[state["cj"]]] -= 1
    unseen = np.maximum(unseen, 0)

    if post:
        ctx = HandContext(post_original=True, wild_rank=state["wild_rank"], locked=locked,
                          template=variant.template, triplets=variant.triplets_possible)
    else:
        ctx = HandContext(post_original=False, must_contain=must,
                          template=variant.template, triplets=variant.triplets_possible)

    legal = list(state["legal_actions"])
    phase = state["phase"]
    if state["current_player"] != pid or phase == "over":
        decision = "none"
    else:
        decision = phase
    an = Analysis(pid=pid, decision=decision, counts=counts, ctx=ctx, post_original=post,
                  wild_rank=wild_rank, wild_known=wild_known, legal=legal, unseen=unseen, dead=dead,
                  opp_known=opp_known, opp_locked=opp_locked, top=top, drawn=drawn,
                  drawn_source=state["drawn_source"], state=state, types=types)
    if light or decision not in ("draw", "discard"):
        return an

    trip = variant.triplets_possible
    if not post:
        an.original_outs = original_outs(counts, unseen, trip)

    if decision == "draw":
        an.current = solve(counts, ctx)
        an.has_original = post or an.current.original is not None
        an.open_legal = A.DRAW_OPEN in legal
        if an.open_legal and top is not None:
            c14 = counts.copy()
            c14[top] += 1
            octx = ctx if post else HandContext(post_original=False, must_contain=top,
                                                template=variant.template, triplets=trip)
            cands = [t for t in range(NUM_TYPES) if c14[t] > locked.count(t) and t != top]
            evals = discard_eval(c14, octx, cands)
            if not post:  # R3.4: only discards that keep an original containing the open card
                evals = {t: e for t, e in evals.items() if e.original is not None}
            if evals:
                best_t = min(evals, key=lambda t: (evals[t].points, -base_points(t)))
                an.open_best_discard = best_t
                an.open_best_points = evals[best_t].points
                if an.open_best_points == 0:
                    an.open_win = any(
                        is_win(_minus(c14, t), octx) is not None for t, e in evals.items() if e.points == 0)
    else:  # discard
        cands = [A.discard_type(a) for a in legal if A.is_discard(a)]
        evals = discard_eval(counts, ctx, cands)
        for t, arr in evals.items():
            c13 = _minus(counts, t)
            if post:
                outs = meld_outs(arr, unseen, wild_rank, True, trip)
                has_orig = True
            else:
                has_orig = arr.original is not None
                outs = meld_outs(arr, unseen, None, False, trip) if has_orig else original_outs(c13, unseen, trip)
            win = arr.points == 0 and is_win(c13, ctx) is not None
            an.discards[t] = DiscardInfo(arr.points, outs, has_orig, arr, win)
        an.current = solve(counts, ctx)
        an.has_original = post or an.current.original is not None

    an.phi = potential(an)
    return an


def _minus(c: np.ndarray, t: int) -> np.ndarray:
    out = c.copy()
    out[t] -= 1
    return out


def potential(an: Analysis) -> float:
    """Shaping potential Φ (§5.4)."""
    if an.decision not in ("draw", "discard"):
        return 0.0
    if an.post_original:
        return -min(an.min_deadwood, 80) / 80.0
    if an.has_original:
        return -0.5
    return -0.5 - 0.3 * (1.0 - min(an.original_outs, 20) / 20.0)


# --------------------------------------------------------------------------- encoding

def encode(an: Analysis) -> np.ndarray:
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    st = an.state

    def put(name, vec):
        o = _OFFSETS[name]
        obs[o:o + len(vec)] = vec

    put("A_hand", _count_bits(an.counts))
    locked = np.zeros(NUM_TYPES, dtype=np.int16)
    for t in an.ctx.locked:
        locked[t] += 1
    put("B_locked", (locked[:NUM_STD, None] > _THR_STD).ravel())
    top = np.zeros(NUM_TYPES + 1, dtype=np.float32)
    top[an.top if an.top is not None else NUM_TYPES] = 1
    put("C_top", top)
    wild = np.zeros(15, dtype=np.float32)
    if not an.wild_known:
        wild[13] = 1
    elif an.wild_rank is None:
        wild[14] = 1
    else:
        wild[an.wild_rank - 1] = 1
    put("D_wild", wild)
    put("E_dead", _count_bits(an.dead))
    put("F_opp_known", _count_bits(an.opp_known))
    flags = np.zeros(6 + NUM_TYPES, dtype=np.float32)
    flags[0] = an.post_original
    flags[1] = st["opponents"][0]["post_original"]
    flags[2] = st["forfeit_window"]
    flags[3] = an.decision == "draw"
    flags[4] = an.decision == "discard"
    flags[5] = an.drawn_source == "open"
    if an.drawn is not None:
        flags[6 + an.drawn] = 1
    put("G_flags", flags)
    put("H_scalars", np.array([st["draw_count"] / 134.0, len(st["discard_pile"]) / 134.0,
                               min(st["turn"] / 100.0, 3.0)], dtype=np.float32))
    if an.discards:
        I = np.zeros((6, NUM_TYPES), dtype=np.float32)
        best = None
        if an.current is not None:
            best = {t for m in an.current.melds for t in m}
        for t, d in an.discards.items():
            I[0, t] = min(d.points, 80) / 80.0
            I[1, t] = 1.0 if best and t in best else 0.0
            I[2, t] = min(d.outs, 20) / 20.0
            I[3, t] = float(d.win)
            I[4, t] = min(d.uncovered, 13) / 13.0
            I[5, t] = float(d.has_original)
        put("I_discards", I.ravel())
    gain = 0.0
    if an.open_best_points is not None and an.current is not None:
        gain = float(np.clip((an.current.points - an.open_best_points) / 80.0, -1, 1))
    put("J_global", np.array([
        min(an.min_deadwood, 80) / 80.0,
        min(an.original_outs, 20) / 20.0,
        float(an.has_original),
        (len(an.current.uncovered) / 13.0) if an.current is not None else 0.0,
        gain,
        float(an.open_legal and not an.post_original),
        float(an.open_win),
    ], dtype=np.float32))
    return obs
