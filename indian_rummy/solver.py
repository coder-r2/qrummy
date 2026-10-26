"""Fast arrangement solver (SPEC §4.3).

Hands are count vectors over the 53 card types. The search runs over the pattern tables in
patterns.py: pick the lowest remaining natural card and either leave it uncovered or cover it
with a pattern that contains it. With at most 4 melds the depth is tiny; memoisation on the
remaining counts makes repeated calls (e.g. one per candidate discard) cheap.

Wild handling. Printed jokers are always wild in secondary melds. For a post-original player
with a wild rank, cards of that rank ("flex" cards) join the wild pool: using one naturally is the
same as using it as a wild in that slot. The exception is the ">= 1 natural" rule (R5.5/R5.6),
which only matters for melds made purely of pool cards: those are valid iff they contain at
least one flex card. Unmelded pool cards score 0 (R7.5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Sequence

from .cards import JOKER, NUM_STD, NUM_TYPES, base_points, rank_of, wild_types
from .melds import can_extend_original
from .patterns import Pattern, original_patterns, patterns_by_type

_POINTS = tuple(base_points(t) for t in range(NUM_STD))
_MEMO_LIMIT = 2_000_000


@dataclass(frozen=True)
class HandContext:
    """Everything the solver needs to know about whose hand this is (R2, R4, R6)."""
    post_original: bool = False
    wild_rank: int | None = None        # the holder's known wild rank (ignored pre-original)
    locked: tuple[int, ...] = ()        # types of the revealed original (post-original)
    must_contain: int | None = None     # pre-original: the original must include this type (R3.4)
    template: tuple[int, ...] = (3, 3, 3, 4)
    triplets: bool = True

    @property
    def flex_rank(self) -> int | None:
        return self.wild_rank if self.post_original else None


@dataclass
class Arrangement:
    melds: list[list[int]]              # card types per meld; the original (if any) first
    uncovered: list[int]                # card types not in any meld
    points: int                         # R7.5 points of the uncovered cards (uncapped)
    original: list[int] | None = None   # the original as used here (incl. extension)
    extra: dict = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        return not self.uncovered


# --------------------------------------------------------------------------- tables / memo

class _Table:
    __slots__ = ("pbt", "dead", "win")

    def __init__(self, triplets: bool, flex_rank: int | None):
        self.pbt = patterns_by_type(triplets, flex_rank)
        self.dead: dict = {}
        self.win: dict = {}

    def trim(self):
        if len(self.dead) > _MEMO_LIMIT:
            self.dead.clear()
        if len(self.win) > _MEMO_LIMIT:
            self.win.clear()


@lru_cache(maxsize=None)
def _table(triplets: bool, flex_rank: int | None) -> _Table:
    return _Table(triplets, flex_rank)


@lru_cache(maxsize=None)
def _remove_slot(slots: tuple[int, ...], size: int) -> tuple[int, ...] | None:
    if size not in slots:
        return None
    lst = list(slots)
    lst.remove(size)
    return tuple(lst)


def _fit_slot(slots: tuple[int, ...], size: int) -> tuple[int, ...] | None:
    """Scoring only (R7.3): a meld may sit in a slot at least its size, so a loser may count
    3/3/3/3 with one card left over. Taking the exact size first never loses (a bigger slot holds more)."""
    for s in sorted(slots):
        if s >= size:
            lst = list(slots)
            lst.remove(s)
            return tuple(lst)
    return None


def _first(N: bytes) -> int:
    return len(N) - len(N.lstrip(b"\x00"))


def _points_of(N: bytes) -> int:
    return sum(c * _POINTS[t] for t, c in enumerate(N) if c)


def _take(N: bytes, nat) -> bytes | None:
    for t, c in nat:
        if N[t] < c:
            return None
    b = bytearray(N)
    for t, c in nat:
        b[t] -= c
    return bytes(b)


# --------------------------------------------------------------------------- deadwood search

def _dead(tbl: _Table, N: bytes, W: int, slots: tuple[int, ...]) -> int:
    key = (N, W, slots)
    hit = tbl.dead.get(key)
    if hit is not None:
        return hit[0]
    i = _first(N)
    if i == NUM_STD:
        res = (0, None)
    elif not slots:
        res = (_points_of(N), None)
    else:
        b = bytearray(N)
        b[i] -= 1
        best = _POINTS[i] + _dead(tbl, bytes(b), W, slots)
        choice = None
        for p in tbl.pbt[i]:
            if p.wilds > W:
                continue
            s2 = _fit_slot(slots, p.size)
            if s2 is None:
                continue
            N2 = _take(N, p.nat)
            if N2 is None:
                continue
            cost = _dead(tbl, N2, W - p.wilds, s2)
            if cost < best:
                best, choice = cost, p
                if best == 0:
                    break
        res = (best, choice)
    tbl.dead[key] = res
    return res[0]


def _dead_path(tbl: _Table, N: bytes, W: int, slots: tuple[int, ...]):
    melds: list[Pattern] = []
    uncovered: list[int] = []
    while True:
        _dead(tbl, N, W, slots)
        _, choice = tbl.dead[(N, W, slots)]
        i = _first(N)
        if i == NUM_STD:
            break
        if not slots:
            uncovered += [t for t, c in enumerate(N) for _ in range(c)]
            break
        if choice is None:
            uncovered.append(i)
            b = bytearray(N)
            b[i] -= 1
            N = bytes(b)
        else:
            melds.append(choice)
            N = _take(N, choice.nat)
            W -= choice.wilds
            slots = _fit_slot(slots, choice.size)
    return melds, uncovered


# --------------------------------------------------------------------------- exact-cover (win) search

def _pool_fits(jk: int, F: int, slots: tuple[int, ...]) -> bool:
    """Remaining pool cards form the remaining melds; each pure-pool meld needs a flex card."""
    return jk + F == sum(slots) and F >= len(slots)


def _win(tbl: _Table, N: bytes, jk: int, F: int, slots: tuple[int, ...]) -> bool:
    key = (N, jk, F, slots)
    hit = tbl.win.get(key)
    if hit is not None:
        return hit[0]
    i = _first(N)
    res = (False, None)
    if i == NUM_STD:
        res = (_pool_fits(jk, F, slots), None)
    elif slots:
        for p in tbl.pbt[i]:
            if p.wilds > jk + F:
                continue
            s2 = _remove_slot(slots, p.size)
            if s2 is None:
                continue
            N2 = _take(N, p.nat)
            if N2 is None:
                continue
            uj = min(jk, p.wilds)  # jokers first: keeps flex cards for pure-pool melds
            if _win(tbl, N2, jk - uj, F - (p.wilds - uj), s2):
                res = (True, p)
                break
    tbl.win[key] = res
    return res[0]


def _win_path(tbl: _Table, N: bytes, jk: int, F: int, slots: tuple[int, ...]):
    melds: list[Pattern] = []
    while _first(N) != NUM_STD:
        _, p = tbl.win[(N, jk, F, slots)]
        melds.append(p)
        N = _take(N, p.nat)
        uj = min(jk, p.wilds)
        jk, F = jk - uj, F - (p.wilds - uj)
        slots = _remove_slot(slots, p.size)
    return melds, slots


# --------------------------------------------------------------------------- helpers

def _counts(hand: Sequence[int] | Iterable[int]) -> list[int]:
    c = [int(x) for x in hand]
    if len(c) != NUM_TYPES:
        raise ValueError("hand must be a 53-long count vector")
    return c


def _sub_template(template: tuple[int, ...], size: int) -> tuple[int, ...] | None:
    return _remove_slot(tuple(sorted(template)), size)


@lru_cache(maxsize=None)
def _extensions(locked: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(e for e in range(NUM_STD) if can_extend_original(list(locked), e))


class _Pool:
    """Concrete jokers / flex cards, used to turn wild slots back into card types for display."""

    def __init__(self, jk: int, flex: dict[int, int]):
        self.jk = jk
        self.flex = dict(flex)

    def take_wild(self) -> int:
        if self.jk:
            self.jk -= 1
            return JOKER
        for t, c in self.flex.items():
            if c:
                self.flex[t] -= 1
                return t
        raise RuntimeError("wild pool exhausted")

    def take_flex(self) -> int:
        for t, c in self.flex.items():
            if c:
                self.flex[t] -= 1
                return t
        raise RuntimeError("no flex card")

    def rest(self) -> list[int]:
        return [JOKER] * self.jk + [t for t, c in self.flex.items() for _ in range(c)]


def _split(counts: list[int], ctx: HandContext):
    """-> natural counts (bytes, 52), jokers, flex counts by type."""
    c = list(counts)
    for t in ctx.locked:
        c[t] -= 1
        if c[t] < 0:
            raise ValueError("locked card not in hand")
    jk = c[JOKER]
    flex: dict[int, int] = {}
    if ctx.flex_rank is not None:
        for f in wild_types(ctx.flex_rank):
            if c[f]:
                flex[f] = c[f]
                c[f] = 0
    return bytes(c[:NUM_STD]), jk, flex


def _post_original_options(counts: list[int], ctx: HandContext):
    """Yield (original types, natural bytes, jk, flex dict, remaining slots) for the locked
    original as-is and for each available natural extension (R4.6)."""
    N, jk, flex = _split(counts, ctx)
    locked = list(ctx.locked)
    slots = _sub_template(ctx.template, len(locked))
    if slots is not None:
        yield locked, N, jk, flex, slots
    if len(locked) == 3:
        slots4 = _sub_template(ctx.template, 4)
        if slots4 is not None:
            for e in _extensions(tuple(sorted(locked))):
                if flex.get(e, 0):
                    f2 = dict(flex)
                    f2[e] -= 1
                    yield locked + [e], N, jk, f2, slots4
                elif N[e]:
                    yield locked + [e], _take(N, ((e, 1),)), jk, flex, slots4


def find_originals(hand: Sequence[int], must_contain: int | None = None,
                   triplets: bool = True, sizes: tuple[int, ...] = (3, 4)) -> list[list[int]]:
    """R4.1: every original (as card types) that can be formed from this hand."""
    counts = _counts(hand)
    out = []
    for p in original_patterns(triplets):
        if p.size not in sizes:
            continue
        if any(counts[t] < c for t, c in p.nat):
            continue
        types = p.natural_types()
        if must_contain is not None and must_contain not in types:
            continue
        out.append(types)
    return out


def _fill(patterns: list[Pattern], pool: _Pool) -> list[list[int]]:
    return [p.natural_types() + [pool.take_wild() for _ in range(p.wilds)] for p in patterns]


# --------------------------------------------------------------------------- public API

def solve(hand: Sequence[int], ctx: HandContext) -> Arrangement:
    """Minimum-deadwood arrangement within the template (R7.3; uncapped).

    Pre-original hands are solved *as if* an original were declared (best available original, or
    none if the hand has no original: then the result is an optimistic lower bound).
    """
    counts = _counts(hand)
    tbl = _table(ctx.triplets, ctx.flex_rank)
    tbl.trim()
    best = None
    if ctx.post_original:
        for orig, N, jk, flex, slots in _post_original_options(counts, ctx):
            cost = _dead(tbl, N, jk + sum(flex.values()), slots)
            if best is None or cost < best[0]:
                best = (cost, orig, N, jk, flex, slots)
        if best is None:
            raise ValueError("locked original does not fit the template")
    else:
        N, jk, _ = _split(counts, ctx)
        for orig in find_originals(counts, ctx.must_contain, ctx.triplets):
            slots = _sub_template(ctx.template, len(orig))
            if slots is None:
                continue
            N2 = _take(N, tuple((t, orig.count(t)) for t in set(orig)))
            cost = _dead(tbl, N2, jk, slots)
            if best is None or cost < best[0]:
                best = (cost, orig, N2, jk, {}, slots)
        if best is None:
            slots = tuple(sorted(ctx.template))
            best = (_dead(tbl, N, jk, slots), None, N, jk, {}, slots)
    cost, orig, N, jk, flex, slots = best
    pats, uncovered = _dead_path(tbl, N, jk + sum(flex.values()), slots)
    pool = _Pool(jk, flex)
    melds = _fill(pats, pool)
    if orig is not None:
        melds.insert(0, list(orig))
    return Arrangement(melds=melds, uncovered=uncovered + pool.rest(), points=cost,
                       original=list(orig) if orig is not None else None)


def deadwood(hand: Sequence[int], ctx: HandContext) -> int:
    return solve(hand, ctx).points


def is_win(hand: Sequence[int], ctx: HandContext) -> Arrangement | None:
    """R6: exact cover of the hand by the full template, including the original.

    Pre-original (same-turn win, R6.2): pass ctx.post_original=False; only printed jokers are wild.
    """
    counts = _counts(hand)
    if sum(counts) != sum(ctx.template):
        return None
    tbl = _table(ctx.triplets, ctx.flex_rank)
    tbl.trim()
    if ctx.post_original:
        options = _post_original_options(counts, ctx)
    else:
        def _pre():
            N, jk, _ = _split(counts, ctx)
            for orig in find_originals(counts, ctx.must_contain, ctx.triplets):
                slots = _sub_template(ctx.template, len(orig))
                if slots is not None:
                    yield orig, _take(N, tuple((t, orig.count(t)) for t in set(orig))), jk, {}, slots
        options = _pre()
    for orig, N, jk, flex, slots in options:
        F = sum(flex.values())
        if _win(tbl, N, jk, F, slots):
            pats, rest_slots = _win_path(tbl, N, jk, F, slots)
            pool = _Pool(jk, flex)
            melds = [list(orig)] + _fill(pats, pool)
            for size in rest_slots:  # pure-pool melds: one flex card + any pool cards
                m = [pool.take_flex()]
                m += [pool.take_wild() for _ in range(size - 1)]
                melds.append(m)
            return Arrangement(melds=melds, uncovered=[], points=0, original=list(orig))
    return None


def discard_eval(hand: Sequence[int], ctx: HandContext,
                 candidates: Iterable[int] | None = None) -> dict[int, Arrangement]:
    """solve(hand - t) for each candidate discard type t (default: every type in hand)."""
    counts = _counts(hand)
    if candidates is None:
        candidates = [t for t in range(NUM_TYPES) if counts[t]]
    out = {}
    for t in candidates:
        c = list(counts)
        c[t] -= 1
        out[t] = solve(c, ctx)
    return out
