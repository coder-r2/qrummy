"use client";

import { AnimatePresence, motion } from "motion/react";
import { forwardRef, useMemo } from "react";
import PlayingCard from "@/components/PlayingCard";
import { sortHand } from "@/lib/cards";
import type { Card } from "@/lib/types";

export type SortMode = "suit" | "rank" | "groups";

type Props = {
  hand: Card[];
  lockedIds: number[];
  groups: number[][];
  sort: SortMode;
  selected: number | null;
  discardable: Set<number>;
  drawnId: number | null;
  highlightIds?: Set<number>;
  cardWidth: number;
  onSelect: (id: number) => void;
  onDiscard: (id: number) => void;
  roundKey: number;
};

/** Orders the hand for display: locked original first, then by the chosen sort. */
function ordered(hand: Card[], lockedIds: number[], groups: number[][], sort: SortMode): (Card | "gap")[] {
  const locked = hand.filter((c) => lockedIds.includes(c.id));
  const rest = hand.filter((c) => !lockedIds.includes(c.id));
  const out: (Card | "gap")[] = [...locked];
  if (locked.length) out.push("gap");
  if (sort === "groups") {
    const byId = new Map(rest.map((c) => [c.id, c]));
    for (const g of groups) {
      const members = g.map((id) => byId.get(id)).filter(Boolean) as Card[];
      if (!members.length) continue;
      members.forEach((c) => byId.delete(c.id));
      out.push(...members, "gap");
    }
    out.push(...sortHand([...byId.values()], "suit"));
  } else {
    out.push(...sortHand(rest, sort));
  }
  while (out.length && out[out.length - 1] === "gap") out.pop();
  return out;
}

const Hand = forwardRef<HTMLDivElement, Props>(function Hand(p, ref) {
  const items = useMemo(() => ordered(p.hand, p.lockedIds, p.groups, p.sort), [p.hand, p.lockedIds, p.groups, p.sort]);
  const n = p.hand.length;
  const overlap = n > 0 ? Math.min(0, -(p.cardWidth * 0.42)) : 0;
  return (
    <div ref={ref} className="flex justify-center items-end min-h-[140px] px-2 pt-6" role="list" aria-label="Your hand">
      <AnimatePresence initial={true}>
        {items.map((it, i) =>
          it === "gap" ? (
            <div key={`gap-${i}`} style={{ width: p.cardWidth * 0.25 }} />
          ) : (
            <motion.button
              key={`${p.roundKey}-${it.id}`}
              layout
              role="listitem"
              initial={{ opacity: 0, y: 60, rotate: -6 }}
              animate={{ opacity: 1, y: 0, rotate: 0 }}
              exit={{ opacity: 0, y: -40, transition: { duration: 0.15 } }}
              transition={{ type: "spring", stiffness: 380, damping: 30, delay: Math.min(i, 14) * 0.025 }}
              style={{ marginLeft: i === 0 ? 0 : overlap, zIndex: i }}
              className="relative focus:outline-none focus-visible:ring-2 focus-visible:ring-gold rounded-md"
              onClick={() => p.onSelect(it.id)}
              onDoubleClick={() => p.discardable.has(it.id) && p.onDiscard(it.id)}
              aria-pressed={p.selected === it.id}
              title={p.discardable.has(it.id) ? "Click to select, double-click to discard" : undefined}
            >
              <PlayingCard
                card={it}
                width={p.cardWidth}
                selected={p.selected === it.id}
                locked={p.lockedIds.includes(it.id)}
                highlight={
                  p.highlightIds?.has(it.id) ? "gold" : it.id === p.drawnId ? "green" : null
                }
              />
            </motion.button>
          ),
        )}
      </AnimatePresence>
    </div>
  );
});

export default Hand;
