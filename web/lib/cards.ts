import type { Card, Suit } from "./types";

/** Build display-only cards from codes like "5H 6H 7H JK" (ids are synthetic). */
export function cards(codes: string, wildRank?: number): Card[] {
  return codes
    .trim()
    .split(/\s+/)
    .map((code, i) => {
      if (code === "JK") return { id: 10000 + i, code, rank: null, suit: null, isJoker: true, isWild: true };
      const suit = code.slice(-1) as Suit;
      const r = code.slice(0, -1);
      const rank = r === "A" ? 1 : r === "J" ? 11 : r === "Q" ? 12 : r === "K" ? 13 : Number(r);
      return { id: 10000 + i, code, rank, suit, isJoker: false, isWild: wildRank === rank };
    });
}

export const RANK_NAMES = ["", "Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King"];

export function sortHand(hand: Card[], by: "suit" | "rank"): Card[] {
  const suitOrder: Record<string, number> = { S: 0, H: 1, C: 2, D: 3 };
  const key = (c: Card) =>
    c.isJoker ? [9, 99] : by === "suit" ? [suitOrder[c.suit!], c.rank!] : [c.rank!, suitOrder[c.suit!]];
  return [...hand].sort((a, b) => {
    const ka = key(a), kb = key(b);
    return ka[0] - kb[0] || ka[1] - kb[1];
  });
}
