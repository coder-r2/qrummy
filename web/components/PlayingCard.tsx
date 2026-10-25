"use client";

import type { Card } from "@/lib/types";

const SUIT_GLYPH: Record<string, string> = { S: "♠", H: "♥", D: "♦", C: "♣" };
const RANK_LABEL = ["", "A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"];

export function cardLabel(c: Pick<Card, "rank" | "suit" | "isJoker">): string {
  if (c.isJoker || !c.rank || !c.suit) return "Joker";
  return `${RANK_LABEL[c.rank]}${SUIT_GLYPH[c.suit]}`;
}

type Props = {
  card?: Card | null;          // null/undefined => face down
  width?: number;
  selected?: boolean;
  locked?: boolean;
  highlight?: "gold" | "green" | "red" | null;
  dim?: boolean;
  className?: string;
};

export default function PlayingCard({ card, width = 72, selected, locked, highlight, dim, className = "" }: Props) {
  const h = Math.round(width * 1.4);
  const ring =
    highlight === "gold" ? "ring-2 ring-gold" : highlight === "green" ? "ring-2 ring-good" : highlight === "red" ? "ring-2 ring-bad" : "";
  return (
    <div
      className={`relative select-none rounded-[7%] card-shadow ${ring} ${selected ? "-translate-y-3" : ""} ${
        dim ? "opacity-45" : ""
      } transition-transform ${className}`}
      style={{ width, height: h }}
      aria-label={card ? cardLabel(card) : "face-down card"}
    >
      {card ? <Face card={card} /> : <Back />}
      {card?.isWild && !card.isJoker && (
        <span className="absolute -top-1.5 -right-1.5 rounded-full bg-gold text-[9px] font-bold text-black px-1.5 py-0.5 shadow">
          WILD
        </span>
      )}
      {locked && (
        <span className="absolute -bottom-1.5 left-1/2 -translate-x-1/2 rounded-full bg-gold text-[9px] font-bold text-black px-1.5 py-0.5 shadow">
          🔒 ORIGINAL
        </span>
      )}
    </div>
  );
}

function Face({ card }: { card: Card }) {
  if (card.isJoker || !card.rank || !card.suit) {
    return (
      <svg viewBox="0 0 100 140" className="w-full h-full">
        <rect x="1" y="1" width="98" height="138" rx="7" fill="var(--color-card)" stroke="#d6d0c2" />
        <text x="10" y="22" fontSize="15" fontWeight="800" fill="#7b2cbf" fontFamily="var(--font-display)">
          J
        </text>
        <text x="50" y="80" fontSize="44" textAnchor="middle" fill="#7b2cbf">
          ★
        </text>
        <text x="50" y="112" fontSize="12" fontWeight="800" textAnchor="middle" fill="#7b2cbf" letterSpacing="2">
          JOKER
        </text>
      </svg>
    );
  }
  const red = card.suit === "H" || card.suit === "D";
  const color = red ? "var(--color-suit-red)" : "var(--color-card-ink)";
  const glyph = SUIT_GLYPH[card.suit];
  const label = RANK_LABEL[card.rank];
  const face = card.rank >= 11;
  return (
    <svg viewBox="0 0 100 140" className="w-full h-full">
      <rect x="1" y="1" width="98" height="138" rx="7" fill="var(--color-card)" stroke="#d6d0c2" />
      <g fill={color} fontFamily="var(--font-sans)" fontWeight="800">
        <text x="8" y="24" fontSize={label.length > 1 ? 19 : 22} letterSpacing="-1">
          {label}
        </text>
        <text x="9" y="44" fontSize="18">
          {glyph}
        </text>
        <g transform="rotate(180 50 70)">
          <text x="8" y="24" fontSize={label.length > 1 ? 19 : 22} letterSpacing="-1">
            {label}
          </text>
          <text x="9" y="44" fontSize="18">
            {glyph}
          </text>
        </g>
        {face ? (
          <>
            <rect x="26" y="34" width="48" height="72" rx="4" fill="none" stroke={color} strokeWidth="1.5" opacity="0.6" />
            <text x="50" y="82" fontSize="38" textAnchor="middle" fontFamily="var(--font-display)">
              {label}
            </text>
          </>
        ) : (
          <text x="50" y="86" fontSize="46" textAnchor="middle">
            {glyph}
          </text>
        )}
      </g>
    </svg>
  );
}

function Back() {
  return (
    <svg viewBox="0 0 100 140" className="w-full h-full">
      <defs>
        <pattern id="lattice" width="10" height="10" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <rect width="10" height="10" fill="#6b1d2a" />
          <path d="M0 0 L0 10" stroke="#d9b24a" strokeWidth="1" opacity="0.45" />
        </pattern>
      </defs>
      <rect x="1" y="1" width="98" height="138" rx="7" fill="#f3ead6" />
      <rect x="6" y="6" width="88" height="128" rx="5" fill="url(#lattice)" stroke="#d9b24a" strokeWidth="1.5" />
      <circle cx="50" cy="70" r="13" fill="#6b1d2a" stroke="#d9b24a" strokeWidth="1.5" />
      <text x="50" y="76" fontSize="16" textAnchor="middle" fill="#d9b24a">
        ♠
      </text>
    </svg>
  );
}
