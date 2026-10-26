// API contract — mirrors server/views.py and server/coach.py (SPEC §7.4). Keep in sync.

export type Suit = "S" | "H" | "D" | "C";

export type Card = {
  id: number;
  code: string;
  rank: number | null;
  suit: Suit | null;
  isJoker: boolean;
  isWild: boolean;
};

export type Mode = "match" | "round";
export type Difficulty = "easy" | "medium" | "hard";
export type CoachMode = "off" | "on" | "guided";
export type Side = "you" | "ai";

export type Phase = "reveal" | "your_draw" | "your_discard" | "ai_turn" | "round_over" | "match_over";

export type RoundSide = { points: number; groups: number[][] };

export type GameView = {
  gameId: string;
  mode: Mode;
  difficulty: Difficulty;
  coach: CoachMode;
  round: number;
  totalRounds: number;
  multiplier: 1 | 2;
  scores: { you: number; ai: number };
  roundHistory: { round: number; multiplier: number; you: number; ai: number; winner: Side | "none" }[];
  phase: Phase;
  dealer: Side;
  you: {
    hand: Card[];
    lockedIds: number[];
    postOriginal: boolean;
    wildRank: number | null | "none";
    suggestedGroups: number[][];
    drawnId: number | null;
  };
  ai: { handCount: number; postOriginal: boolean; original: Card[] | null; hand: Card[] | null };
  table: {
    drawCount: number;
    discardTop: Card | null;
    discardCount: number;
    cj: { faceUp: boolean; card: Card | null };
  };
  legal: {
    drawClosed: boolean;
    drawOpen: boolean;
    discardIds: number[];
    forfeit: boolean;
    revealOptions: number[][];
    revealRecommended: number | null;
  };
  roundResult: null | {
    winner: Side | "none";
    reason: "declare" | "forfeit" | "exhausted";
    multiplier: number;
    you: RoundSide;
    ai: RoundSide;
  };
  matchWinner: Side | "draw" | null;
};

export type GameEvent = {
  seq: number;
  actor: Side | "dealer";
  type: "deal" | "draw" | "discard" | "reveal" | "forfeit" | "declare" | "reshuffle" | "cj_flip" | "round_end";
  source?: "closed" | "open";
  card?: Card;
  cards?: Card[];
  dealer?: Side;
  by?: Side; // declare events (same as actor)
  reason?: string;
  points?: { you: number; ai: number };
  count?: number;
};

export type Payload = { view: GameView; events: GameEvent[] };

export type ActionReq =
  | { type: "draw"; source: "closed" | "open" }
  | { type: "discard"; card_id: number }
  | { type: "reveal"; card_ids: number[] }
  | { type: "forfeit" }
  | { type: "next_round" };

export type HintAction = {
  action: ActionReq;
  label: string;
  q: number;
  preference: number;
  winProbability: number | null;
  deadwoodAfter: number | null;
  tags: string[];
};

export type Hint = {
  available: boolean;
  winProbability?: number | null;
  decision?: "draw" | "discard" | "forfeit_window";
  actions?: HintAction[];
  recommendedIndex?: number;
  explanation?: { text: string; source: "template" | "gemini" };
};

export type Explain = { text: string | null; source: "gemini" | "template" | "none" };
