"use client";

import { useCallback, useRef, useState } from "react";
import { api, ApiError } from "./api";
import type { ActionReq, Card, CoachMode, Difficulty, GameEvent, GameView, Mode, Payload } from "./types";

/** What the table shows while events animate; replaced by the server view once the queue drains. */
export type Display = {
  hand: Card[];
  lockedIds: number[];
  aiCount: number;
  aiOriginal: Card[] | null;
  pile: (Card | null)[];          // known discard pile, top = last (public info, tracked client-side)
  drawCount: number;
  cjFaceUp: boolean;
};

export type Flight = {
  key: number;
  card: Card | null;               // null = face down
  from: "draw" | "discard" | "you" | "ai";
  to: "draw" | "discard" | "you" | "ai";
};

const DUR = { draw: 380, discard: 340, reveal: 700, think: 550, other: 250 };

function displayOf(v: GameView): Display {
  return {
    hand: v.you.hand,
    lockedIds: v.you.lockedIds,
    aiCount: v.ai.handCount,
    aiOriginal: v.ai.original,
    pile: v.table.discardTop ? [v.table.discardTop] : [],
    drawCount: v.table.drawCount,
    cjFaceUp: v.table.cj.faceUp,
  };
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export function useGame() {
  const [view, setView] = useState<GameView | null>(null);
  const [display, setDisplay] = useState<Display | null>(null);
  const [flight, setFlight] = useState<Flight | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [log, setLog] = useState<GameEvent[]>([]);
  const [banner, setBanner] = useState<string | null>(null);
  const [roundKey, setRoundKey] = useState(0);
  const flightKey = useRef(0);
  const dispRef = useRef<Display | null>(null);

  const setDisp = (d: Display) => {
    dispRef.current = d;
    setDisplay(d);
  };

  const fly = async (f: Omit<Flight, "key">, ms: number) => {
    flightKey.current += 1;
    setFlight({ ...f, key: flightKey.current });
    await sleep(ms);
    setFlight(null);
  };

  const animate = useCallback(async (events: GameEvent[], finalView: GameView) => {
    let thinkingShown = false;
    for (const ev of events) {
      const d = dispRef.current;
      if (!d) break;
      setLog((l) => [...l.slice(-40), ev]);
      if (ev.actor === "ai" && !thinkingShown && (ev.type === "draw" || ev.type === "forfeit")) {
        thinkingShown = true;
        setBanner("AI is thinking…");
        await sleep(DUR.think);
        setBanner(null);
      }
      if (ev.type === "deal") {
        setRoundKey((k) => k + 1);
        continue;
      }
      if (ev.type === "draw") {
        const open = ev.source === "open";
        const who = ev.actor === "you" ? "you" : "ai";
        if (open) setDisp({ ...d, pile: d.pile.slice(0, -1) });
        else setDisp({ ...d, drawCount: Math.max(0, d.drawCount - 1) });
        await fly({ card: ev.card ?? null, from: open ? "discard" : "draw", to: who }, DUR.draw);
        const d2 = dispRef.current!;
        if (who === "you" && ev.card) setDisp({ ...d2, hand: [...d2.hand, ev.card] });
        else if (who === "ai") setDisp({ ...d2, aiCount: d2.aiCount + 1 });
      } else if (ev.type === "discard" && ev.card) {
        const who = ev.actor === "you" ? "you" : "ai";
        const card = ev.card;
        if (who === "you") setDisp({ ...d, hand: d.hand.filter((c) => c.id !== card.id) });
        else setDisp({ ...d, aiCount: Math.max(0, d.aiCount - 1) });
        await fly({ card, from: who, to: "discard" }, DUR.discard);
        const d2 = dispRef.current!;
        setDisp({ ...d2, pile: [...d2.pile, card] });
      } else if (ev.type === "reveal" && ev.cards) {
        if (ev.actor === "you") setDisp({ ...d, lockedIds: ev.cards.map((c) => c.id) });
        else setDisp({ ...d, aiOriginal: ev.cards });
        setBanner(ev.actor === "you" ? "You revealed your original — the closed joker is yours!" : "The AI revealed its original rummy");
        await sleep(DUR.reveal);
        setBanner(null);
      } else if (ev.type === "cj_flip") {
        setDisp({ ...d, cjFaceUp: true });
        await sleep(DUR.other);
      } else if (ev.type === "reshuffle") {
        setDisp({ ...d, drawCount: ev.count ?? d.drawCount, pile: d.pile.slice(-1) });
        await sleep(DUR.other);
      } else if (ev.type === "forfeit") {
        setBanner(ev.actor === "you" ? "You forfeited the round" : "The AI forfeited the round");
        await sleep(900);
        setBanner(null);
      } else if (ev.type === "declare") {
        setBanner(ev.actor === "you" ? "Rummy! You declared." : "The AI declared rummy");
        await sleep(1100);
        setBanner(null);
      }
    }
    setView(finalView);
    setDisp(displayOf(finalView));
  }, []);

  const handle = useCallback(
    async (p: Promise<Payload>, firstLoad = false) => {
      setBusy(true);
      setError(null);
      try {
        const res = await p;
        if (firstLoad || res.events.some((e) => e.type === "deal")) {
          // new round: show the fresh deal, then animate anything the AI already did
          const pre = res.events.filter((e) => e.type === "deal" || e.type === "reveal");
          setView({ ...res.view, phase: res.view.phase });
          setDisp(displayOf(res.view));
          setRoundKey((k) => k + 1);
          setLog(pre);
          await animate(res.events.filter((e) => e.type !== "deal" && !(e.type === "reveal")), res.view);
        } else {
          await animate(res.events, res.view);
        }
      } catch (e) {
        setError(e instanceof ApiError ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    [animate],
  );

  const start = useCallback(
    (mode: Mode, difficulty: Difficulty, coach: CoachMode) => handle(api.newGame(mode, difficulty, coach), true),
    [handle],
  );

  const act = useCallback(
    (action: ActionReq) => {
      if (!view || busy) return;
      return handle(api.act(view.gameId, action));
    },
    [view, busy, handle],
  );

  return { view, display, flight, busy, error, log, banner, roundKey, start, act, setError };
}
