"use client";

import { AnimatePresence, motion } from "motion/react";
import { useEffect, useMemo, useRef, useState } from "react";
import PlayingCard, { cardLabel } from "@/components/PlayingCard";
import { RANK_NAMES } from "@/lib/cards";
import type { useGame } from "@/lib/useGame";
import Hand, { type SortMode } from "./Hand";
import { RevealModal, RoundSummary, ScoreBoard } from "./Modals";
import TutorPanel from "./TutorPanel";

type Game = ReturnType<typeof useGame>;

function useCardWidth(ref: React.RefObject<HTMLDivElement | null>) {
  const [w, setW] = useState(64);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const avail = el.clientWidth - 24;
      setW(Math.max(34, Math.min(78, Math.floor(avail / (14 * 0.6 + 0.6)))));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [ref]);
  return w;
}

export default function Table({
  game,
  coachEnabled,
  setCoachEnabled,
  guided,
  onNewGame,
}: {
  game: Game;
  coachEnabled: boolean;
  setCoachEnabled: (v: boolean) => void;
  guided?: boolean;
  onNewGame: () => void;
}) {
  const { view, display, flight, busy, banner, roundKey, act, log } = game;
  const tableRef = useRef<HTMLDivElement>(null);
  const drawRef = useRef<HTMLDivElement>(null);
  const discardRef = useRef<HTMLDivElement>(null);
  const handRef = useRef<HTMLDivElement>(null);
  const aiRef = useRef<HTMLDivElement>(null);
  const cardW = useCardWidth(tableRef);
  const [sort, setSort] = useState<SortMode>("suit");
  const [selected, setSelected] = useState<number | null>(null);
  const [highlight, setHighlight] = useState<number[]>([]);
  const [confirmForfeit, setConfirmForfeit] = useState(false);

  useEffect(() => setSelected(null), [view?.phase, view?.you.hand.length]);

  const discardable = useMemo(() => new Set(view?.legal.discardIds ?? []), [view]);
  if (!view || !display) return null;

  const myTurn = !busy && (view.phase === "your_draw" || view.phase === "your_discard");
  const top = display.pile[display.pile.length - 1] ?? null;
  const wild = view.you.wildRank;
  const cjCard = view.table.cj.card;

  const rect = (which: "draw" | "discard" | "you" | "ai") => {
    const el = { draw: drawRef, discard: discardRef, you: handRef, ai: aiRef }[which].current;
    const base = tableRef.current?.getBoundingClientRect();
    if (!el || !base) return { x: 0, y: 0 };
    const r = el.getBoundingClientRect();
    return { x: r.left - base.left + r.width / 2 - cardW / 2, y: r.top - base.top + r.height / 2 - (cardW * 1.4) / 2 };
  };

  const doDiscard = (id: number) => {
    if (!discardable.has(id)) return;
    setSelected(null);
    act({ type: "discard", card_id: id });
  };

  const status = (() => {
    switch (view.phase) {
      case "your_draw":
        return view.legal.drawOpen
          ? "Your turn — draw from the closed pile or take the open card"
          : view.you.postOriginal
            ? "Your turn — draw a card"
            : "Your turn — draw from the closed pile (the open card only if it completes your original)";
      case "your_discard":
        return selected !== null ? "Discard the selected card?" : "Select a card to discard (double-click to discard instantly)";
      case "ai_turn":
        return "AI's turn";
      case "reveal":
        return "Reveal your original rummy";
      default:
        return "";
    }
  })();

  return (
    <div className="grid lg:grid-cols-[1fr_320px] gap-4">
      <div className="flex flex-col gap-3 min-w-0">
        <ScoreBoard view={view} />
        <div ref={tableRef} className="felt relative rounded-[1.75rem] p-3 sm:p-5 overflow-hidden">
          {/* AI side */}
          <div className="flex items-start justify-between gap-3">
            <div ref={aiRef} className="flex items-center">
              {Array.from({ length: display.aiCount }).map((_, i) => (
                <motion.div key={i} layout style={{ marginLeft: i ? -cardW * 0.72 : 0 }}>
                  <PlayingCard width={cardW * 0.7} />
                </motion.div>
              ))}
              <span className="ml-3 text-xs text-white/70">
                AI · {display.aiCount} cards{view.ai.postOriginal ? " · has original" : ""}
              </span>
            </div>
            {display.aiOriginal && (
              <div className="flex flex-col items-end gap-1">
                <span className="text-[10px] uppercase tracking-wider text-gold-soft">AI original</span>
                <div className="flex gap-0.5">
                  {display.aiOriginal.map((c) => (
                    <motion.div key={c.id} initial={{ scale: 0.6, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}>
                      <PlayingCard card={c} width={cardW * 0.6} />
                    </motion.div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* centre */}
          <div className="flex items-center justify-center gap-6 sm:gap-10 my-5 sm:my-7">
            <div className="flex flex-col items-center gap-1">
              <button
                ref={drawRef as unknown as React.RefObject<HTMLButtonElement>}
                disabled={!myTurn || view.phase !== "your_draw" || !view.legal.drawClosed}
                onClick={() => act({ type: "draw", source: "closed" })}
                className="relative rounded-md disabled:cursor-default enabled:hover:-translate-y-1 transition-transform enabled:ring-2 enabled:ring-gold/70"
                aria-label="Draw from the closed pile"
              >
                <PlayingCard width={cardW} />
                <span className="absolute -bottom-2 -right-2 text-[10px] rounded-full bg-black/70 px-1.5 py-0.5">
                  {display.drawCount}
                </span>
              </button>
              <span className="text-[11px] text-white/70">Closed pile</span>
            </div>
            <div className="flex flex-col items-center gap-1">
              <button
                ref={discardRef as unknown as React.RefObject<HTMLButtonElement>}
                disabled={!myTurn || view.phase !== "your_draw" || !view.legal.drawOpen}
                onClick={() => act({ type: "draw", source: "open" })}
                className="rounded-md disabled:cursor-default enabled:hover:-translate-y-1 transition-transform enabled:ring-2 enabled:ring-gold/70"
                aria-label={top ? `Take ${cardLabel(top)} from the discard pile` : "Discard pile"}
              >
                <AnimatePresence mode="popLayout">
                  {top ? (
                    <motion.div key={top.id} initial={{ scale: 0.9 }} animate={{ scale: 1 }}>
                      <PlayingCard card={top} width={cardW} />
                    </motion.div>
                  ) : (
                    <div style={{ width: cardW, height: cardW * 1.4 }} className="rounded-md border-2 border-dashed border-white/25" />
                  )}
                </AnimatePresence>
              </button>
              <span className="text-[11px] text-white/70">Discard pile</span>
            </div>
            <div className="flex flex-col items-center gap-1">
              <motion.div
                key={cjCard ? "up" : "down"}
                initial={{ rotateY: 90 }}
                animate={{ rotateY: 0 }}
                transition={{ duration: 0.45 }}
                className="rotate-90 origin-center"
              >
                <PlayingCard card={cjCard} width={cardW * 0.8} highlight={cjCard ? "gold" : null} />
              </motion.div>
              <span className="text-[11px] text-white/70 mt-2 text-center">
                {view.table.cj.faceUp ? "Open joker" : "Closed joker"}
                <br />
                {wild === "none"
                  ? "no wild rank"
                  : typeof wild === "number"
                    ? <b className="text-gold-soft">Wild: {RANK_NAMES[wild]}s</b>
                    : "wild rank hidden"}
              </span>
            </div>
          </div>

          <AnimatePresence>
            {banner && (
              <motion.div
                key={banner}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="absolute left-1/2 top-[44%] -translate-x-1/2 z-30 rounded-full bg-black/75 px-4 py-2 text-sm whitespace-nowrap"
              >
                {banner}
              </motion.div>
            )}
          </AnimatePresence>

          <p className="text-center text-sm text-white/85 min-h-5">{status}</p>

          <Hand
            ref={handRef}
            hand={display.hand}
            lockedIds={display.lockedIds}
            groups={view.you.suggestedGroups}
            sort={sort}
            selected={selected}
            discardable={discardable}
            drawnId={view.you.drawnId}
            highlightIds={new Set(highlight)}
            cardWidth={cardW}
            roundKey={roundKey}
            onSelect={(id) => setSelected((s) => (s === id ? null : id))}
            onDiscard={doDiscard}
          />

          <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
            <div className="flex gap-1 text-xs">
              {(["suit", "rank", "groups"] as SortMode[]).map((m) => (
                <button
                  key={m}
                  onClick={() => setSort(m)}
                  className={`px-2.5 py-1 rounded-md ${sort === m ? "bg-black/50 text-white" : "bg-black/20 text-white/70 hover:text-white"}`}
                >
                  {m === "groups" ? "Auto-group" : `By ${m}`}
                </button>
              ))}
            </div>
            <div className="flex gap-2">
              {view.legal.forfeit && myTurn && (
                <button
                  onClick={() => setConfirmForfeit(true)}
                  className="px-3 py-1.5 rounded-md text-xs bg-black/30 text-white/80 hover:bg-bad hover:text-white"
                >
                  Forfeit (40 pts)
                </button>
              )}
              <button
                disabled={!myTurn || view.phase !== "your_discard" || selected === null || !discardable.has(selected)}
                onClick={() => selected !== null && doDiscard(selected)}
                className="px-4 py-1.5 rounded-md text-sm font-semibold bg-gold text-black disabled:opacity-35"
              >
                Discard
              </button>
            </div>
          </div>

          {/* flying card overlay */}
          <AnimatePresence>
            {flight && (
              <motion.div
                key={flight.key}
                className="absolute z-40 pointer-events-none"
                initial={{ ...rect(flight.from), scale: 1 }}
                animate={{ ...rect(flight.to), scale: flight.to === "ai" ? 0.7 : 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.33, ease: "easeInOut" }}
                style={{ left: 0, top: 0 }}
              >
                <PlayingCard card={flight.card} width={cardW} />
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <details className="rounded-xl bg-panel border border-line p-3 text-xs text-muted">
          <summary className="cursor-pointer">Move log</summary>
          <ul className="mt-2 space-y-0.5 max-h-40 overflow-y-auto">
            {[...log].reverse().map((e) => (
              <li key={e.seq + e.type}>{describe(e)}</li>
            ))}
          </ul>
        </details>
      </div>

      <div className="flex flex-col gap-3">
        {!guided && (
          <label className="flex items-center justify-between rounded-xl bg-panel border border-line p-3 text-sm">
            <span>Coach</span>
            <input type="checkbox" checked={coachEnabled} onChange={(e) => setCoachEnabled(e.target.checked)} className="accent-[#d9b24a] w-4 h-4" />
          </label>
        )}
        <TutorPanel
          view={view}
          enabled={coachEnabled || !!guided}
          alwaysOn={guided}
          busy={busy}
          onPlay={(a) => act(a)}
          onHighlight={setHighlight}
        />
        {game.error && (
          <div className="rounded-xl border border-bad/50 bg-bad/10 p-3 text-sm">
            {game.error}
            <button className="block mt-2 underline" onClick={() => game.setError(null)}>
              dismiss
            </button>
          </div>
        )}
      </div>

      <RevealModal view={view} busy={busy} onReveal={(ids) => act({ type: "reveal", card_ids: ids })} />
      <RoundSummary view={view} busy={busy} onNext={() => (view.mode === "round" ? onNewGame() : act({ type: "next_round" }))} onNewGame={onNewGame} />

      <AnimatePresence>
        {confirmForfeit && (
          <motion.div className="fixed inset-0 z-50 bg-black/60 flex items-center justify-center p-4" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <div className="bg-panel border border-line rounded-2xl p-5 max-w-sm">
              <h3 className="font-semibold text-lg">Forfeit this round?</h3>
              <p className="text-sm text-muted mt-1">
                You take a fixed 40 points{view.multiplier === 2 ? " (×2 this round = 80)" : ""} and the AI wins the round.
              </p>
              <div className="mt-4 flex gap-2 justify-end">
                <button className="px-3 py-2 text-sm" onClick={() => setConfirmForfeit(false)}>
                  Keep playing
                </button>
                <button
                  className="px-3 py-2 text-sm rounded-md bg-bad text-white"
                  onClick={() => {
                    setConfirmForfeit(false);
                    act({ type: "forfeit" });
                  }}
                >
                  Forfeit
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function describe(e: { actor: string; type: string; source?: string; card?: { rank: number | null; suit: string | null; isJoker: boolean } }) {
  const who = e.actor === "you" ? "You" : e.actor === "ai" ? "AI" : "Dealer";
  const c = e.card ? cardLabel(e.card as never) : "a card";
  switch (e.type) {
    case "draw":
      return `${who} drew ${e.source === "open" ? c + " from the discard pile" : e.actor === "you" ? c : "from the closed pile"}`;
    case "discard":
      return `${who} discarded ${c}`;
    case "reveal":
      return `${who} revealed an original rummy`;
    case "declare":
      return `${who} declared rummy`;
    case "forfeit":
      return `${who} forfeited`;
    case "reshuffle":
      return "The discards were reshuffled into a new draw pile";
    case "cj_flip":
      return "The closed joker was turned face up";
    case "deal":
      return "New round dealt";
    default:
      return `${who}: ${e.type}`;
  }
}
