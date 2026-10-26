"use client";

import { AnimatePresence, motion } from "motion/react";
import { useEffect, useState } from "react";
import PlayingCard from "@/components/PlayingCard";
import type { Card, GameView } from "@/lib/types";

function Overlay({ children }: { children: React.ReactNode }) {
  return (
    <motion.div
      className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <motion.div
        className="bg-panel border border-line rounded-2xl p-5 w-full max-w-2xl max-h-[90dvh] overflow-y-auto"
        initial={{ scale: 0.92, y: 20 }}
        animate={{ scale: 1, y: 0 }}
        transition={{ type: "spring", stiffness: 260, damping: 22 }}
      >
        {children}
      </motion.div>
    </motion.div>
  );
}

export function RevealModal({
  view,
  onReveal,
  busy,
}: {
  view: GameView;
  onReveal: (ids: number[]) => void;
  busy: boolean;
}) {
  const opts = view.legal.revealOptions;
  const [pick, setPick] = useState(view.legal.revealRecommended ?? 0);
  useEffect(() => setPick(view.legal.revealRecommended ?? 0), [view.legal.revealRecommended, opts.length]);
  const byId = new Map(view.you.hand.map((c) => [c.id, c]));
  return (
    <AnimatePresence>
      {view.phase === "reveal" && !busy && (
        <Overlay>
          <h2 className="font-display text-2xl font-bold">You have an original rummy!</h2>
          <p className="text-muted text-sm mt-1 mb-4">
            Choose which cards to reveal to the AI. They will be locked for the rest of the round, and you get to see the
            closed joker. <span className="text-gold">★ Coach&apos;s pick</span> keeps your other cards most flexible.
          </p>
          <div className="flex flex-col gap-3">
            {opts.map((o, i) => (
              <button
                key={i}
                onClick={() => setPick(i)}
                className={`flex items-center gap-3 rounded-xl border p-3 text-left ${
                  pick === i ? "border-gold bg-gold/10" : "border-line hover:border-muted"
                }`}
              >
                <div className="flex gap-1">
                  {o.map((id) => (
                    <PlayingCard key={id} card={byId.get(id) as Card} width={48} />
                  ))}
                </div>
                <span className="text-sm">
                  {i === view.legal.revealRecommended && <span className="text-gold">★ Coach&apos;s pick</span>}
                  {o.length === 4 && <span className="text-muted block text-xs">4 cards (3 is usually better)</span>}
                </span>
              </button>
            ))}
          </div>
          <button
            disabled={busy}
            onClick={() => onReveal(opts[pick])}
            className="mt-5 w-full py-3 rounded-lg bg-gold text-black font-semibold disabled:opacity-50"
          >
            Reveal &amp; peek at the closed joker
          </button>
        </Overlay>
      )}
    </AnimatePresence>
  );
}

function Groups({ hand, groups, wild }: { hand: Card[]; groups: number[][]; wild?: boolean }) {
  const byId = new Map(hand.map((c) => [c.id, c]));
  const used = new Set(groups.flat());
  const loose = hand.filter((c) => !used.has(c.id));
  return (
    <div className="flex flex-wrap gap-3 items-end">
      {groups.map((g, i) => (
        <div key={i} className="flex flex-wrap gap-0.5 rounded-lg bg-good/10 border border-good/30 p-1">
          {g.map((id) => byId.get(id) && <PlayingCard key={id} card={byId.get(id)!} width={38} />)}
        </div>
      ))}
      {loose.length > 0 && (
        <div className="flex flex-wrap gap-0.5 rounded-lg bg-bad/10 border border-bad/30 p-1" title="Loose cards, not in any rummy (they count against you)">
          {loose.map((c) => (
            <PlayingCard key={c.id} card={c} width={38} />
          ))}
        </div>
      )}
      {wild && null}
    </div>
  );
}

export function RoundSummary({
  view,
  onNext,
  onNewGame,
  busy,
}: {
  view: GameView;
  onNext: () => void;
  onNewGame: () => void;
  busy: boolean;
}) {
  const r = view.roundResult;
  const show = (view.phase === "round_over" || view.phase === "match_over") && r;
  const [hidden, setHidden] = useState(false);
  useEffect(() => setHidden(false), [view.round, view.phase]);
  if (!show || !r) return null;
  const headline =
    r.winner === "you" ? "You won the round!" : r.winner === "ai" ? "The AI won the round" : "No winner this round";
  const reason = r.reason === "forfeit" ? " (forfeit)" : r.reason === "exhausted" ? " (turn limit)" : "";
  const final = view.phase === "match_over";
  return (
    <AnimatePresence>
      {!hidden && (
        <Overlay>
          <p className="text-xs uppercase tracking-widest text-muted">
            Round {view.round} of {view.totalRounds}
            {r.multiplier === 2 && <span className="ml-2 text-gold">×2 round</span>}
          </p>
          <h2 className="font-display text-3xl font-bold mt-1">
            {headline}
            <span className="text-muted text-lg">{reason}</span>
          </h2>
          <div className="grid gap-4 mt-5">
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="font-semibold">Your hand</span>
                <span>
                  {r.you.points} pts{r.multiplier === 2 ? ` × 2 = ${r.you.points * 2}` : ""}
                </span>
              </div>
              <Groups hand={view.you.hand} groups={r.you.groups} />
            </div>
            {view.ai.hand && (
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="font-semibold">AI&apos;s hand</span>
                  <span>
                    {r.ai.points} pts{r.multiplier === 2 ? ` × 2 = ${r.ai.points * 2}` : ""}
                  </span>
                </div>
                <Groups hand={view.ai.hand} groups={r.ai.groups} />
              </div>
            )}
          </div>
          <div className="mt-5 rounded-lg bg-panel-2 p-3 flex justify-between text-sm">
            <span>Match total</span>
            <span>
              You <b>{view.scores.you}</b> · AI <b>{view.scores.ai}</b>
            </span>
          </div>
          {final && (
            <p className="mt-4 text-center font-display text-2xl">
              {view.matchWinner === "you" ? "🏆 You beat the AI!" : view.matchWinner === "ai" ? "The AI wins the match" : "It's a draw!"}
            </p>
          )}
          <div className="mt-5 flex gap-3">
            <button className="px-4 py-3 rounded-lg border border-line text-sm" onClick={() => setHidden(true)}>
              View table
            </button>
            {final ? (
              <button className="flex-1 py-3 rounded-lg bg-gold text-black font-semibold" onClick={onNewGame}>
                Play again
              </button>
            ) : (
              <button disabled={busy} className="flex-1 py-3 rounded-lg bg-gold text-black font-semibold disabled:opacity-50" onClick={onNext}>
                {view.mode === "round" ? "Done" : `Deal round ${view.round + 1}`}
              </button>
            )}
          </div>
        </Overlay>
      )}
    </AnimatePresence>
  );
}

export function ScoreBoard({ view }: { view: GameView }) {
  if (view.totalRounds === 1) return null;
  const cells = Array.from({ length: view.totalRounds }, (_, i) => view.roundHistory[i]);
  return (
    <div className="rounded-xl bg-panel border border-line p-3 text-xs overflow-x-auto">
      <table className="w-full text-center tabular-nums">
        <thead>
          <tr className="text-muted">
            <th className="text-left font-normal pr-2">Round</th>
            {cells.map((_, i) => (
              <th key={i} className={`font-normal px-1 ${i + 1 === view.round ? "text-gold" : ""}`}>
                {i + 1}
                {(i === 0 || i === view.totalRounds - 1) && <sup className="text-gold">×2</sup>}
              </th>
            ))}
            <th className="font-semibold pl-2">Total</th>
          </tr>
        </thead>
        <tbody>
          {(["you", "ai"] as const).map((who) => (
            <tr key={who}>
              <td className="text-left pr-2 py-0.5">{who === "you" ? "You" : "AI"}</td>
              {cells.map((c, i) => (
                <td key={i} className={c && c.winner === who ? "text-good" : ""}>
                  {c ? c[who] : "·"}
                </td>
              ))}
              <td className="font-semibold pl-2">{view.scores[who]}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
