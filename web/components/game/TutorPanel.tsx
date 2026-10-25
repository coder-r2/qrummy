"use client";

import { motion } from "motion/react";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { ActionReq, GameView, Hint } from "@/lib/types";

const TAG_TEXT: Record<string, string> = {
  completes_win: "wins the round",
  completes_original: "completes original",
  keeps_original: "keeps original",
  reduces_deadwood: "fewer points",
  sheds_high_card: "sheds high card",
  dead_card: "dead card",
  opponent_may_want: "AI may want it",
  breaks_meld: "breaks a meld",
  discards_wild: "throws a wild!",
  unknown_card: "unknown card",
  keeps_plans_hidden: "hides your plan",
  reveals_card_to_opponent: "shows your hand",
  forfeit_fixed_40: "fixed 40 pts",
  outs: "outs",
  deadwood_after: "points left",
};

function tagLabel(t: string) {
  const [k, v] = t.split(":");
  const base = TAG_TEXT[k] ?? k.replace(/_/g, " ");
  return v !== undefined ? `${base}: ${v}` : base;
}

const pct = (p: number | null | undefined) => (p === null || p === undefined ? "—" : `${Math.round(p * 100)}%`);

export default function TutorPanel({
  view,
  enabled,
  busy,
  onPlay,
  onHighlight,
  alwaysOn,
}: {
  view: GameView;
  enabled: boolean;
  busy: boolean;
  onPlay: (a: ActionReq) => void;
  onHighlight: (ids: number[]) => void;
  alwaysOn?: boolean;
}) {
  const [hint, setHint] = useState<Hint | null>(null);
  const [text, setText] = useState<{ text: string; source: string } | null>(null);
  const [loading, setLoading] = useState(false);
  const myTurn = view.phase === "your_draw" || view.phase === "your_discard";
  const stateKey = `${view.gameId}-${view.round}-${view.phase}-${view.you.hand.length}-${view.table.discardTop?.id}-${view.table.drawCount}`;

  useEffect(() => {
    setHint(null);
    setText(null);
    onHighlight([]);
    if (!enabled || !myTurn || busy) return;
    let alive = true;
    setLoading(true);
    api
      .hint(view.gameId)
      .then((h) => {
        if (!alive) return;
        setHint(h);
        if (h.explanation) setText(h.explanation);
        const top = h.actions?.[0]?.action;
        if (top && top.type === "discard") onHighlight([top.card_id]);
        return api.explain(view.gameId).then((e) => {
          if (alive && e.text && e.source === "gemini") setText({ text: e.text, source: "gemini" });
        });
      })
      .catch(() => alive && setText({ text: "The coach is unavailable right now.", source: "error" }))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stateKey, enabled, busy]);

  if (!enabled) return null;
  const win = hint?.winProbability ?? null;
  return (
    <aside className="rounded-xl bg-panel border border-line p-4 text-sm flex flex-col gap-3" aria-live="polite">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold">Coach{alwaysOn ? " (guided)" : ""}</h3>
        {loading && <span className="text-xs text-muted animate-pulse">thinking…</span>}
      </div>
      <div>
        <div className="flex justify-between text-xs text-muted mb-1">
          <span>Your chance to win this round</span>
          <span className="text-ink font-semibold">{pct(win)}</span>
        </div>
        <div className="h-2.5 rounded-full bg-panel-2 overflow-hidden">
          <motion.div
            className="h-full bg-gradient-to-r from-bad via-gold to-good"
            initial={false}
            animate={{ width: win === null ? "0%" : `${Math.round(win * 100)}%` }}
            transition={{ type: "spring", stiffness: 120, damping: 20 }}
          />
        </div>
      </div>
      {!myTurn && <p className="text-muted">Waiting for your turn…</p>}
      {myTurn && hint?.actions && (
        <>
          <ol className="flex flex-col gap-1.5">
            {hint.actions.slice(0, 5).map((a, i) => (
              <li key={i} className={`rounded-lg p-2 border ${i === 0 ? "border-gold/60 bg-gold/5" : "border-line"}`}>
                <div className="flex items-center justify-between gap-2">
                  <span className="font-medium">
                    {i === 0 && <span className="text-gold">★ </span>}
                    {a.label}
                  </span>
                  <button
                    className="text-xs px-2 py-0.5 rounded bg-panel-2 hover:bg-gold hover:text-black disabled:opacity-40"
                    disabled={busy}
                    onClick={() => onPlay(a.action)}
                  >
                    Play
                  </button>
                </div>
                <div className="mt-1 flex items-center gap-2 text-xs text-muted">
                  <span title="How strongly the AI's Q-network prefers this move">AI pref {pct(a.preference)}</span>
                  <span>·</span>
                  <span title="Estimated chance to win the round after this move">win {pct(a.winProbability)}</span>
                </div>
                <div className="mt-1 flex flex-wrap gap-1">
                  {a.tags
                    .filter((t) => !t.startsWith("deadwood_after"))
                    .slice(0, 4)
                    .map((t) => (
                      <span key={t} className="px-1.5 py-0.5 rounded bg-panel-2 text-[11px] text-muted">
                        {tagLabel(t)}
                      </span>
                    ))}
                </div>
              </li>
            ))}
          </ol>
          {text && (
            <p className="leading-relaxed border-t border-line pt-3">
              {text.text}
              {text.source === "gemini" && <span className="ml-1 text-[10px] text-gold align-middle">AI-written</span>}
            </p>
          )}
        </>
      )}
    </aside>
  );
}
