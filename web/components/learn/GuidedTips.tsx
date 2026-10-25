"use client";

import { AnimatePresence, motion } from "motion/react";
import { useEffect, useRef, useState } from "react";
import type { GameView } from "@/lib/types";

type Tip = { id: string; title: string; body: string };

/** One-time concept callouts for the guided round (SPEC §8.6). */
function tipFor(v: GameView, seen: Set<string>): Tip | null {
  const t = (id: string, title: string, body: string) => (seen.has(id) ? null : { id, title, body });
  if (v.phase === "reveal") return null; // the reveal modal explains itself and would cover the tip
  if (v.legal.forfeit && v.phase === "your_draw")
    return t("forfeit", "The forfeit window", "On your first turn only, you may forfeit for a fixed 40 points. Usually worth it only with a hopeless hand.");
  if (v.phase === "your_draw" && !v.you.postOriginal && !v.legal.drawOpen)
    return t("closed-only", "Draw from the closed pile", "Before your original, you can only take the open card if it completes your original. Otherwise draw from the closed pile.");
  if (v.phase === "your_draw" && !v.you.postOriginal && v.legal.drawOpen)
    return t("open-ok", "The open card completes an original!", "This open card makes a pure sequence or triplet with your cards — take it, and you'll reveal at the end of this turn.");
  if (v.phase === "your_discard")
    return t("discard", "Now discard", "Every turn ends with a discard. Throw a card that isn't helping any meld, ideally a high one (A, K, Q, J score 10).");
  if (v.you.postOriginal && typeof v.you.wildRank === "number")
    return t("wild", "Wild cards unlocked", "Every card of the wild rank now works like a joker for you, and scores 0 if left unmelded.");
  if (v.ai.postOriginal)
    return t("ai-orig", "The AI revealed", "The AI has its original now and can use its own wild rank. Watch which cards it picks up.");
  return null;
}

export default function GuidedTips({ view }: { view: GameView }) {
  const seen = useRef(new Set<string>());
  const [tip, setTip] = useState<Tip | null>(null);
  useEffect(() => {
    if (tip) return;
    const next = tipFor(view, seen.current);
    if (next) {
      seen.current.add(next.id);
      setTip(next);
    }
  }, [view, tip]);
  return (
    <AnimatePresence>
      {tip && (
        <motion.div
          key={tip.id}
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          className="rounded-xl border border-gold/50 bg-gold/10 p-4 flex gap-4 items-start"
          role="status"
        >
          <span className="text-2xl">💡</span>
          <div className="flex-1">
            <p className="font-semibold">{tip.title}</p>
            <p className="text-sm text-ink/85 mt-0.5">{tip.body}</p>
          </div>
          <button className="text-xs px-3 py-1.5 rounded-md bg-gold text-black font-semibold" onClick={() => setTip(null)}>
            Got it
          </button>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
