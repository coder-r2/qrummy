"use client";

import Link from "next/link";
import { motion } from "motion/react";
import PlayingCard from "@/components/PlayingCard";
import type { Card } from "@/lib/types";

const FAN: Card[] = [
  { id: 1, code: "QH", rank: 12, suit: "H", isJoker: false, isWild: false },
  { id: 2, code: "KH", rank: 13, suit: "H", isJoker: false, isWild: false },
  { id: 3, code: "AH", rank: 1, suit: "H", isJoker: false, isWild: false },
  { id: 4, code: "JK", rank: null, suit: null, isJoker: true, isWild: true },
  { id: 5, code: "7S", rank: 7, suit: "S", isJoker: false, isWild: true },
];

const FEATURES = [
  { title: "A real RL opponent", body: "A Deep Q-Network trained with RLCard from expert demonstrations and self-play, running in the cloud." },
  { title: "A coach that explains", body: "Win probability, a ranking of every legal move and a plain-English reason, updated on every turn." },
  { title: "House rules, enforced", body: "3 decks + 6 jokers, 13-card hands, strict 3/3/3/4, the closed joker and the first-rummy rule." },
];

export default function Home() {
  return (
    <div className="mx-auto max-w-6xl px-4">
      <section className="grid md:grid-cols-2 gap-10 items-center py-14 md:py-20">
        <div>
          <p className="text-gold text-sm font-semibold tracking-widest uppercase mb-3">Indian Rummy · Game of 7s</p>
          <h1 className="font-display text-4xl md:text-6xl font-bold leading-tight">
            Beat the AI.
            <br />
            <span className="text-gold-soft">Learn why.</span>
          </h1>
          <p className="mt-5 text-muted text-lg max-w-prose">
            Play seven rounds against an AI trained with reinforcement learning. Turn on the coach to see your chance
            of winning and why each move is good or bad.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Link href="/play/" className="px-5 py-3 rounded-lg bg-gold text-black font-semibold hover:brightness-110">
              Play the AI
            </Link>
            <Link href="/learn/" className="px-5 py-3 rounded-lg bg-panel-2 border border-line hover:border-gold">
              Learn to play
            </Link>
            <Link href="/rules/" className="px-5 py-3 rounded-lg text-muted hover:text-ink">
              Read the rules →
            </Link>
          </div>
        </div>
        <div className="relative h-72 md:h-96 felt rounded-[2rem] flex items-center justify-center overflow-hidden">
          {FAN.map((c, i) => {
            const mid = (FAN.length - 1) / 2;
            return (
              <motion.div
                key={c.id}
                className="absolute"
                initial={{ y: 200, rotate: 0, opacity: 0 }}
                animate={{ y: Math.abs(i - mid) * 10, x: (i - mid) * 52, rotate: (i - mid) * 9, opacity: 1 }}
                transition={{ delay: 0.15 + i * 0.09, type: "spring", stiffness: 140, damping: 16 }}
                whileHover={{ y: -18 }}
              >
                <PlayingCard card={c} width={96} />
              </motion.div>
            );
          })}
        </div>
      </section>

      <section className="grid md:grid-cols-3 gap-4 pb-16">
        {FEATURES.map((f) => (
          <div key={f.title} className="rounded-xl bg-panel border border-line p-5">
            <h3 className="font-semibold mb-2">{f.title}</h3>
            <p className="text-muted text-sm leading-relaxed">{f.body}</p>
          </div>
        ))}
      </section>

      <section className="pb-20">
        <h2 className="font-display text-2xl font-bold mb-4">How the AI works</h2>
        <ol className="grid md:grid-cols-4 gap-4 text-sm">
          {[
            ["Rules engine", "A Python engine validates every meld, joker and phase rule; a fast solver finds the best arrangement of any hand."],
            ["Learning", "The DQN learns by imitating a strong hand-written strategy, then plays itself. In our tests the imitation stage gave the strongest network, so that is the one you meet."],
            ["Decisions", "Each turn it scores every legal draw and discard (its Q-values) from what it can legally see — never your cards."],
            ["Coaching", "A separate value network estimates your win chance; Gemini turns the facts into a short explanation."],
          ].map(([t, b], i) => (
            <li key={t} className="rounded-xl bg-panel border border-line p-5">
              <span className="text-gold font-display text-2xl">{i + 1}</span>
              <h3 className="font-semibold mt-1 mb-1">{t}</h3>
              <p className="text-muted leading-relaxed">{b}</p>
            </li>
          ))}
        </ol>
      </section>
    </div>
  );
}
