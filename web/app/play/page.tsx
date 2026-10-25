"use client";

import { useState } from "react";
import ServerWake, { useServerReady } from "@/components/ServerWake";
import Table from "@/components/game/Table";
import type { Difficulty } from "@/lib/types";
import { useGame } from "@/lib/useGame";

const LEVELS: { id: Difficulty; name: string; blurb: string }[] = [
  { id: "easy", name: "Easy", blurb: "A casual player who makes plenty of mistakes." },
  { id: "medium", name: "Medium", blurb: "Solid fundamentals, occasionally careless." },
  { id: "hard", name: "Hard", blurb: "The strongest AI we trained. Good luck." },
];

export default function Play() {
  const { ready, secs } = useServerReady();
  const game = useGame();
  const [difficulty, setDifficulty] = useState<Difficulty>("medium");
  const [coach, setCoach] = useState(false);
  const [started, setStarted] = useState(false);

  const start = () => {
    setStarted(true);
    game.start("match", difficulty, coach ? "on" : "off");
  };

  return (
    <div className="mx-auto max-w-7xl px-3 sm:px-4 py-6">
      {!ready && <ServerWake ready={ready} secs={secs} />}
      {ready && (!started || !game.view) && (
        <div className="max-w-2xl mx-auto py-6">
          <h1 className="font-display text-4xl font-bold">Beat the AI</h1>
          <p className="text-muted mt-2">
            A full Game of 7s: seven rounds, the first and last count double, lowest total wins. New to the house rules?{" "}
            <a href="../rules/" className="text-gold underline">Read them first</a>.
          </p>
          <div className="grid sm:grid-cols-3 gap-3 mt-6">
            {LEVELS.map((l) => (
              <button
                key={l.id}
                onClick={() => setDifficulty(l.id)}
                className={`rounded-xl border p-4 text-left transition-colors ${
                  difficulty === l.id ? "border-gold bg-gold/10" : "border-line bg-panel hover:border-muted"
                }`}
              >
                <div className="font-semibold">{l.name}</div>
                <div className="text-xs text-muted mt-1">{l.blurb}</div>
              </button>
            ))}
          </div>
          <label className="mt-4 flex items-center gap-3 rounded-xl border border-line bg-panel p-4 cursor-pointer">
            <input type="checkbox" checked={coach} onChange={(e) => setCoach(e.target.checked)} className="accent-[#d9b24a] w-4 h-4" />
            <span>
              <span className="font-semibold">Coach</span>
              <span className="block text-xs text-muted">Shows your win chance, ranks your moves and explains why. You can toggle it any time.</span>
            </span>
          </label>
          <button
            onClick={start}
            disabled={game.busy}
            className="mt-6 w-full py-3.5 rounded-xl bg-gold text-black font-semibold text-lg disabled:opacity-50"
          >
            {game.busy ? "Dealing…" : "Deal the first round"}
          </button>
          {game.error && <p className="mt-3 text-bad text-sm">{game.error}</p>}
        </div>
      )}
      {ready && started && game.view && (
        <Table
          game={game}
          coachEnabled={coach}
          setCoachEnabled={setCoach}
          onNewGame={() => {
            setStarted(false);
          }}
        />
      )}
    </div>
  );
}
