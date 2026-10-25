"use client";

import { useEffect, useState } from "react";
import ServerWake, { useServerReady } from "@/components/ServerWake";
import Table from "@/components/game/Table";
import GuidedTips from "@/components/learn/GuidedTips";
import Puzzles from "@/components/learn/Puzzles";
import Strategy from "@/components/learn/Strategy";
import { useGame } from "@/lib/useGame";

type Tab = "guided" | "puzzles" | "strategy";

const TABS: { id: Tab; name: string; blurb: string }[] = [
  { id: "guided", name: "Guided round", blurb: "One round against the Easy AI with the coach always on and tips as you go." },
  { id: "puzzles", name: "Puzzles", blurb: "Quick quizzes on originals, melds, discards and scoring." },
  { id: "strategy", name: "Strategy notes", blurb: "What the trained AI learned, in plain words." },
];

function Guided() {
  const { ready, secs } = useServerReady();
  const game = useGame();
  const [started, setStarted] = useState(false);
  const start = () => {
    setStarted(true);
    game.start("round", "easy", "guided");
  };
  if (!ready) return <ServerWake ready={ready} secs={secs} />;
  if (!started || !game.view)
    return (
      <div className="max-w-2xl rounded-2xl bg-panel border border-line p-6">
        <p className="text-ink/90">
          You&apos;ll play a single round against the Easy AI. The coach stays on and ranks every move; callouts explain each
          rule the first time it matters. Nothing is saved, so refresh any time to start over.
        </p>
        <button onClick={start} disabled={game.busy} className="mt-5 px-6 py-3 rounded-xl bg-gold text-black font-semibold disabled:opacity-50">
          {game.busy ? "Dealing…" : "Start the guided round"}
        </button>
        {game.error && <p className="mt-3 text-bad text-sm">{game.error}</p>}
      </div>
    );
  return (
    <div className="grid gap-3">
      <GuidedTips view={game.view} />
      <Table game={game} coachEnabled setCoachEnabled={() => {}} guided onNewGame={() => setStarted(false)} />
    </div>
  );
}

export default function Learn() {
  const [tab, setTab] = useState<Tab>("guided");
  useEffect(() => {
    const h = window.location.hash.slice(1) as Tab;
    if (TABS.some((t) => t.id === h)) setTab(h);
  }, []);
  const pick = (t: Tab) => {
    setTab(t);
    history.replaceState(null, "", `#${t}`);
  };
  return (
    <div className="mx-auto max-w-7xl px-3 sm:px-4 py-6">
      <h1 className="font-display text-4xl font-bold">Learn</h1>
      <p className="text-muted mt-2">Practise the house rules at your own pace.</p>
      <div className="grid sm:grid-cols-3 gap-3 mt-5 mb-6">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => pick(t.id)}
            className={`rounded-xl border p-4 text-left transition-colors ${tab === t.id ? "border-gold bg-gold/10" : "border-line bg-panel hover:border-muted"}`}
          >
            <div className="font-semibold">{t.name}</div>
            <div className="text-xs text-muted mt-1">{t.blurb}</div>
          </button>
        ))}
      </div>
      {tab === "guided" && <Guided />}
      {tab === "puzzles" && <Puzzles />}
      {tab === "strategy" && <Strategy />}
    </div>
  );
}
