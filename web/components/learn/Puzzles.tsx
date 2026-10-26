"use client";

import { motion } from "motion/react";
import { useEffect, useMemo, useState } from "react";
import PlayingCard from "@/components/PlayingCard";
import { cards, RANK_NAMES } from "@/lib/cards";

type Puzzle = {
  id: string;
  type: "valid_original" | "valid_meld" | "best_discard" | "score_hand";
  prompt: string;
  options: string[];
  answer: number;
  explanations: string[];
  hand?: string;
  locked?: string;
  wildRank?: number | null;
  postOriginal?: boolean;
  melds?: string[];
  unmelded?: string;
};

const TYPES: { id: Puzzle["type"] | "all"; label: string }[] = [
  { id: "all", label: "Mixed" },
  { id: "valid_original", label: "Originals" },
  { id: "valid_meld", label: "Rummies & wilds" },
  { id: "best_discard", label: "Best discard" },
  { id: "score_hand", label: "Scoring" },
];

const base = process.env.NEXT_PUBLIC_BASE_PATH || "";

function Row({ codes, wild, w = 44, hl }: { codes: string; wild?: number | null; w?: number; hl?: "green" | "red" | "gold" | null }) {
  if (!codes) return null;
  return (
    <div className="flex flex-wrap gap-1">
      {cards(codes, wild ?? undefined).map((c, i) => (
        <PlayingCard key={i} card={c} width={w} highlight={hl ?? null} />
      ))}
    </div>
  );
}

export default function Puzzles() {
  const [all, setAll] = useState<Puzzle[]>([]);
  const [kind, setKind] = useState<(typeof TYPES)[number]["id"]>("all");
  const [idx, setIdx] = useState(0);
  const [picked, setPicked] = useState<number | null>(null);
  const [score, setScore] = useState({ right: 0, total: 0 });

  useEffect(() => {
    fetch(`${base}/puzzles.json`)
      .then((r) => r.json())
      .then((d) => {
        const ps: Puzzle[] = d.puzzles;
        // deterministic interleave so "Mixed" alternates types
        setAll(ps.sort((a, b) => Number(a.id.split("-")[1]) - Number(b.id.split("-")[1])));
      })
      .catch(() => setAll([]));
  }, []);

  const list = useMemo(() => (kind === "all" ? all : all.filter((p) => p.type === kind)), [all, kind]);
  const p = list[idx % Math.max(1, list.length)];
  useEffect(() => {
    setIdx(0);
    setPicked(null);
  }, [kind]);

  if (!all.length) return <p className="text-muted">Loading puzzles…</p>;
  if (!p) return null;
  const done = picked !== null;
  const lockedCodes = (p.locked || "").split(" ").filter(Boolean);

  const choose = (i: number) => {
    if (done) return;
    setPicked(i);
    setScore((s) => ({ right: s.right + (i === p.answer ? 1 : 0), total: s.total + 1 }));
  };

  return (
    <div className="grid gap-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap gap-1">
          {TYPES.map((t) => (
            <button
              key={t.id}
              onClick={() => setKind(t.id)}
              className={`px-3 py-1.5 rounded-md text-sm ${kind === t.id ? "bg-gold text-black" : "bg-panel-2 text-muted hover:text-ink"}`}
            >
              {t.label}
            </button>
          ))}
        </div>
        <span className="text-sm text-muted">
          Score {score.right}/{score.total} · puzzle {(idx % list.length) + 1} of {list.length}
        </span>
      </div>

      <motion.div key={p.id} initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} className="rounded-2xl bg-panel border border-line p-5">
        <p className="font-semibold text-lg">{p.prompt}</p>
        {p.wildRank ? <p className="text-sm text-gold mt-1">Wild rank: {RANK_NAMES[p.wildRank]}s</p> : null}

        {p.hand && (
          <div className="felt relative rounded-xl p-3 mt-4">
            <p className="text-xs text-white/70 mb-2">
              Your hand{lockedCodes.length ? ` · locked original: ${p.locked}` : ""}
            </p>
            <Row codes={p.hand} wild={p.postOriginal === false ? null : p.wildRank} w={42} />
          </div>
        )}

        <div className={`mt-4 grid gap-3 ${p.type === "best_discard" || p.type === "score_hand" ? "grid-cols-2 sm:grid-cols-4" : "sm:grid-cols-2"}`}>
          {p.options.map((o, i) => {
            const state = !done ? null : i === p.answer ? "green" : i === picked ? "red" : null;
            return (
              <button
                key={i}
                onClick={() => choose(i)}
                className={`rounded-xl border p-3 text-left transition-colors ${
                  state === "green" ? "border-good bg-good/10" : state === "red" ? "border-bad bg-bad/10" : "border-line hover:border-muted"
                }`}
              >
                {p.type === "score_hand" ? (
                  <span className="text-2xl font-display">{o} pts</span>
                ) : p.type === "best_discard" ? (
                  <Row codes={o} wild={p.wildRank} w={52} />
                ) : (
                  <Row codes={o} wild={p.type === "valid_meld" ? p.wildRank : undefined} w={46} />
                )}
                {done && (i === p.answer || i === picked) && <p className="text-xs mt-2 text-ink/80">{p.explanations[i]}</p>}
              </button>
            );
          })}
        </div>

        {done && p.type === "score_hand" && p.melds && (
          <div className="mt-4 rounded-lg bg-panel-2 p-3 text-sm">
            <p className="text-muted mb-2">Best arrangement:</p>
            <div className="flex flex-wrap gap-3">
              {p.melds.map((m, i) => (
                <div key={i} className="rounded-md bg-good/10 border border-good/30 p-1">
                  <Row codes={m} wild={p.wildRank} w={32} />
                </div>
              ))}
              {p.unmelded && (
                <div className="rounded-md bg-bad/10 border border-bad/30 p-1">
                  <Row codes={p.unmelded} wild={p.wildRank} w={32} />
                </div>
              )}
            </div>
          </div>
        )}

        {done && (
          <div className="mt-4 flex justify-between items-center">
            <p className={picked === p.answer ? "text-good font-semibold" : "text-bad font-semibold"}>
              {picked === p.answer ? "Correct!" : "Not quite."}
            </p>
            <button
              className="px-4 py-2 rounded-lg bg-gold text-black font-semibold"
              onClick={() => {
                setIdx((i) => i + 1);
                setPicked(null);
              }}
            >
              Next puzzle →
            </button>
          </div>
        )}
      </motion.div>
    </div>
  );
}
