"use client";

import { useEffect, useState } from "react";
import CardRow from "@/components/CardRow";

type Stats = {
  agent: string;
  rounds: number;
  first_player_win_pct: number;
  dealt_original_pct: number;
  reveal_turn_median: number;
  reveal_turn_p75: number;
  never_reveal_pct: number;
  never_reveal_win_pct: number;
  first_reveal_win_pct: number;
  reveal_lead_win_pct: Record<string, number | null>;
  original_kind_pct: Record<string, number | null>;
  loser_post_points_mean: number;
  loser_post_points_median: number;
  open_draw_pct: number;
  round_turns_median: number;
  printed_joker_cj_pct: number;
};

const base = process.env.NEXT_PUBLIC_BASE_PATH || "";

function Stat({ value, label }: { value: string; label: string }) {
  return (
    <div className="rounded-xl bg-panel-2 border border-line p-4">
      <div className="font-display text-3xl text-gold">{value}</div>
      <div className="text-xs text-muted mt-1">{label}</div>
    </div>
  );
}

function Note({ n, title, children }: { n: number; title: string; children: React.ReactNode }) {
  return (
    <section className="rounded-2xl bg-panel border border-line p-5">
      <h3 className="font-semibold text-lg">
        <span className="text-gold mr-2">{n}.</span>
        {title}
      </h3>
      <div className="mt-2 text-ink/85 text-sm leading-relaxed space-y-3">{children}</div>
    </section>
  );
}

export default function Strategy() {
  const [s, setS] = useState<Stats | null>(null);
  useEffect(() => {
    fetch(`${base}/stats.json`)
      .then((r) => (r.ok ? r.json() : null))
      .then(setS)
      .catch(() => setS(null));
  }, []);
  const v = (x: number | null | undefined, suffix = "%") => (x === null || x === undefined ? "–" : `${x}${suffix}`);

  return (
    <div className="grid gap-4 max-w-4xl">
      {s && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <Stat value={v(s.first_reveal_win_pct)} label="win rate of whoever reveals their original first" />
          <Stat value={v(s.reveal_turn_median, "")} label="median turns a player takes before revealing (0 = dealt one)" />
          <Stat value={v(s.never_reveal_win_pct)} label="win rate when a player never reveals" />
          <Stat value={v(s.round_turns_median, "")} label="median turns in a round (both players)" />
        </div>
      )}
      <Note n={1} title="Race to your original">
        <p>
          Nothing matters more early on. Until you reveal, your secondary rummies can&apos;t use wild cards and a loss costs the full
          80 points. {s && <>In {s.rounds.toLocaleString()} self-play rounds, the player who revealed first won {v(s.first_reveal_win_pct)} of the time, and a player who never revealed won only {v(s.never_reveal_win_pct)}.</>}
        </p>
        <p>
          Keep cards that are one away from a pure sequence (like <b>5♥ 6♥</b> or <b>5♥ 7♥</b>) over loose high cards. Two
          adjacent cards in a suit can be completed from either end, a gap only from the middle.
        </p>
        <CardRow codes="5H 6H 7H" />
      </Note>
      <Note n={2} title="Three cards beat four">
        <p>
          A 3-card original frees one more card for your secondary rummies, and a 3·3·3·4 win only has one 4-card slot.
          {s?.original_kind_pct && <> The AI chose a 3-card sequence {v(s.original_kind_pct.seq3)} of the time and a triplet {v(s.original_kind_pct.triplet)}.</>} If a natural 4th card arrives later, it extends the original automatically.
        </p>
      </Note>
      <Note n={3} title="The open pile is a signal">
        <p>
          Before your original, you can only take the open card if it completes one, so don&apos;t count on it. After you reveal,
          it&apos;s fair game{s && <> — about {v(s.open_draw_pct)} of all draws came from the open pile</>}. Every open pick shows your
          opponent what you&apos;re building, and every discard shows what you&apos;re not.
        </p>
        <p>Avoid discarding a card right next to what your opponent just picked up.</p>
      </Note>
      <Note n={4} title="Wilds are worth more than points">
        <p>
          Once you&apos;ve revealed, every card of the wild rank and every printed joker scores 0 and fills any gap. Never throw one
          away unless it wins you the round. A wild only helps a rummy that already has a natural card, so pair it with a
          real card early.
        </p>
      </Note>
      <Note n={5} title="Shed high cards when you're behind">
        <p>
          A, K, Q and J score 10 each. If your opponent has revealed and you haven&apos;t, you&apos;re already looking at 80, so build
          aggressively. Once you&apos;re post-original, every point you hold counts: drop a lone King before a lone 3.
          {s && <> A revealed loser averaged {v(s.loser_post_points_mean, "")} points (median {v(s.loser_post_points_median, "")}) against the flat 80 for never revealing.</>}
        </p>
      </Note>
      <Note n={6} title="Forfeit only hopeless hands">
        <p>
          Forfeiting costs 40, half of the 80 you take without an original. It&apos;s only allowed on your first turn, so it pays
          only when your hand has almost no pairs or near-sequences. {s && <>About {v(s.dealt_original_pct)} of hands are dealt with an original already in them.</>}
        </p>
      </Note>
      <Note n={7} title="Game of 7s: the double rounds">
        <p>
          Rounds 1 and 7 count double. Play your normal game in round 1. In round 7, check the scoreboard: when you&apos;re well
          ahead, a cheap forfeit (if you&apos;re dealt a bad hand) can lock in the match; when behind, you need the win, not a
          low score.
        </p>
      </Note>
      {s && (
        <p className="text-xs text-muted">
          Figures: {s.rounds.toLocaleString()} rounds of AI-vs-AI self-play ({s.agent}), measured by the game engine. Going first won{" "}
          {v(s.first_player_win_pct)} of rounds. A printed joker turns up as the closed joker {v(s.printed_joker_cj_pct)} of the time,
          and then there&apos;s no wild rank that round.
        </p>
      )}
    </div>
  );
}
