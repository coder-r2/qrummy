import CardRow from "@/components/CardRow";

export const metadata = { title: "Rules — QRummy" };

function Section({ id, title, children }: { id: string; title: string; children: React.ReactNode }) {
  return (
    <section id={id} className="scroll-mt-20 py-6 border-b border-line last:border-0">
      <h2 className="font-display text-2xl font-bold mb-3">{title}</h2>
      <div className="text-[15px] leading-relaxed text-ink/90 space-y-3">{children}</div>
    </section>
  );
}

const TOC = [
  ["setup", "Setup"],
  ["rummies", "Rummies"],
  ["jokers", "Jokers & wild cards"],
  ["original", "The original rummy"],
  ["turn", "Your turn"],
  ["winning", "Winning a round"],
  ["scoring", "Scoring"],
  ["forfeit", "Forfeiting"],
  ["match", "Game of 7s"],
];

export default function Rules() {
  return (
    <div className="mx-auto max-w-6xl px-4 py-10 grid md:grid-cols-[200px_1fr] gap-10">
      <aside className="hidden md:block">
        <nav className="sticky top-20 text-sm space-y-1">
          {TOC.map(([id, t]) => (
            <a key={id} href={`#${id}`} className="block text-muted hover:text-ink py-0.5">
              {t}
            </a>
          ))}
        </nav>
      </aside>
      <article>
        <h1 className="font-display text-4xl font-bold mb-2">How to play</h1>
        <p className="text-muted mb-4">
          QRummy plays a two-player house variant of Indian Rummy. Lowest score wins: you want to arrange your whole hand into rummies
          before the AI does.
        </p>

        <Section id="setup" title="Setup">
          <p>
            Three standard decks plus <b>6 printed jokers</b> — 162 cards. Each player gets <b>13 cards</b>. One card is turned
            face-up to start the discard pile.
          </p>
          <p>
            The dealer then sets aside one random card face-down: the <b>closed joker (CJ)</b>. Its rank, in every suit,
            is the round's <b>wild rank</b> — but you only find out what it is after you complete your first rummy. If the
            CJ is itself a printed joker, there is no wild rank that round. The dealer alternates each round and the
            other player goes first.
          </p>
        </Section>

        <Section id="rummies" title="Rummies">
          <p>A <b>rummy</b> is a valid group of exactly <b>3 or 4 cards</b>. Your first one is the <b>original</b>; every
          rummy after it is a <b>secondary rummy</b>.</p>
          <p>
            <b>Sequence</b> — consecutive cards of one suit. The Ace can be low or high, but sequences can't wrap around.
          </p>
          <div>
            <CardRow codes="4H 5H 6H" ok caption="sequence" />
            <CardRow codes="QS KS AS" ok caption="Ace high" />
            <CardRow codes="AD 2D 3D 4D" ok caption="Ace low, 4 cards" />
            <CardRow codes="KC AC 2C" ok={false} caption="no wrap-around" />
          </div>
          <p>
            <b>Set</b> — same rank, all different suits. <b>Triplet</b> — three <i>identical</i> cards (possible because there are
            three decks).
          </p>
          <div>
            <CardRow codes="9S 9H 9C" ok caption="set" />
            <CardRow codes="9S 9H 9C 9D" ok caption="set of 4" />
            <CardRow codes="9S 9S 9H" ok={false} caption="repeated suit" />
            <CardRow codes="7D 7D 7D" ok caption="triplet" />
          </div>
        </Section>

        <Section id="jokers" title="Jokers & wild cards">
          <p>
            <b>Printed jokers</b> can stand in for any card in a secondary sequence or set, for everyone, at any time — but never in
            your original rummy, and never in a triplet. Once you know the wild rank, every card of that rank (any suit) works
            the same way for you, and can also still be used as itself.
          </p>
          <p>A secondary rummy may contain any number of wild cards, but it needs at least one ordinary card.</p>
          <div>
            <CardRow codes="5C JK 7C" ok caption="joker as 6♣" />
            <CardRow codes="KH KS JK" ok caption="joker in a set" />
            <CardRow codes="2D 3S 4D" wild={3} ok caption="3♠ wild (wild rank 3)" />
            <CardRow codes="JK JK JK" ok={false} caption="no ordinary card" />
          </div>
        </Section>

        <Section id="original" title="The original (first rummy)">
          <p>
            Your first rummy — the <b>original</b> — must be a <b>pure sequence</b> (no jokers or wilds) or a <b>triplet</b>.
            Until you have one you are <i>pre-original</i>: you may only take the open discard if it completes your
            original.
          </p>
          <div>
            <CardRow codes="8S 9S 10S" ok caption="pure sequence" />
            <CardRow codes="QD QD QD" ok caption="triplet" />
            <CardRow codes="8S JK 10S" ok={false} caption="jokers not allowed" />
            <CardRow codes="8S 8H 8C" ok={false} caption="a set doesn't count" />
          </div>
          <p>
            As soon as you have one (after your discard), you reveal it to the AI and peek at the closed joker: from then on
            the wild rank is yours to use. The revealed cards are <b>locked</b> — you can never discard them — but a 3-card
            sequence may be extended by one more natural card when you declare. If your dealt hand already contains an
            original, you reveal it before the first turn.
          </p>
          <p>When both players have revealed, the closed joker is turned face-up.</p>
        </Section>

        <Section id="turn" title="Your turn">
          <ol className="list-decimal pl-5 space-y-1">
            <li>
              <b>Draw</b> one card: the top of the closed pile, or the top of the discard pile (pre-original: only if it completes your
              original).
            </li>
            <li>
              <b>Discard</b> one card. You can't throw back the card you just took from the discard pile, or any copy of it.
            </li>
          </ol>
          <p>If the draw pile runs out, the discards (except the top card) are shuffled into a new draw pile.</p>
        </Section>

        <Section id="winning" title="Winning a round">
          <p>
            After your discard, if your 13 cards split into rummies of exactly <b>3 + 3 + 3 + 4</b> — your original plus three secondary rummies —
            you win the round immediately. You can even complete your original and win in the same turn, but then only
            printed jokers count as wild (you haven't seen the closed joker yet).
          </p>
          <div>
            <CardRow codes="5H 6H 7H" caption="original" width={44} />
            <CardRow codes="2S 2D 2C" caption="set" width={44} />
            <CardRow codes="9C 10C JC" caption="sequence" width={44} />
            <CardRow codes="KS KD KC KH" caption="set of 4" width={44} />
          </div>
        </Section>

        <Section id="scoring" title="Scoring">
          <ul className="list-disc pl-5 space-y-1">
            <li>The winner scores 0.</li>
            <li>A loser with no original scores <b>80</b>.</li>
            <li>
              A loser with an original scores the points of the loose cards left out of their rummies (best arrangement),
              up to a maximum of <b>80</b>. Up to four rummies count, the original included, and at most one of them may
              have 4 cards — so <b>3 + 3 + 3 + 3</b> with a single loose card is allowed, and only that card scores.
            </li>
            <li>A, K, Q, J = 10 · 2–10 = face value · jokers = 0 · wild-rank cards = 0 once you know the wild rank.</li>
          </ul>
        </Section>

        <Section id="forfeit" title="Forfeiting">
          <p>
            On your <b>first turn only</b> you may forfeit for a fixed <b>40</b> points — either before drawing, or right after a
            closed draw by throwing that card back. The other player wins the round with 0.
          </p>
        </Section>

        <Section id="match" title="Game of 7s">
          <p>
            A match is <b>7 rounds</b>. Rounds 1 and 7 count <b>double</b>. After seven rounds the lowest total wins; equal totals
            are a draw.
          </p>
        </Section>
      </article>
    </div>
  );
}
