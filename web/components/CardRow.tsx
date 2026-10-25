import PlayingCard from "./PlayingCard";
import { cards } from "@/lib/cards";

export default function CardRow({
  codes,
  wild,
  ok,
  caption,
  width = 52,
}: {
  codes: string;
  wild?: number;
  ok?: boolean;
  caption?: string;
  width?: number;
}) {
  return (
    <figure className="inline-flex flex-col items-start gap-2 mr-6 mb-4 align-top">
      <div className="flex gap-1.5">
        {cards(codes, wild).map((c) => (
          <PlayingCard key={c.id} card={c} width={width} highlight={ok === undefined ? null : ok ? "green" : "red"} />
        ))}
      </div>
      {caption && (
        <figcaption className={`text-xs ${ok === false ? "text-bad" : ok ? "text-good" : "text-muted"}`}>
          {ok === undefined ? "" : ok ? "✓ " : "✗ "}
          {caption}
        </figcaption>
      )}
    </figure>
  );
}
