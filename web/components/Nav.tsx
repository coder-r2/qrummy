"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Home" },
  { href: "/rules/", label: "Rules" },
  { href: "/play/", label: "Play the AI" },
  { href: "/learn/", label: "Learn" },
];

export default function Nav() {
  const path = usePathname() || "/";
  const active = (href: string) => (href === "/" ? path === "/" : path.startsWith(href.replace(/\/$/, "")));
  return (
    <header className="sticky top-0 z-40 border-b border-line bg-shell/85 backdrop-blur">
      <nav className="mx-auto max-w-6xl flex items-center gap-1 px-4 h-14">
        <Link href="/" className="font-display text-lg font-bold mr-4 flex items-center gap-2">
          <span className="text-gold">♠</span>QRummy
        </Link>
        <div className="flex gap-1 overflow-x-auto">
          {LINKS.slice(1).map((l) => (
            <Link
              key={l.href}
              href={l.href}
              className={`px-3 py-1.5 rounded-md text-sm whitespace-nowrap transition-colors ${
                active(l.href) ? "bg-panel-2 text-ink" : "text-muted hover:text-ink hover:bg-panel"
              }`}
            >
              {l.label}
            </Link>
          ))}
        </div>
      </nav>
    </header>
  );
}
