"use client";

import { useEffect, useState } from "react";
import { wakeServer } from "@/lib/api";

/** The free Render backend sleeps when idle; this pings /api/health until it answers. */
export function useServerReady() {
  const [ready, setReady] = useState<boolean | null>(null);
  const [secs, setSecs] = useState(0);
  useEffect(() => {
    let alive = true;
    wakeServer((s) => alive && setSecs(s)).then((ok) => alive && setReady(ok));
    return () => {
      alive = false;
    };
  }, []);
  return { ready, secs };
}

export default function ServerWake({ ready, secs }: { ready: boolean | null; secs: number }) {
  if (ready) return null;
  return (
    <div className="rounded-xl border border-line bg-panel p-6 text-center max-w-md mx-auto my-10">
      {ready === null ? (
        <>
          <div className="mx-auto mb-4 h-8 w-8 rounded-full border-2 border-gold border-t-transparent animate-spin" />
          <p className="font-semibold">Waking up the AI…</p>
          <p className="text-muted text-sm mt-1">
            The free server sleeps when nobody is playing. This can take up to a minute{secs ? ` (${secs}s)` : ""}.
          </p>
        </>
      ) : (
        <>
          <p className="font-semibold">The AI server isn&apos;t responding.</p>
          <p className="text-muted text-sm mt-1">Please refresh the page in a minute.</p>
        </>
      )}
    </div>
  );
}
