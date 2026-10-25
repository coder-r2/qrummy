import type { ActionReq, CoachMode, Difficulty, Explain, Hint, Mode, Payload } from "./types";

export const API_BASE = (process.env.NEXT_PUBLIC_API_BASE || "http://localhost:7860").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    });
  } catch {
    throw new ApiError(0, "Can't reach the AI server. It may be waking up — try again in a few seconds.");
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {}
    throw new ApiError(res.status, String(detail));
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => call<{ status: string; model_loaded: boolean; value_loaded: boolean; llm_configured: boolean }>("/api/health"),
  newGame: (mode: Mode, difficulty: Difficulty, coach: CoachMode) =>
    call<Payload>("/api/games", { method: "POST", body: JSON.stringify({ mode, difficulty, coach }) }),
  act: (gameId: string, action: ActionReq) =>
    call<Payload>(`/api/games/${gameId}/actions`, { method: "POST", body: JSON.stringify(action) }),
  hint: (gameId: string) => call<Hint>(`/api/games/${gameId}/hint`, { method: "POST" }),
  explain: (gameId: string) => call<Explain>(`/api/games/${gameId}/hint/explain`, { method: "POST" }),
  end: (gameId: string) => call<{ ok: boolean }>(`/api/games/${gameId}`, { method: "DELETE" }),
};

/** Ping /api/health until the (possibly sleeping) Space answers. */
export async function wakeServer(onTick?: (secs: number) => void, timeoutS = 120): Promise<boolean> {
  const t0 = Date.now();
  while ((Date.now() - t0) / 1000 < timeoutS) {
    try {
      await api.health();
      return true;
    } catch {
      onTick?.(Math.round((Date.now() - t0) / 1000));
      await new Promise((r) => setTimeout(r, 3000));
    }
  }
  return false;
}
