"""QRummy FastAPI backend (SPEC §7).

    uvicorn server.server:app --port 7860          (one worker: sessions live in memory)
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from indian_rummy.game import IllegalAction

from .ai import Models
from .coach import Coach
from .llm import GeminiExplainer
from .sessions import SessionStore
from .views import game_view

VERSION = "1.0.0"
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("qrummy")

models = Models()
store = SessionStore(models)
coach = Coach(models)
llm = GeminiExplainer()


@contextlib.asynccontextmanager
async def lifespan(_app):
    async def janitor():
        while True:
            await asyncio.sleep(300)
            n = store.expire()
            if n:
                log.info("expired %d sessions", n)
    task = asyncio.create_task(janitor())
    yield
    task.cancel()


app = FastAPI(title="QRummy API", version=VERSION, lifespan=lifespan)
origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins + ["http://localhost:3000", "http://127.0.0.1:3000"],
                   allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"])


class NewGame(BaseModel):
    mode: Literal["match", "round"] = "match"
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    coach: Literal["off", "on", "guided"] = "off"
    seed: int | None = None


class Action(BaseModel):
    type: Literal["draw", "discard", "reveal", "forfeit", "next_round"]
    source: Literal["closed", "open"] | None = None
    card_id: int | None = None
    card_ids: list[int] | None = None


def _session(game_id: str):
    s = store.get(game_id)
    if s is None:
        raise HTTPException(404, "game not found (it may have expired); start a new game")
    return s


def _payload(s) -> dict:
    return {"view": game_view(s), "events": s.take_events()}


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": VERSION, "model_loaded": models.coach_dqn is not None,
            "value_loaded": models.value is not None, "llm_configured": llm.configured,
            "sessions": len(store.sessions)}


@app.post("/api/games")
async def new_game(req: NewGame):
    s = await run_in_threadpool(store.create, req.mode, req.difficulty, req.coach, req.seed)
    async with s.lock:
        return _payload(s)


@app.get("/api/games/{game_id}")
async def get_game(game_id: str):
    s = _session(game_id)
    async with s.lock:
        return {"view": game_view(s), "events": []}


@app.post("/api/games/{game_id}/actions")
async def act(game_id: str, action: Action):
    s = _session(game_id)
    async with s.lock:
        try:
            await run_in_threadpool(s.apply, action.model_dump())
        except (IllegalAction, ValueError, KeyError, TypeError) as e:
            raise HTTPException(409, f"illegal action: {e}")
        s.last_facts = None
        return _payload(s)


@app.post("/api/games/{game_id}/hint")
async def hint(game_id: str):
    s = _session(game_id)
    async with s.lock:
        h = await run_in_threadpool(coach.hint, s)
        s.last_facts = h.get("facts")
        return h


@app.post("/api/games/{game_id}/hint/explain")
async def explain(game_id: str):
    s = _session(game_id)
    async with s.lock:
        facts = getattr(s, "last_facts", None)
        if facts is None:
            h = await run_in_threadpool(coach.hint, s)
            facts = s.last_facts = h.get("facts")
            template = h.get("explanation", {}).get("text")
        else:
            template = None
    if facts is None:
        return {"text": None, "source": "none"}
    text = await llm.explain(facts, game_id)
    if text:
        return {"text": text, "source": "gemini"}
    return {"text": template, "source": "template"}


@app.delete("/api/games/{game_id}")
async def end_game(game_id: str):
    store.delete(game_id)
    return {"ok": True}
