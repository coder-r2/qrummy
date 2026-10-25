"""Gemini explanations with template fallback (SPEC §6.3, D-037).

Env: GEMINI_API_KEY, GEMINI_MODEL, LLM_RATE_PER_MIN (default 10), LLM_TIMEOUT_S (default 4).
Never raises: any problem returns None and the caller keeps the template text.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from collections import OrderedDict

import httpx

log = logging.getLogger("qrummy.llm")

SYSTEM = (
    "You are a friendly Indian Rummy coach. Explain the recommended move to the player in at most "
    "45 words, plain text, no markdown, no lists, no greetings or sign-offs; start with the move itself. "
    "Use ONLY the facts given in the JSON; never invent "
    "cards, probabilities or rules, and never mention information the player cannot see. "
    "House rules: hands of 13 cards must form melds of exactly 3/3/3/4; the first 'original' rummy must be "
    "a pure sequence or an identical triplet and unlocks the closed joker's wild rank; unmelded cards "
    "score points (A/K/Q/J=10), lowest score wins."
)


class GeminiExplainer:
    def __init__(self):
        self.key = os.environ.get("GEMINI_API_KEY", "").strip()
        self.model = os.environ.get("GEMINI_MODEL", "").strip()
        self.rate = float(os.environ.get("LLM_RATE_PER_MIN", "10"))
        self.timeout = float(os.environ.get("LLM_TIMEOUT_S", "4"))
        self.cache: OrderedDict[str, str] = OrderedDict()
        self.tokens = self.rate
        self.last = time.monotonic()
        self.inflight: set[str] = set()

    @property
    def configured(self) -> bool:
        return bool(self.key and self.model)

    def _take_token(self) -> bool:
        now = time.monotonic()
        self.tokens = min(self.rate, self.tokens + (now - self.last) * self.rate / 60.0)
        self.last = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False

    async def explain(self, facts: dict, session_id: str) -> str | None:
        if not self.configured:
            return None
        blob = json.dumps(facts, sort_keys=True, ensure_ascii=False)
        key = hashlib.sha256(blob.encode()).hexdigest()
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        if session_id in self.inflight or not self._take_token():
            return None
        self.inflight.add(session_id)
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
            body = {
                "systemInstruction": {"parts": [{"text": SYSTEM}]},
                "contents": [{"role": "user", "parts": [{"text": "Facts: " + blob}]}],
                "generationConfig": {"temperature": 0.4, "maxOutputTokens": 200},
            }
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(url, json=body, headers={"x-goog-api-key": self.key})
            if r.status_code != 200:
                log.warning("gemini HTTP %s: %s", r.status_code, r.text[:200])
                return None
            data = r.json()
            cand = data["candidates"][0]
            text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", [])).strip()
            if not text or cand.get("finishReason") not in (None, "STOP"):
                # a reply cut off by the token limit (or blocked) is worse than the template text
                log.warning("gemini unusable reply: finishReason=%s", cand.get("finishReason"))
                return None
            self.cache[key] = text
            if len(self.cache) > 2000:
                self.cache.popitem(last=False)
            return text
        except (httpx.HTTPError, KeyError, IndexError, ValueError, asyncio.TimeoutError) as e:
            log.warning("gemini failed: %s", e)
            return None
        finally:
            self.inflight.discard(session_id)
