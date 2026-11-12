"""Thin Claude client: disk cache, prompt caching, cost accounting, robust JSON parsing.

The disk cache makes every eval run reproducible and makes re-runs free: identical
(model, system, messages, params) requests are served from SQLite.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from filings_copilot.config import LLMConfig


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0
    cache_hits: int = 0

    def add(self, other: Usage) -> None:
        for f in self.__dataclass_fields__:
            setattr(self, f, getattr(self, f) + getattr(other, f))


@dataclass
class LLMResponse:
    text: str
    usage: Usage = field(default_factory=Usage)
    cached: bool = False


class LLM:
    def __init__(self, cfg: LLMConfig, cache_path: Path | None = None, client: Any = None) -> None:
        self.cfg = cfg
        self.total = Usage()
        self._lock = threading.Lock()
        self._client = client
        self._db: sqlite3.Connection | None = None
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(cache_path, check_same_thread=False)
            self._db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic  # imported lazily so retrieval-only use needs no API key

            self._client = anthropic.Anthropic(max_retries=self.cfg.max_retries)
        return self._client

    def price(self, model: str, usage: Usage) -> float:
        p = self.cfg.prices.get(model)
        if p is None:
            return 0.0
        return (
            usage.input_tokens * p.input
            + usage.cache_write_tokens * p.input * 1.25
            + usage.cache_read_tokens * p.input * 0.10
            + usage.output_tokens * p.output
        ) / 1e6

    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        cached_prefix: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """One-turn completion.

        ``cached_prefix`` is a large, reused block of context (e.g. a whole 10-K section)
        placed before the prompt with a cache breakpoint, so repeated calls pay ~10% for it.
        """
        model = model or self.cfg.answer_model
        params = {
            "model": model,
            "max_tokens": max_tokens or self.cfg.max_tokens,
        }
        content: list[dict[str, Any]] = []
        if cached_prefix:
            content.append(
                {"type": "text", "text": cached_prefix, "cache_control": {"type": "ephemeral"}}
            )
        content.append({"type": "text", "text": prompt})
        request: dict[str, Any] = {**params, "messages": [{"role": "user", "content": content}]}
        if system:
            request["system"] = system

        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        if self._db is not None:
            # One SQLite connection is shared by worker threads; all access goes through the lock.
            with self._lock:
                row = self._db.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
                if row:
                    self.total.cache_hits += 1
            if row:
                return LLMResponse(text=json.loads(row[0])["text"], cached=True)

        resp = self.client.messages.create(**request)
        text = "".join(getattr(b, "text", "") for b in resp.content)
        u = resp.usage
        usage = Usage(
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
            calls=1,
        )
        usage.cost_usd = self.price(model, usage)
        with self._lock:
            self.total.add(usage)
            if self._db is not None:
                self._db.execute(
                    "INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, json.dumps({"text": text}))
                )
                self._db.commit()
        return LLMResponse(text=text, usage=usage)

    def complete_json(self, prompt: str, **kwargs: Any) -> Any:
        return parse_json(self.complete(prompt, **kwargs).text)


def parse_json(text: str) -> Any:
    """Extract the first JSON object or array from a model response."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    start = min((i for i in (candidate.find("{"), candidate.find("[")) if i != -1), default=-1)
    if start == -1:
        raise ValueError(f"No JSON found in model output: {text[:200]!r}")
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(candidate[start:])
    return obj
