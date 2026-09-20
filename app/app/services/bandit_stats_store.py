# -*- coding: utf-8 -*-
# Objective: Persistence of contextual bandit statistics (Redis + MariaDB).
"""Read, parse and persist per-(context, model) bandit statistics.

Redis holds the hot copy (one hash per context, one JSON field per model);
MariaDB (``bandit_context_stats``) holds the durable copy. A context absent
from Redis used to hit MariaDB on *every* request; :func:`db_fallback` now
writes DB hits back to Redis and remembers DB misses for a short TTL
(:class:`ColdContextCache`), so a cold context costs one query per window.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from prometheus_client import Counter
from sqlalchemy import text

from app.observability import registry as _registry

logger = logging.getLogger("app.bandits")

#: Divergência Redis/MariaDB: enquanto o Redis viver, ninguém repara.
BANDIT_DB_PERSIST_FAILURES = Counter(
    "bandit_db_persist_failures_total",
    "Bandit posteriors written to Redis but not to MariaDB",
    registry=_registry,
)

ModelStats = Dict[str, float]
ContextStats = Dict[str, ModelStats]

_UPSERT_SQL = text(
    """
    INSERT INTO bandit_context_stats
      (context_label, model, avg_reward, count, var, M2)
    VALUES (:ctx, :model, :avg, :count, :var, :M2)
    ON DUPLICATE KEY UPDATE
      avg_reward = VALUES(avg_reward),
      count = VALUES(count),
      var = VALUES(var),
      M2 = VALUES(M2),
      last_update = CURRENT_TIMESTAMP
    """
)  # executemany: o PyMySQL não substitui parâmetros após VALUES (...); daí VALUES(coluna)
_SELECT_SQL = text(
    """
    SELECT model, avg_reward, count, var, M2
    FROM bandit_context_stats
    WHERE context_label = :ctx
    """
)


def _finite(s: Dict[str, Any], key: str, default: float) -> float:
    try:
        v = float(s.get(key, default))
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _non_negative_int(s: Dict[str, Any], key: str, default: int) -> int:
    try:
        return max(0, int(s.get(key, default)))
    except Exception:
        return default


def sanitize_model_stats(raw: Optional[Dict[str, Any]]) -> ModelStats:
    """Centralized sanitization for per-model stats used by selection/update."""
    s = raw or {}
    return {
        "mean": max(0.0, min(1.0, _finite(s, "mean", _finite(s, "avg", 0.0)))),
        "count": _non_negative_int(s, "count", 0),
        "var": max(0.0, _finite(s, "var", 0.0)),
        "M2": max(0.0, _finite(s, "M2", 0.0)),
        "alpha": max(1e-6, _finite(s, "alpha", 1.0)),
        "beta": max(1e-6, _finite(s, "beta", 1.0)),
    }


def parse_redis_stats(raw_map: Dict[Any, Any]) -> ContextStats:
    """Parse one context hash (``model -> JSON``); bytes keys and bad fields are tolerated."""
    stats: ContextStats = {}
    for key, payload in (raw_map or {}).items():
        model = key.decode() if isinstance(key, bytes) else key
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        if isinstance(obj, dict):
            stats[model] = sanitize_model_stats(obj)
    return stats


def serialize_model_stats(s: ModelStats) -> str:
    """JSON payload stored per model in the context hash."""
    return json.dumps(
        {
            "mean": float(s.get("mean", 0.0)),
            "count": int(s.get("count", 0)),
            "var": float(s.get("var", 0.0)),
            "M2": float(s.get("M2", 0.0)),
            "alpha": float(s.get("alpha", 1.0)),
            "beta": float(s.get("beta", 1.0)),
        }
    )


def _stats_from_row(row: Any) -> ModelStats:
    mean, count = float(row[1] or 0.0), float(row[2] or 0.0)
    return sanitize_model_stats(
        {
            "mean": mean,
            "count": int(count),
            "var": float(row[3] or 0.0),
            "M2": float(row[4] or 0.0),
            "alpha": 1.0 + mean * count,
            "beta": 1.0 + max(0.0, count - mean * count),
        }
    )


def load_stats_from_db(get_engine: Callable[[], Any], ctx: str) -> ContextStats:
    """Load one context's statistics from MariaDB (empty on failure, including engine creation)."""
    try:
        with get_engine().connect() as conn:
            rows = conn.execute(_SELECT_SQL, {"ctx": ctx}).fetchall()
        return {row[0]: _stats_from_row(row) for row in rows}
    except Exception as e:
        logger.warning(f"[bandit] Falha DB ctx={ctx}: {e}")
        return {}


def upsert_stats_db(get_engine: Callable[[], Any], updates: Iterable[Tuple[str, str, ModelStats]]) -> None:
    """Persist many ``(context, model, stats)`` rows in one transaction (executemany)."""
    params = [
        {
            "ctx": ctx,
            "model": model,
            "avg": float(s.get("mean", 0.0)),
            "count": int(s.get("count", 0)),
            "var": float(s.get("var", 0.0)),
            "M2": float(s.get("M2", 0.0)),
        }
        for ctx, model, s in updates
    ]
    if not params:
        return
    try:
        with get_engine().begin() as conn:
            conn.execute(_UPSERT_SQL, params)
    except Exception as e:
        # `error`, não `warning`: a leitura prefere o Redis, por isso uma
        # escrita falhada aqui não se nota — até o Redis ser limpo, altura em
        # que todos os posteriores regridem para o último estado que chegou a
        # disco e o routing simplesmente piora, sem erro em lado nenhum.
        BANDIT_DB_PERSIST_FAILURES.inc(len(params))
        logger.error(f"[bandit] {len(params)} posteriores NÃO foram persistidos no DB: {e}")


class ColdContextCache:
    """Remember contexts with no stats in MariaDB for ``ttl_s`` (bounded LRU)."""

    def __init__(self, ttl_s: float = 30.0, maxsize: int = 4096):
        self.ttl_s = ttl_s
        self.maxsize = maxsize
        self._lock = threading.Lock()
        self._misses: "OrderedDict[str, float]" = OrderedDict()

    def is_cold(self, ctx: str) -> bool:
        with self._lock:
            expires = self._misses.get(ctx)
            if expires is None:
                return False
            if expires < time.monotonic():
                del self._misses[ctx]
                return False
            return True

    def mark_cold(self, ctx: str) -> None:
        with self._lock:
            self._misses[ctx] = time.monotonic() + self.ttl_s
            self._misses.move_to_end(ctx)
            while len(self._misses) > self.maxsize:
                self._misses.popitem(last=False)

    def forget(self, ctx: str) -> None:
        with self._lock:
            self._misses.pop(ctx, None)

    def clear(self) -> None:
        with self._lock:
            self._misses.clear()


cold_contexts = ColdContextCache()


def db_fallback(
    ctx: str,
    loader: Callable[[str], ContextStats],
    backfill: Callable[[str, ContextStats], None],
    cache: ColdContextCache = cold_contexts,
) -> ContextStats:
    """DB read for a context missing from Redis, with write-back and negative caching."""
    if cache.is_cold(ctx):
        return {}
    stats = loader(ctx)
    if stats:
        backfill(ctx, stats)
    else:
        cache.mark_cold(ctx)
    return stats
