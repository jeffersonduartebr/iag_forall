# -*- coding: utf-8 -*-
# Objective: Shared exponential moving averages of latency/cost/quality per (modality, model).
"""EMA of observed latency, inference cost and quality, shared through Redis.

The feedback loop runs in several processes (API workers, Celery). Keeping
the EMA only in process memory made each worker compute its own diverging
averages, and the API — which routes — never saw them. Each observation now
updates ``ema:<modality>`` (hash field = model) atomically (WATCH/MULTI with
retries), and routing reads a snapshot cached for ``SNAPSHOT_TTL_S``.

The entry layout matches ``ema_history`` (``ema_latency``, ``ema_cost``,
``ema_quality``, ``ema_alignment``, ``updates``).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

EMA_ALPHA = 0.2
EMA_KEY = "ema:{modality}"


def _ema_key(modality: str) -> str:
    """Redis key for one modality's EMAs, qualified by the quality semantics.

    Identity under the default semantics, so existing keys are untouched.
    """
    from .quality_semantics import namespaced

    return EMA_KEY.format(modality=namespaced(modality))
SNAPSHOT_TTL_S = 5.0
MIN_UPDATES_FOR_ROUTING = 3

_snapshot_lock = threading.Lock()
_snapshots: Dict[str, Tuple[float, Dict[str, Dict[str, Any]]]] = {}


def next_ema(prev: Optional[Dict[str, Any]], latency_s: float, quality: float, cost: float) -> Dict[str, Any]:
    """One exponential-moving-average step (alpha = EMA_ALPHA); first sample seeds the averages."""
    if not prev:
        return {"ema_latency": latency_s, "ema_quality": quality, "ema_cost": cost, "ema_alignment": 1.0, "updates": 1}
    a = EMA_ALPHA
    return {
        "ema_latency": a * latency_s + (1 - a) * prev["ema_latency"],
        "ema_quality": a * quality + (1 - a) * prev["ema_quality"],
        "ema_cost": a * cost + (1 - a) * prev["ema_cost"],
        "ema_alignment": prev.get("ema_alignment", 1.0),
        "updates": prev.get("updates", 0) + 1,
    }


def _get_rds():
    from app.utils.redis_client import ensure_redis_connected, get_redis_async_safe

    return get_redis_async_safe() or ensure_redis_connected(max_wait_s=0.0, min_retry_interval_s=2.0)


def _decode(raw: Any) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    try:
        entry = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
    except Exception:
        return None
    return entry if isinstance(entry, dict) and "ema_latency" in entry else None


def update_shared_ema(
    modality: str, model: str, latency_s: float, quality: float, cost: float, rds: Any = None, retries: int = 5
) -> Optional[Dict[str, Any]]:
    """Atomically fold one observation into the shared EMA; returns the new entry (None on failure)."""
    rds = rds if rds is not None else _get_rds()
    if not rds:
        return None
    key = _ema_key(modality)
    for _ in range(retries):
        try:
            with rds.pipeline() as pipe:
                pipe.watch(key)
                entry = next_ema(_decode(pipe.hget(key, model)), latency_s, quality, cost)
                pipe.multi()
                pipe.hset(key, model, json.dumps(entry))
                pipe.execute()
                return entry
        except Exception as exc:  # WatchError: outro processo gravou no meio; tenta de novo
            if type(exc).__name__ != "WatchError":
                logger.debug("[ema] update failed for %s/%s: %s", modality, model, exc)
                return None
    return None


def load_ema_snapshot(modality: str, rds: Any = None) -> Dict[str, Dict[str, Any]]:
    """``model -> EMA entry`` for one modality, cached in-process for SNAPSHOT_TTL_S."""
    now = time.monotonic()
    with _snapshot_lock:
        cached = _snapshots.get(modality)
        if cached is not None and now - cached[0] < SNAPSHOT_TTL_S:
            return cached[1]
    snapshot: Dict[str, Dict[str, Any]] = {}
    rds = rds if rds is not None else _get_rds()
    try:
        raw_map = rds.hgetall(_ema_key(modality)) if rds else {}
        for field, raw in (raw_map if isinstance(raw_map, dict) else {}).items():
            entry = _decode(raw)
            if entry is not None:
                snapshot[field.decode() if isinstance(field, bytes) else str(field)] = entry
    except Exception as exc:
        logger.debug("[ema] snapshot read failed for %s: %s", modality, exc)
    with _snapshot_lock:
        _snapshots[modality] = (now, snapshot)
    return snapshot


def reset_ema_snapshots() -> None:
    with _snapshot_lock:
        _snapshots.clear()


def routing_latency_cost(entry: Optional[Dict[str, Any]], *, is_local: bool, is_sota: bool) -> Tuple[float, float]:
    """Estimated latency (s) and cost (USD/request) for S(m).

    Uses the shared EMA once it has MIN_UPDATES_FOR_ROUTING observations;
    otherwise the historical heuristics (local 0.5 s, others 2.0 s; frontier
    0.01 USD, others 1e-6 USD).
    """
    if entry and int(entry.get("updates", 0)) >= MIN_UPDATES_FOR_ROUTING:
        return float(entry["ema_latency"]), float(entry["ema_cost"])
    return (0.5 if is_local else 2.0), (0.01 if is_sota else 0.000001)
