# Objective: Admin log streaming via Loki.
"""Stream application logs for the admin console."""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, AsyncIterator, Dict, Optional

import httpx

LOKI_URL = os.getenv("LOKI_URL", "http://loki:3100").rstrip("/")
DEFAULT_QUERY = '{container=~".*api.*|.*celery.*|.*nsga.*"}'


def _parse_log_line(raw: str) -> Dict[str, Any]:
    raw = (raw or "").strip()
    if not raw:
        return {"message": "", "level": "info", "timestamp": time.time()}
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return {
                "timestamp": obj.get("timestamp") or obj.get("ts") or time.time(),
                "level": obj.get("level") or obj.get("levelname") or "info",
                "event": obj.get("event") or obj.get("message") or obj.get("msg") or "",
                "correlation_id": obj.get("correlation_id"),
                "message": raw,
            }
    except json.JSONDecodeError:
        pass
    return {"timestamp": time.time(), "level": "info", "event": raw, "message": raw}


async def _fetch_loki_logs(
    *,
    query: str,
    start_ns: int,
    end_ns: int,
    limit: int = 200,
) -> list[Dict[str, Any]]:
    params = {
        "query": query,
        "start": str(start_ns),
        "end": str(end_ns),
        "limit": str(limit),
        "direction": "forward",
    }
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{LOKI_URL}/loki/api/v1/query_range", params=params)
            if resp.status_code != 200:
                return []
            payload = resp.json()
            streams = payload.get("data", {}).get("result") or []
            out: list[Dict[str, Any]] = []
            for stream in streams:
                labels = stream.get("stream") or {}
                for ts, line in stream.get("values") or []:
                    parsed = _parse_log_line(line)
                    parsed["container"] = labels.get("container") or labels.get("container_name")
                    parsed["ts_ns"] = int(ts)
                    out.append(parsed)
            out.sort(key=lambda x: x.get("ts_ns", 0))
            return out
    except Exception:
        return []


async def stream_logs(
    *,
    query: str = DEFAULT_QUERY,
    poll_s: float = 2.0,
    last_ts_ns: Optional[int] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """Yield new log entries by polling Loki."""
    cursor = last_ts_ns or int((time.time() - 30) * 1_000_000_000)
    while True:
        now_ns = int(time.time() * 1_000_000_000)
        entries = await _fetch_loki_logs(query=query, start_ns=cursor, end_ns=now_ns, limit=100)
        for entry in entries:
            ts = int(entry.get("ts_ns") or cursor)
            if ts > cursor:
                cursor = ts
            yield entry
        await asyncio.sleep(max(0.5, poll_s))
