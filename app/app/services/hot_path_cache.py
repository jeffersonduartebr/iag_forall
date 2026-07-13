# Objective: Lightweight TTL caches for hot-path governance reads.
"""Thread-safe TTL caches used to avoid repeated sync DB reads on request paths."""

from __future__ import annotations

import threading
import time
from typing import Any, Dict

_CACHE_MISS = object()


class TTLCache:
    """Small in-process TTL cache for hot-path read-through helpers."""

    def __init__(self, ttl_s: float, maxsize: int = 2048) -> None:
        self._ttl_s = max(0.0, float(ttl_s))
        self._maxsize = max(1, int(maxsize))
        self._data: Dict[str, tuple[Any, float]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Any:
        """Return a cached value or ``_CACHE_MISS`` when absent or expired."""
        now = time.time()
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return _CACHE_MISS
            value, ts = entry
            if self._ttl_s > 0 and (now - ts) > self._ttl_s:
                self._data.pop(key, None)
                return _CACHE_MISS
            return value

    def set(self, key: str, value: Any) -> None:
        """Store one cache entry and evict the oldest key when over capacity."""
        now = time.time()
        with self._lock:
            self._data[key] = (value, now)
            if len(self._data) > self._maxsize:
                oldest = min(self._data.items(), key=lambda item: item[1][1])[0]
                self._data.pop(oldest, None)

    def invalidate(self, key: str) -> None:
        """Drop one cache entry."""
        with self._lock:
            self._data.pop(key, None)

    def clear(self) -> None:
        """Drop all cache entries."""
        with self._lock:
            self._data.clear()


def cache_hit(value: Any) -> bool:
    """Return whether a cache lookup returned a stored value."""
    return value is not _CACHE_MISS
