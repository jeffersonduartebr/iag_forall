# -*- coding: utf-8 -*-
# Objective: Short-lived verdict cache for judge results.
"""Bounded LRU + TTL cache of judge verdicts keyed by (query, answer[:500]).

Extracted from ``app.judges`` (re-exported there). The instances live in
``app.judges`` (``_verdict_cache`` for binary scores, ``_rubric_cache`` for
rubric payloads).
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

VERDICT_CACHE_SIZE = 10000  # Optimized for high-capacity environment (64GB RAM) - ~10MB memory


VERDICT_CACHE_TTL_S = 300  # 5 minutos


class VerdictCache:
    """Store recently computed judge verdicts for repeated query/answer pairs."""

    def __init__(self, maxsize: int = VERDICT_CACHE_SIZE, ttl_s: int = VERDICT_CACHE_TTL_S):
        """Create a bounded verdict cache with TTL-based expiration."""
        self.maxsize = maxsize
        self.ttl_s = ttl_s
        self._lock = threading.Lock()
        self._data: "OrderedDict[str, Tuple[Any, float]]" = OrderedDict()  # key -> (payload, timestamp)
        self._hits = 0
        self._misses = 0

    def _make_key(self, query: str, answer: str) -> str:
        """Gera chave de cache baseada em hash(query + answer[:500])."""
        payload = f"{query}|{answer[:500]}".encode("utf-8", errors="ignore")
        return hashlib.sha256(payload).hexdigest()

    def get(self, query: str, answer: str) -> Optional[Any]:
        """Return a cached verdict payload when the query/answer pair is still fresh."""
        key = self._make_key(query, answer)
        now = time.time()
        with self._lock:
            if key not in self._data:
                self._misses += 1
                return None
            score, ts = self._data[key]
            if self.ttl_s > 0 and (now - ts) > self.ttl_s:
                del self._data[key]
                self._misses += 1
                return None
            self._data.move_to_end(key)
            self._hits += 1
            return score

    def set(self, query: str, answer: str, score: Any) -> None:
        """Store one verdict payload (binary score or rubric result) in the cache."""
        key = self._make_key(query, answer)
        now = time.time()
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = (score, now)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def stats(self) -> Dict[str, Any]:
        """Return cache hit, miss, and occupancy statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "size": len(self._data),
        }
