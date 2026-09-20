# Objective: The in-process LRU that fronts the settings backends.
"""Split out of ``settings_dynamic``, which was over its size limit.

A settings read goes env -> this cache -> Redis -> MariaDB. The cache is what
keeps the hot path off the network, and it has its own TTL semantics, so it
reads better on its own than buried between the backend readers it fronts.
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from typing import Any, Optional


class LRUCache:
    """Small thread-safe cache used to reduce repeated settings lookups.

    The cache stores resolved setting values for a short period because many
    runtime paths read the same keys on every request. It is deliberately simple
    and only supports the operations needed by the settings facade.
    """

    def __init__(self, maxsize: int = 512, ttl_s: int = 30):
        """Create a bounded cache with LRU eviction and optional TTL."""
        self.maxsize = maxsize
        self.ttl_s = ttl_s
        self._lock = threading.Lock()
        self._data: "OrderedDict[str, tuple[Any, float]]" = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        """Return a cached value when present and still fresh."""
        now = time.time()
        with self._lock:
            if key not in self._data:
                return None
            value, ts = self._data[key]
            if self.ttl_s > 0 and (now - ts) > self.ttl_s:
                try:
                    del self._data[key]
                except KeyError:
                    pass
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key: str, value: Any) -> None:
        """Insert or refresh one cached setting value."""
        now = time.time()
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = (value, now)
            if len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        """Remove all cached entries immediately."""
        with self._lock:
            self._data.clear()


SETTINGS_CACHE_SIZE = int(os.getenv("SETTINGS_CACHE_SIZE", "2000"))  # Optimized for high-capacity
SETTINGS_CACHE_TTL_S = int(os.getenv("SETTINGS_CACHE_TTL_S", "300"))  # 5 min - reduces Redis/DB lookups by ~80%

_lru = LRUCache(maxsize=SETTINGS_CACHE_SIZE, ttl_s=SETTINGS_CACHE_TTL_S)
