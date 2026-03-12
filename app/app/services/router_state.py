"""State helpers for router EMA caches and batch persistence."""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional, Tuple


EMA_MAX_ENTRIES = 50000
EMA_TTL_SECONDS = 86400
EMA_BATCH_INTERVAL_S = 60
EMA_BATCH_MAX_SIZE = 500


class EMAHistoryCache:
    """LRU cache with TTL for EMA history records."""

    def __init__(self, maxsize: int = EMA_MAX_ENTRIES, ttl_s: int = EMA_TTL_SECONDS):
        self.maxsize = maxsize
        self.ttl_s = ttl_s
        self._lock = threading.Lock()
        self._data: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._access_order: list = []

    def get(self, key: Tuple[str, str]) -> Optional[Dict[str, Any]]:
        now = time.time()
        with self._lock:
            if key not in self._data:
                return None
            entry = self._data[key]
            last_update = entry.get("_last_update", 0)
            if self.ttl_s > 0 and (now - last_update) > self.ttl_s:
                del self._data[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                return None
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            return entry

    def set(self, key: Tuple[str, str], value: Dict[str, Any]) -> None:
        now = time.time()
        value["_last_update"] = now
        with self._lock:
            if key in self._data and key in self._access_order:
                self._access_order.remove(key)
            self._data[key] = value
            self._access_order.append(key)
            while len(self._data) > self.maxsize and self._access_order:
                oldest = self._access_order.pop(0)
                self._data.pop(oldest, None)

    def __contains__(self, key: Tuple[str, str]) -> bool:
        return key in self._data

    def items(self):
        with self._lock:
            return list(self._data.items())

    def cleanup_expired(self) -> int:
        now = time.time()
        removed = 0
        with self._lock:
            expired_keys = []
            for key, entry in self._data.items():
                last_update = entry.get("_last_update", 0)
                if self.ttl_s > 0 and (now - last_update) > self.ttl_s:
                    expired_keys.append(key)
            for key in expired_keys:
                del self._data[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                removed += 1
        return removed

    def size(self) -> int:
        return len(self._data)


class EMABatchQueue:
    """Queue for batching EMA updates to reduce DB writes."""

    def __init__(self, max_size: int = EMA_BATCH_MAX_SIZE, flush_interval: int = EMA_BATCH_INTERVAL_S):
        self.max_size = max_size
        self.flush_interval = flush_interval
        self._lock = threading.Lock()
        self._queue: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._last_flush = time.time()

    def add(self, modality: str, model: str, record: Dict[str, Any]) -> None:
        key = (modality, model)
        with self._lock:
            self._queue[key] = record.copy()
            self._on_queue_size_changed(len(self._queue))
            if len(self._queue) >= self.max_size:
                self._flush_locked()

    def _on_queue_size_changed(self, size: int) -> None:
        """Hook for metrics update in router_core."""

    def _flush_locked(self) -> int:
        if not self._queue:
            return 0
        items = list(self._queue.items())
        self._queue.clear()
        self._on_queue_size_changed(0)
        self._last_flush = time.time()
        return self._persist_batch(items)

    def _persist_batch(self, items: list) -> int:
        """Hook for persistence in router_core."""
        return len(items)

    def flush(self) -> int:
        with self._lock:
            return self._flush_locked()

    def should_flush(self) -> bool:
        return (time.time() - self._last_flush) >= self.flush_interval

    def size(self) -> int:
        with self._lock:
            return len(self._queue)
