# -*- coding: utf-8 -*-
# Objective: Application runtime code for health.
"""Run deep health checks for the runtime's critical dependencies.

This module centralizes the expensive health probes used by operational routes.
It inspects infrastructure dependencies such as Redis, MariaDB, ChromaDB, and
Ollama, plus internal control-plane signals such as circuit breaker state.

Because these checks can be comparatively expensive, results are cached for a
short TTL and exposed through small helper functions that distinguish between
liveness, readiness, and full diagnostic views.
"""

from __future__ import annotations

import asyncio
import logging
import time
import os
import threading
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Health check cache configuration
HEALTH_CACHE_TTL_S = 30  # Cache health results for 30 seconds


class HealthCache:
    """Store recent deep-health results behind a thread-safe TTL cache."""

    def __init__(self, ttl_s: int = HEALTH_CACHE_TTL_S):
        """Initialize cache storage and the lock protecting shared state."""
        self.ttl_s = ttl_s
        self._lock = threading.Lock()
        self._cache: Optional[Dict[str, Any]] = None
        self._cache_time: float = 0

    def get(self) -> Optional[Dict[str, Any]]:
        """Return a copy of the cached result when the TTL has not expired."""
        with self._lock:
            if self._cache is None:
                return None
            if (time.time() - self._cache_time) > self.ttl_s:
                self._cache = None
                return None
            return self._cache.copy()

    def set(self, result: Dict[str, Any]) -> None:
        """Persist one health snapshot and refresh the cache timestamp."""
        with self._lock:
            self._cache = result.copy()
            self._cache_time = time.time()

    def invalidate(self) -> None:
        """Drop any cached snapshot so the next request performs fresh probes."""
        with self._lock:
            self._cache = None
            self._cache_time = 0


_health_cache = HealthCache()


class HealthStatus(str, Enum):
    """Enumerate the aggregate status values returned by deep health checks."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """Capture the status of one dependency or internal subsystem probe."""
    name: str
    healthy: bool
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the dataclass into the JSON-friendly shape used by routes."""
        result = {"name": self.name, "healthy": self.healthy}
        if self.latency_ms is not None:
            result["latency_ms"] = round(self.latency_ms, 2)
        if self.error:
            result["error"] = self.error
        if self.details:
            result["details"] = self.details
        return result


async def check_redis_health() -> ComponentHealth:
    """Check Redis connectivity and latency."""
    try:
        from .utils.redis_client import check_redis_health as redis_health
        result = await asyncio.to_thread(redis_health)
        return ComponentHealth(
            name="redis",
            healthy=result["healthy"],
            latency_ms=result.get("latency_ms"),
            error=result.get("error"),
            details={"pool_size": result.get("pool_size")},
        )
    except Exception as e:
        return ComponentHealth(name="redis", healthy=False, error=str(e))


async def check_database_health() -> ComponentHealth:
    """Check MariaDB connectivity."""
    try:
        from .db import check_db_health

        result = await asyncio.to_thread(check_db_health)
        return ComponentHealth(
            name="mariadb",
            healthy=result.get("healthy", False),
            latency_ms=result.get("latency_ms"),
            error=result.get("error"),
            details={"pool_status": (result.get("pool_stats") or {}).get("status")},
        )
    except Exception as e:
        return ComponentHealth(name="mariadb", healthy=False, error=str(e))


async def check_vectorstore_health() -> ComponentHealth:
    """Check ChromaDB connectivity."""
    try:
        import chromadb
        chroma_path = os.getenv("CHROMA_PERSIST_PATH", "/data/chroma")

        start = time.time()
        client = chromadb.PersistentClient(path=chroma_path)
        collections = client.list_collections()
        latency_ms = (time.time() - start) * 1000

        return ComponentHealth(
            name="chromadb",
            healthy=True,
            latency_ms=latency_ms,
            details={"collections": len(collections)},
        )
    except Exception as e:
        return ComponentHealth(name="chromadb", healthy=False, error=str(e))


async def check_ollama_health() -> ComponentHealth:
    """Check Ollama server availability."""
    try:
        ollama_host = os.getenv("OLLAMA_HOST", "http://ollama:11434")
        from .providers_async import get_http_client

        start = time.time()
        client = await get_http_client()
        resp = await client.head(f"{ollama_host}/", timeout=5.0)
        resp.raise_for_status()
        latency_ms = (time.time() - start) * 1000

        return ComponentHealth(
            name="ollama",
            healthy=True,
            latency_ms=latency_ms,
            details={"probe": "HEAD /"},
        )
    except Exception as e:
        return ComponentHealth(name="ollama", healthy=False, error=str(e))


async def check_circuit_breakers_health() -> ComponentHealth:
    """Check circuit breaker status."""
    try:
        from .reliability import get_circuit_breaker_manager

        manager = get_circuit_breaker_manager()
        statuses = manager.get_all_statuses()

        open_breakers = [s for s in statuses if s.get("state") == "open"]

        return ComponentHealth(
            name="circuit_breakers",
            healthy=len(open_breakers) == 0,
            details={
                "total": len(statuses),
                "open": len(open_breakers),
                "open_models": [s["model"] for s in open_breakers],
            },
        )
    except Exception as e:
        return ComponentHealth(name="circuit_breakers", healthy=True, error=str(e))


async def get_full_health_check(force_refresh: bool = False) -> Dict[str, Any]:
    """Run all deep probes and assemble the operational health payload.

    The function executes dependency checks concurrently, derives an aggregate
    health state from individual probe results, and caches the final payload so
    repeated dashboard or readiness calls do not continuously pound external
    services. `force_refresh` bypasses the TTL cache when callers need a fresh
    diagnostic snapshot.
    """
    # Check cache first (unless force_refresh)
    if not force_refresh:
        cached = _health_cache.get()
        if cached is not None:
            cached["cached"] = True
            return cached

    start = time.time()

    # Run all checks concurrently
    checks = await asyncio.gather(
        check_redis_health(),
        check_database_health(),
        check_vectorstore_health(),
        check_ollama_health(),
        check_circuit_breakers_health(),
        return_exceptions=True,
    )

    # Process results
    components = []
    for check in checks:
        if isinstance(check, Exception):
            components.append(ComponentHealth(
                name="unknown",
                healthy=False,
                error=str(check)
            ).to_dict())
        else:
            components.append(check.to_dict())

    # Determine overall status
    healthy_count = sum(1 for c in components if c["healthy"])
    total = len(components)

    if healthy_count == total:
        status = HealthStatus.HEALTHY
    elif healthy_count >= total // 2:
        status = HealthStatus.DEGRADED
    else:
        status = HealthStatus.UNHEALTHY

    result = {
        "status": status.value,
        "timestamp": time.time(),
        "duration_ms": round((time.time() - start) * 1000, 2),
        "components": {c["name"]: c for c in components},
        "summary": {
            "healthy": healthy_count,
            "total": total,
        },
        "cached": False,
    }

    # Cache the result
    _health_cache.set(result)

    return result


def invalidate_health_cache() -> None:
    """Clear the cached deep-health payload after runtime state changes."""
    _health_cache.invalidate()


async def get_liveness_check() -> Dict[str, Any]:
    """Return a lightweight signal that the process is alive."""
    return {"status": "alive", "timestamp": time.time()}


async def get_readiness_check() -> Dict[str, Any]:
    """Return a lightweight readiness verdict derived from deep health status."""
    # Check critical dependencies only
    redis_health = await check_redis_health()
    db_health = await check_database_health()

    readiness_mode = os.getenv("READINESS_MODE", "strict").strip().lower()
    if readiness_mode == "degraded":
        ready = redis_health.healthy or db_health.healthy
    else:
        ready = redis_health.healthy and db_health.healthy

    return {
        "status": "ready" if ready else "not_ready",
        "mode": readiness_mode,
        "timestamp": time.time(),
        "redis": redis_health.healthy,
        "database": db_health.healthy,
    }
