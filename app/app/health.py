# -*- coding: utf-8 -*-
"""
health.py — Deep Health Checks for All Dependencies
----------------------------------------------------
Provides comprehensive health checks for:
- Redis, MariaDB, ChromaDB, Ollama
- Circuit breaker status
- System resources

Features:
- Result caching (30s TTL) to avoid repeated deep checks
"""

from __future__ import annotations

import asyncio
import logging
import time
import os
import httpx
import threading
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Health check cache configuration
HEALTH_CACHE_TTL_S = 30  # Cache health results for 30 seconds


class HealthCache:
    """Thread-safe cache for health check results."""

    def __init__(self, ttl_s: int = HEALTH_CACHE_TTL_S):
        """Inicializa estado interno necessário para uso da classe."""
        self.ttl_s = ttl_s
        self._lock = threading.Lock()
        self._cache: Optional[Dict[str, Any]] = None
        self._cache_time: float = 0

    def get(self) -> Optional[Dict[str, Any]]:
        """Get cached health result if still valid."""
        with self._lock:
            if self._cache is None:
                return None
            if (time.time() - self._cache_time) > self.ttl_s:
                self._cache = None
                return None
            return self._cache.copy()

    def set(self, result: Dict[str, Any]) -> None:
        """Cache a health check result."""
        with self._lock:
            self._cache = result.copy()
            self._cache_time = time.time()

    def invalidate(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache = None
            self._cache_time = 0


_health_cache = HealthCache()


class HealthStatus(str, Enum):
    """Classe `HealthStatus`: organiza responsabilidades de health."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """Classe `ComponentHealth`: organiza responsabilidades de health."""
    name: str
    healthy: bool
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Executa to dict."""
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

        start = time.time()
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{ollama_host}/api/tags")
            resp.raise_for_status()
            data = resp.json()
        latency_ms = (time.time() - start) * 1000

        models = [m.get("name") for m in data.get("models", [])]
        return ComponentHealth(
            name="ollama",
            healthy=True,
            latency_ms=latency_ms,
            details={"models_loaded": len(models), "models": models[:5]},
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
    """
    Run all health checks and return comprehensive status.

    Uses caching (30s TTL) to avoid repeated deep checks.

    Args:
        force_refresh: If True, bypass cache and run fresh checks

    Returns:
        Dict with overall status and individual component health
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
    """Invalidate the health check cache (call when components change)."""
    _health_cache.invalidate()


async def get_liveness_check() -> Dict[str, Any]:
    """Simple liveness check (is the app running?)."""
    return {"status": "alive", "timestamp": time.time()}


async def get_readiness_check() -> Dict[str, Any]:
    """Readiness check (is the app ready to serve traffic?)."""
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
