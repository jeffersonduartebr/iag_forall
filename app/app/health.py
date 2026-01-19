# -*- coding: utf-8 -*-
"""
health.py — Deep Health Checks for All Dependencies
----------------------------------------------------
Provides comprehensive health checks for:
- Redis, MariaDB, ChromaDB, Ollama
- Circuit breaker status
- System resources
"""

from __future__ import annotations

import asyncio
import logging
import time
import os
import httpx
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    name: str
    healthy: bool
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
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
        result = redis_health()
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
        from sqlalchemy import create_engine, text
        from .settings_dynamic import settings

        db_url = f"mysql+pymysql://{settings.DB_USER}:{settings.DB_PASS}@{settings.DB_HOST}:3306/{settings.DB_NAME}"
        engine = create_engine(db_url, pool_pre_ping=True)

        start = time.time()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        latency_ms = (time.time() - start) * 1000

        return ComponentHealth(name="mariadb", healthy=True, latency_ms=latency_ms)
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


async def get_full_health_check() -> Dict[str, Any]:
    """
    Run all health checks and return comprehensive status.

    Returns:
        Dict with overall status and individual component health
    """
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

    return {
        "status": status.value,
        "timestamp": time.time(),
        "duration_ms": round((time.time() - start) * 1000, 2),
        "components": {c["name"]: c for c in components},
        "summary": {
            "healthy": healthy_count,
            "total": total,
        },
    }


async def get_liveness_check() -> Dict[str, Any]:
    """Simple liveness check (is the app running?)."""
    return {"status": "alive", "timestamp": time.time()}


async def get_readiness_check() -> Dict[str, Any]:
    """Readiness check (is the app ready to serve traffic?)."""
    # Check critical dependencies only
    redis_health = await check_redis_health()
    db_health = await check_database_health()

    ready = redis_health.healthy or db_health.healthy  # At least one must work

    return {
        "status": "ready" if ready else "not_ready",
        "timestamp": time.time(),
        "redis": redis_health.healthy,
        "database": db_health.healthy,
    }
