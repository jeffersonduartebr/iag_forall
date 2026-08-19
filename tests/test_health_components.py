# Objective: Test coverage for health components behavior and regressions.
"""Test coverage for health components behavior and regressions.

This test module verifies expected behavior, regression boundaries, and failure
handling for the corresponding runtime component.
"""


import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app import health


class _Response:
    """Represent `_Response` within this module.

The class groups the state and behavior required for Response."""
    def __init__(self, data=None, fail=False):
        """Initialize the internal state required by this instance.

The constructor keeps setup local to the object so callers can use it without additional bootstrapping."""
        self._data = data or {}
        self._fail = fail

    def raise_for_status(self):
        """Execute the raise for status routine.

This helper encapsulates one focused step used by the surrounding workflow."""
        if self._fail:
            raise RuntimeError("bad status")

    def json(self):
        """Execute the json routine.

This helper encapsulates one focused step used by the surrounding workflow."""
        return self._data


@pytest.mark.asyncio
async def test_component_health_checks_success_paths(monkeypatch):
    """Testa component health checks success paths."""
    monkeypatch.setattr(
        "app.utils.redis_client.check_redis_health",
        lambda: {"healthy": True, "latency_ms": 1.2, "pool_size": 4},
    )
    monkeypatch.setattr(
        "app.db.check_db_health",
        lambda: {"healthy": True, "latency_ms": 1.0, "pool_stats": {"status": "ok"}},
    )

    fake_chromadb = SimpleNamespace(
        PersistentClient=lambda path: SimpleNamespace(list_collections=lambda: ["c1", "c2"])
    )
    monkeypatch.setitem(sys.modules, "chromadb", fake_chromadb)

    client = SimpleNamespace(head=AsyncMock(return_value=_Response({"models": [{"name": "m1"}, {"name": "m2"}]})))
    monkeypatch.setattr("app.providers_async.get_http_client", AsyncMock(return_value=client))

    manager = SimpleNamespace(get_all_statuses=lambda: [{"model": "a", "state": "closed"}])
    monkeypatch.setattr("app.reliability.get_circuit_breaker_manager", lambda: manager)

    redis = await health.check_redis_health()
    db = await health.check_database_health()
    chroma = await health.check_vectorstore_health()
    ollama = await health.check_ollama_health()
    cb = await health.check_circuit_breakers_health()

    assert redis.healthy and redis.details["pool_size"] == 4
    assert db.healthy
    assert chroma.healthy and chroma.details["collections"] == 2
    assert ollama.healthy and ollama.details["probe"] == "HEAD /"
    assert cb.healthy is True


@pytest.mark.asyncio
async def test_component_health_error_paths_and_cache(monkeypatch):
    """Testa component health error paths and cache."""
    monkeypatch.setattr("app.utils.redis_client.check_redis_health", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    redis = await health.check_redis_health()
    assert redis.healthy is False

    client = SimpleNamespace(head=AsyncMock(return_value=_Response(fail=True)))
    monkeypatch.setattr("app.providers_async.get_http_client", AsyncMock(return_value=client))
    ollama = await health.check_ollama_health()
    assert ollama.healthy is False

    async def _ok(name, healthy=True):
        """Execute the ok routine.

This helper encapsulates one focused step used by the surrounding workflow."""
        return health.ComponentHealth(name=name, healthy=healthy)

    monkeypatch.setattr(health, "check_redis_health", lambda: _ok("redis", True))
    monkeypatch.setattr(health, "check_database_health", lambda: _ok("mariadb", True))
    monkeypatch.setattr(health, "check_vectorstore_health", lambda: _ok("chromadb", True))
    monkeypatch.setattr(health, "check_ollama_health", lambda: _ok("ollama", True))
    monkeypatch.setattr(health, "check_circuit_breakers_health", lambda: _ok("circuit_breakers", True))

    health.invalidate_health_cache()
    first = await health.get_full_health_check(force_refresh=True)
    second = await health.get_full_health_check(force_refresh=False)

    assert first["status"] == "healthy"
    assert second["cached"] is True


@pytest.mark.asyncio
async def test_component_to_dict_and_liveness():
    """Testa component to dict and liveness."""
    comp = health.ComponentHealth(name="x", healthy=False, latency_ms=1.2345, error="err", details={"a": 1})
    d = comp.to_dict()
    assert d["latency_ms"] == 1.23
    assert d["error"] == "err"
    assert d["details"]["a"] == 1

    live = await health.get_liveness_check()
    assert live["status"] == "alive"


@pytest.mark.asyncio
async def test_full_health_handles_exception_results_and_degraded_readiness(monkeypatch):
    """Full health should classify exception entries and readiness should honor degraded mode."""
    async def _redis():
        return health.ComponentHealth(name="redis", healthy=False)

    async def _db():
        return health.ComponentHealth(name="mariadb", healthy=True)

    async def _vector_fail():
        raise RuntimeError("vector fail")

    monkeypatch.setattr(health, "check_redis_health", _redis)
    monkeypatch.setattr(health, "check_database_health", _db)
    monkeypatch.setattr(health, "check_vectorstore_health", _vector_fail)
    monkeypatch.setattr(health, "check_ollama_health", _db)
    monkeypatch.setattr(health, "check_circuit_breakers_health", _db)

    health.invalidate_health_cache()
    result = await health.get_full_health_check(force_refresh=True)
    assert result["status"] in {"degraded", "unhealthy"}
    assert "unknown" in result["components"]

    monkeypatch.setenv("READINESS_MODE", "degraded")
    ready = await health.get_readiness_check()
    assert ready["status"] == "ready"

    monkeypatch.setenv("READINESS_MODE", "strict")
    not_ready = await health.get_readiness_check()
    assert not_ready["status"] == "not_ready"
