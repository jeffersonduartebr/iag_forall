# Objective: Test coverage for health readiness mode behavior and regressions.
"""Test coverage for health readiness mode behavior and regressions.

This test module verifies expected behavior, regression boundaries, and failure
handling for the corresponding runtime component.
"""


import os
import sys

import pytest

# Prioriza /app do projeto para resolver pacote "app" correto nos testes.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app")))


@pytest.fixture(autouse=True)
def _isolate_app_modules():
    """Reimporta ``app.*`` isoladamente por teste e restaura o estado global depois
    (evita que um ``sys.modules.pop`` de coleção contamine a suíte inteira)."""
    saved = {m: sys.modules[m] for m in list(sys.modules) if m == "app" or m.startswith("app.")}
    for _m in saved:
        sys.modules.pop(_m, None)
    try:
        yield
    finally:
        for _m in [m for m in list(sys.modules) if m == "app" or m.startswith("app.")]:
            sys.modules.pop(_m, None)
        sys.modules.update(saved)


@pytest.mark.asyncio
async def test_readiness_strict_requires_both(monkeypatch):
    """Testa readiness strict requires both."""
    from app import health

    class Obj:
        """Represent `Obj` within this module.

The class groups the state and behavior required for Obj."""
        def __init__(self, healthy):
            """Initialize the internal state required by this instance.

The constructor keeps setup local to the object so callers can use it without additional bootstrapping."""
            self.healthy = healthy

    async def _redis_ok():
        """Execute the redis ok routine.

This helper encapsulates one focused step used by the surrounding workflow."""
        return Obj(True)

    async def _db_down():
        """Execute the db down routine.

This helper encapsulates one focused step used by the surrounding workflow."""
        return Obj(False)

    monkeypatch.setenv("READINESS_MODE", "strict")
    monkeypatch.setattr("app.health.check_redis_health", _redis_ok)
    monkeypatch.setattr("app.health.check_database_health", _db_down)

    result = await health.get_readiness_check()

    assert result["mode"] == "strict"
    assert result["status"] == "not_ready"
    assert result["redis"] is True
    assert result["database"] is False


@pytest.mark.asyncio
async def test_readiness_degraded_accepts_one_dependency(monkeypatch):
    """Testa readiness degraded accepts one dependency."""
    from app import health

    class Obj:
        """Represent `Obj` within this module.

The class groups the state and behavior required for Obj."""
        def __init__(self, healthy):
            """Initialize the internal state required by this instance.

The constructor keeps setup local to the object so callers can use it without additional bootstrapping."""
            self.healthy = healthy

    async def _redis_ok():
        """Execute the redis ok routine.

This helper encapsulates one focused step used by the surrounding workflow."""
        return Obj(True)

    async def _db_down():
        """Execute the db down routine.

This helper encapsulates one focused step used by the surrounding workflow."""
        return Obj(False)

    monkeypatch.setenv("READINESS_MODE", "degraded")
    monkeypatch.setattr("app.health.check_redis_health", _redis_ok)
    monkeypatch.setattr("app.health.check_database_health", _db_down)

    result = await health.get_readiness_check()

    assert result["mode"] == "degraded"
    assert result["status"] == "ready"
    assert result["redis"] is True
    assert result["database"] is False


@pytest.mark.asyncio
async def test_readiness_defaults_to_strict(monkeypatch):
    """Testa readiness defaults to strict."""
    from app import health

    class Obj:
        """Represent `Obj` within this module.

The class groups the state and behavior required for Obj."""
        def __init__(self, healthy):
            """Initialize the internal state required by this instance.

The constructor keeps setup local to the object so callers can use it without additional bootstrapping."""
            self.healthy = healthy

    async def _redis_ok():
        """Execute the redis ok routine.

This helper encapsulates one focused step used by the surrounding workflow."""
        return Obj(True)

    async def _db_ok():
        """Execute the db ok routine.

This helper encapsulates one focused step used by the surrounding workflow."""
        return Obj(True)

    monkeypatch.delenv("READINESS_MODE", raising=False)
    monkeypatch.setattr("app.health.check_redis_health", _redis_ok)
    monkeypatch.setattr("app.health.check_database_health", _db_ok)

    result = await health.get_readiness_check()

    assert result["mode"] == "strict"
    assert result["status"] == "ready"
