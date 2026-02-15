"""Módulo `tests/test_health_readiness_mode.py`: descreve responsabilidades e integrações deste arquivo."""

import os
import sys

import pytest

# Prioriza /app do projeto para resolver pacote "app" correto nos testes.
for _mod in [m for m in list(sys.modules) if m == "app" or m.startswith("app.")]:
    sys.modules.pop(_mod, None)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app")))


@pytest.mark.asyncio
async def test_readiness_strict_requires_both(monkeypatch):
    """Testa readiness strict requires both."""
    from app import health

    class Obj:
        """Classe `Obj`: concentra responsabilidades de test health readiness mode."""
        def __init__(self, healthy):
            """Inicializa estado interno necessário para uso da classe."""
            self.healthy = healthy

    async def _redis_ok():
        """Executa redis ok."""
        return Obj(True)

    async def _db_down():
        """Executa db down."""
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
        """Classe `Obj`: concentra responsabilidades de test health readiness mode."""
        def __init__(self, healthy):
            """Inicializa estado interno necessário para uso da classe."""
            self.healthy = healthy

    async def _redis_ok():
        """Executa redis ok."""
        return Obj(True)

    async def _db_down():
        """Executa db down."""
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
        """Classe `Obj`: concentra responsabilidades de test health readiness mode."""
        def __init__(self, healthy):
            """Inicializa estado interno necessário para uso da classe."""
            self.healthy = healthy

    async def _redis_ok():
        """Executa redis ok."""
        return Obj(True)

    async def _db_ok():
        """Executa db ok."""
        return Obj(True)

    monkeypatch.delenv("READINESS_MODE", raising=False)
    monkeypatch.setattr("app.health.check_redis_health", _redis_ok)
    monkeypatch.setattr("app.health.check_database_health", _db_ok)

    result = await health.get_readiness_check()

    assert result["mode"] == "strict"
    assert result["status"] == "ready"
