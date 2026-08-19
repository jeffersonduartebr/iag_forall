# Objective: Shared pytest fixtures, test doubles, and test-time configuration.
"""Shared pytest fixtures, test doubles, and test-time configuration.

This module is part of the tracked codebase and should remain aligned with the
current runtime architecture and operational documentation.
"""


import os
import sys
from unittest.mock import MagicMock

import pytest

# Prioriza o pacote de aplicação real em app/app (importável como "app").
APP_PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
TESTS_ROOT = os.path.abspath(os.path.dirname(__file__))
if APP_PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, APP_PACKAGE_ROOT)
if TESTS_ROOT not in sys.path:
    sys.path.insert(0, TESTS_ROOT)

os.environ.setdefault("ADMIN_TOKEN", "test-admin-token-for-ci")
os.environ.setdefault("REQUIRE_API_AUTH", "0")

# Importa os módulos para garantir que estão disponíveis antes do mocking
try:
    import app.settings_dynamic  # noqa: F401  (garante importável antes do mocking)
    import app.utils.redis_client  # noqa: F401  (garante importável antes do mocking)
except ImportError:
    pass  # Ignora se não conseguir importar


def _make_mock_db_engine():
    """Return a SQLAlchemy-like engine stub for unit tests."""
    mock_conn = MagicMock()
    mock_conn.execute.return_value = MagicMock(fetchone=MagicMock(return_value=None))

    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_conn)
    mock_ctx.__exit__ = MagicMock(return_value=False)

    mock_engine = MagicMock()
    mock_engine.connect.return_value = mock_ctx
    mock_engine.begin.return_value = mock_ctx
    return mock_engine


def pytest_configure(config):
    """Register shared custom markers even when pytest.ini is not mounted."""
    config.addinivalue_line("markers", "integration: tests that require external services or credentials")
    config.addinivalue_line("markers", "slow: long-running tests")

@pytest.fixture(autouse=True)
def mock_dependencies(monkeypatch):
    """
    Mocka automaticamente conexões externas para TODOS os testes.
    """
    try:
        from app.settings_dynamic import _lru

        _lru.clear()
    except Exception:
        pass

    mock_engine = _make_mock_db_engine()
    try:
        import app.db as db_module

        db_module._engine = None
        db_module._engine_initialized = False
        monkeypatch.setattr("app.db.get_engine", lambda: mock_engine)
        monkeypatch.setattr("app.db.check_db_health", lambda: {"healthy": True, "latency_ms": 1.0, "pool_stats": {"status": "ok"}})
    except Exception:
        pass

    # Mock Redis
    mock_redis = MagicMock()
    try:
        monkeypatch.setattr("app.utils.redis_client.get_redis", lambda *args: mock_redis)
        monkeypatch.setattr("app.settings_dynamic.get_redis", lambda *args: mock_redis)

        async def _mock_get_redis_async():
            return None

        monkeypatch.setattr("app.utils.redis_client.get_redis_async", _mock_get_redis_async)
        monkeypatch.setattr("app.utils.redis_client._async_redis_client", None)
    except (ImportError, AttributeError):
        pass  # Ignora se os módulos não existirem

    # Mock SQLAlchemy Engine (fallback for modules that still call create_engine directly)
    try:
        monkeypatch.setattr("sqlalchemy.create_engine", lambda *args, **kwargs: mock_engine)
    except (ModuleNotFoundError, AttributeError):
        pass

    # Mock ChromaDB
    try:
        monkeypatch.setattr("chromadb.PersistentClient", MagicMock())
    except (ImportError, AttributeError):
        pass

    try:
        from app.providers_async import reset_provider_runtime_state

        reset_provider_runtime_state()
    except Exception:
        pass

    try:
        from app.reliability import RequestDeduplicator

        RequestDeduplicator._instance = None
    except Exception:
        pass

    try:
        async def _noop_record_request_outcome_async(*, settings_getter, success):
            return None

        monkeypatch.setattr(
            "app.services.router_resilience.record_request_outcome_async",
            _noop_record_request_outcome_async,
        )
    except Exception:
        pass

    # Mock Settings Defaults para consistência
    monkeypatch.setenv("NSGA_W_QUALITY", "1.0")
    monkeypatch.setenv("NSGA_W_LATENCY", "0.5")
    monkeypatch.setenv("NSGA_W_COST", "50.0")
