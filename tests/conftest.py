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


try:
    from hypothesis import settings as hypothesis_settings

    # Sem deadline: CI compartilhado tem latência variável. O perfil "ci" fixa a semente.
    hypothesis_settings.register_profile("dev", deadline=None, max_examples=100)
    hypothesis_settings.register_profile("ci", deadline=None, max_examples=200, derandomize=True)
    hypothesis_settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))
except ImportError:
    pass


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
    config.addinivalue_line("markers", "contract: OpenAPI contract fuzzing (schemathesis)")

@pytest.fixture
def fake_redis_server():
    """One in-memory Redis server shared by ``fake_redis`` and ``fake_aioredis``."""
    import fakeredis

    return fakeredis.FakeServer()


@pytest.fixture
def fake_redis(monkeypatch, fake_redis_server):
    """Opt-in sync fakeredis client behind the ``app.utils.redis_client`` getters.

    The autouse ``MagicMock`` Redis accepts any call and returns mocks, so
    parsing code silently falls into its error branches; this fixture gives
    real Redis semantics. Modules that bound a getter at import time
    (``from ..redis_client import get_redis``) still need their own patch.
    """
    import fakeredis

    client = fakeredis.FakeRedis(server=fake_redis_server)
    for name in ("get_redis", "get_redis_async_safe", "get_redis_sync_nonblocking", "ensure_redis_connected"):
        monkeypatch.setattr(f"app.utils.redis_client.{name}", lambda *a, **k: client)
    return client


@pytest.fixture
def fake_aioredis(monkeypatch, fake_redis_server):
    """Opt-in ``redis.asyncio`` fakeredis client (same server as ``fake_redis``)."""
    import fakeredis

    client = fakeredis.FakeAsyncRedis(server=fake_redis_server)

    async def _get_redis_async():
        return client

    monkeypatch.setattr("app.utils.redis_client.get_redis_async", _get_redis_async)
    monkeypatch.setattr("app.utils.redis_async_ops.get_redis_async", _get_redis_async)
    return client


@pytest.fixture(autouse=True)
def _isolate_experiment_manifests(monkeypatch, tmp_path):
    """Keep eval runs from rewriting the tracked manifests in thesis_results/."""
    monkeypatch.setenv("EXPERIMENT_MANIFEST_DIR", str(tmp_path / "experiment_manifests"))


@pytest.fixture(autouse=True)
def mock_dependencies(monkeypatch):
    """
    Mocka automaticamente conexões externas para TODOS os testes.
    """
    try:
        import app.settings_dynamic as settings_dynamic_module

        settings_dynamic_module._lru.clear()
        settings_dynamic_module._last_prime = 0.0
    except Exception:
        pass
    try:
        from app.services.bandit_stats_store import cold_contexts

        cold_contexts.clear()
    except Exception:
        pass
    judges_module = sys.modules.get("app.judges")
    if judges_module is not None:
        judges_module._judge_stats_cache.clear()
    ema_module = sys.modules.get("app.services.ema_store")
    if ema_module is not None:
        ema_module.reset_ema_snapshots()

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
