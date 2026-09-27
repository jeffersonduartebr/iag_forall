# Objective: A real PostgreSQL 16 for the SQL verification suite: fresh database, alembic_pg head, clean tables.
"""Fixtures of ``tests/pg``: every test calls the real function against a real PostgreSQL.

Skipped unless ``ARISTO_PG_TEST_URL`` points at a server (``postgresql+psycopg2://user:pass@host:port/postgres``).
The session creates a throwaway database, applies ``alembic -c alembic_pg.ini upgrade head`` to it (the same
chain production runs) and drops it at the end; each test starts from empty tables (``TRUNCATE ... RESTART
IDENTITY CASCADE``).

The root ``conftest`` mocks the database for unit tests (``mock_dependencies`` replaces ``app.db.get_engine`` and
``sqlalchemy.create_engine``). This package overrides that fixture by name, so here the engine is real; Redis stays
a ``MagicMock`` (it is not what these tests verify).
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

RAIZ = Path(__file__).resolve().parents[2]
URL = os.getenv("ARISTO_PG_TEST_URL", "").strip()


def _aplicar_alembic(url) -> None:
    """``alembic -c alembic_pg.ini upgrade head`` against ``url``, through the env vars ``app.db_url`` reads."""
    from alembic.config import Config

    from alembic import command

    env = {
        "DB_DIALECT": "postgresql",
        "DB_HOST": url.host or "127.0.0.1",
        "DB_PORT": str(url.port or 5432),
        "DB_USER": url.username or "postgres",
        "DB_PASS": url.password or "",
        "DB_NAME": url.database,
    }
    antes = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        cfg = Config()  # sem arquivo: o env.py não reconfigura o logging da sessão de testes
        cfg.set_main_option("script_location", str(RAIZ / "alembic_pg"))
        command.upgrade(cfg, "head")
    finally:
        for chave, valor in antes.items():
            if valor is None:
                os.environ.pop(chave, None)
            else:
                os.environ[chave] = valor


@pytest.fixture(scope="session")
def pg_engine():
    """Engine of a fresh database with the alembic_pg schema (skips the suite without ``ARISTO_PG_TEST_URL``)."""
    if not URL:
        pytest.skip("ARISTO_PG_TEST_URL não definida: suíte PostgreSQL desligada")
    servidor = make_url(URL)
    nome = f"aristo_teste_{uuid.uuid4().hex[:10]}"
    admin = create_engine(servidor, isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{nome}"'))
    alvo = servidor.set(database=nome)
    try:
        _aplicar_alembic(alvo)
        engine = create_engine(alvo, pool_pre_ping=True, connect_args={"options": "-c timezone=UTC"})
        yield engine
        engine.dispose()
    finally:
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{nome}" WITH (FORCE)'))
        admin.dispose()


@pytest.fixture(scope="session")
def _tabelas(pg_engine):
    with pg_engine.connect() as conn:
        return conn.execute(
            text(
                "SELECT tablename FROM pg_tables WHERE schemaname = current_schema() AND tablename <> 'alembic_version'"
            )
        ).scalars().all()


def _limpar_caches() -> None:
    """Process caches that would carry rows from one test into the next."""
    import app.roadmap_features as rf
    import app.settings_dynamic as sd

    sd._lru.clear()
    sd._last_prime = 0.0
    for cache in (rf._HOTPATH_POLICY_CACHE, rf._HOTPATH_TENANT_BUDGET_CACHE, rf._HOTPATH_TENANT_USAGE_CACHE):
        cache.clear()
    judges = sys.modules.get("app.judges")
    if judges is not None:
        judges._judge_stats_cache.clear()


@pytest.fixture(autouse=True)
def mock_dependencies(monkeypatch, pg_engine, _tabelas):
    """Overrides the root fixture: the database is real (empty tables), Redis is a ``MagicMock``."""
    with pg_engine.begin() as conn:
        conn.execute(text(f"TRUNCATE {', '.join(_tabelas)} RESTART IDENTITY CASCADE"))
    import app.db as db

    monkeypatch.setattr(db, "_engine", pg_engine)  # quem importou get_engine antes do patch recebe este
    monkeypatch.setattr(db, "get_engine", lambda: pg_engine)
    redis = MagicMock()
    redis.get.return_value = None
    monkeypatch.setattr("app.utils.redis_client.get_redis", lambda *a, **k: None)
    monkeypatch.setattr("app.settings_dynamic.get_redis", lambda *a, **k: redis, raising=False)
    monkeypatch.setattr("app.settings_dynamic._get_rds", lambda: None)
    _limpar_caches()
    yield pg_engine
    _limpar_caches()


@pytest.fixture
def sql(pg_engine):
    """``sql("SELECT ...", **params)`` -> list of row mappings (autocommitted helper for arrange/assert)."""

    def executar(comando: str, **params):
        with pg_engine.begin() as conn:
            resultado = conn.execute(text(comando), params)
            return [dict(r._mapping) for r in resultado] if resultado.returns_rows else resultado.rowcount

    return executar
