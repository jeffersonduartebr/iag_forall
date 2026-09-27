# Objective: One place builds the database URL and driver options for MariaDB (default) and PostgreSQL (Cloud SQL).
from __future__ import annotations

import pytest

from app import db


def test_postgres_url_uses_psycopg2_and_its_port(monkeypatch):
    monkeypatch.setenv("DB_DIALECT", "postgresql")
    monkeypatch.delenv("DB_PORT", raising=False)
    monkeypatch.setenv("DB_HOST", "pgbouncer")
    monkeypatch.setenv("DB_USER", "aristo")
    monkeypatch.setenv("DB_PASS", "s@nha:/x")
    monkeypatch.setenv("DB_NAME", "aristo")
    assert db.get_db_url() == "postgresql+psycopg2://aristo:s%40nha%3A%2Fx@pgbouncer:5432/aristo"


def test_postgresql_is_the_default(monkeypatch):
    monkeypatch.delenv("DB_DIALECT", raising=False)
    monkeypatch.delenv("DB_PORT", raising=False)
    assert db.get_db_url().startswith("postgresql+psycopg2://") and ":5432/" in db.get_db_url()


def test_an_unknown_dialect_is_an_error(monkeypatch):
    monkeypatch.setenv("DB_DIALECT", "oracle")
    with pytest.raises(ValueError):
        db.get_db_url()


def test_postgres_connections_have_timeouts_keepalives_and_tls(monkeypatch):
    monkeypatch.setenv("DB_DIALECT", "postgresql")
    monkeypatch.setenv("DB_SSLMODE", "require")
    args = db._connect_args()
    assert args["connect_timeout"] > 0 and args["keepalives"] == 1 and args["sslmode"] == "require"
    assert "read_timeout" not in args  # opção do PyMySQL: o psycopg2 recusaria
