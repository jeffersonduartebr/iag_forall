# Objective: Batched (executemany) upserts compile to PostgreSQL ON CONFLICT with every parameter bound.
"""Batched ``INSERT ... ON CONFLICT DO UPDATE`` statements, compiled offline for psycopg2.

The MariaDB version of these tests ran PyMySQL's executemany rewrite, which substitutes parameters only inside
``VALUES (...)`` (a bound parameter in the update tail reached the server as ``%(avg)s``). Under psycopg2 the
statement is compiled once in ``pyformat`` and executed per row, so the checks here are that each statement
compiles for the PostgreSQL dialect, that every ``:name`` became a ``%(name)s`` the rows provide, and that the
update tail reads the proposed row through ``EXCLUDED`` with no MySQL syntax left. The behaviour against a real
server (same key twice in one batch, never-regressing count) is in ``tests/pg``.
"""

import re
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from app.services import bandit_stats_store
from sqlalchemy.dialects import postgresql

from app import update_nsga_best_params

_MYSQL = ("ON DUPLICATE KEY", "VALUES(", "IF(", "INSERT IGNORE")


def _psycopg2_sql(stmt, rows):
    """SQL psycopg2 receives for ``executemany(stmt, rows)``; fails if a row lacks a bound name."""
    compilado = stmt.compile(dialect=postgresql.psycopg2.dialect())
    sql = str(compilado)
    nomes = set(re.findall(r"%\((\w+)\)s", sql))
    for row in rows:
        assert nomes <= set(row), f"parâmetros sem valor: {nomes - set(row)}"
    return sql


def _assert_postgres_upsert(sql):
    assert "ON CONFLICT" in sql and "EXCLUDED." in sql, sql
    assert not any(m in sql for m in _MYSQL), sql
    tail = sql.split("DO UPDATE", 1)[1]
    assert "%(" not in tail, "o UPDATE lê EXCLUDED, não parâmetros"


def test_bandit_stats_batch_upsert_is_fully_bound():
    rows = [
        {"ctx": "c1", "model": "m1", "avg": 0.5, "count": 3, "var": 0.1, "m2": 0.2},
        {"ctx": "c2", "model": "m2", "avg": 0.7, "count": 5, "var": 0.0, "m2": 0.0},
    ]
    final = _psycopg2_sql(bandit_stats_store._UPSERT_SQL, rows)
    _assert_postgres_upsert(final)
    assert "ON CONFLICT (context_label, model)" in final
    assert "GREATEST(COALESCE(bandit_context_stats.count, 0), EXCLUDED.count)" in final


def test_ema_batch_upsert_is_fully_bound():
    from app.services import ema_persistencia

    rows = [
        ema_persistencia.linha("text", "m1", {"ema_latency": 1.0, "ema_quality": 8.0, "ema_cost": 0.1, "updates": 3}),
        ema_persistencia.linha("text", "m2", {"ema_latency": 2.0, "ema_quality": 6.0, "ema_cost": 0.2, "updates": 4}),
    ]
    final = _psycopg2_sql(ema_persistencia._UPSERT, rows)
    _assert_postgres_upsert(final)
    assert "ON CONFLICT (model, modality, semantics, policy_namespace)" in final


def test_nsga_weights_batch_upsert_is_fully_bound(monkeypatch):
    captured = []

    class _Conn:
        def execute(self, stmt, params):
            captured.append((stmt, params))

    @contextmanager
    def begin():
        yield _Conn()

    monkeypatch.setattr(update_nsga_best_params, "engine", SimpleNamespace(begin=begin))
    monkeypatch.setattr(update_nsga_best_params, "rds", None)
    update_nsga_best_params.persist_weights("text", {"m1": 0.6, "m2": 0.4})

    ((stmt, rows),) = captured
    assert len(rows) == 2  # um executemany com todos os modelos
    final = _psycopg2_sql(stmt, rows)
    _assert_postgres_upsert(final)
    assert "ON CONFLICT (modality, model)" in final


@pytest.mark.parametrize("tail", ["b = :b", "b = EXCLUDED.b"])
def test_the_check_detects_bound_parameters_in_the_update_tail(tail):
    from sqlalchemy import text

    final = _psycopg2_sql(text(f"INSERT INTO t (a, b) VALUES (:a, :b) ON CONFLICT (a) DO UPDATE SET {tail}"),
                          [{"a": 1, "b": 2}])
    assert ("%(" in final.split("DO UPDATE", 1)[1]) == (tail == "b = :b")
