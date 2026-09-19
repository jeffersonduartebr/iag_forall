# Objective: Regression tests for batched (executemany) upserts against PyMySQL's real SQL rewrite.
"""Batched ``INSERT ... ON DUPLICATE KEY UPDATE`` must survive PyMySQL's executemany rewrite.

PyMySQL turns ``executemany`` of an ``INSERT ... VALUES (...)`` into one
multi-row INSERT, substituting parameters only inside ``VALUES (...)``; the
``ON DUPLICATE KEY UPDATE`` tail is copied verbatim. A bound parameter there
(``avg_reward = :avg``) reached MariaDB as ``%(avg)s`` (error 1064), which the
end-to-end run caught and the fake engines in the unit tests did not. These
tests run the real rewrite offline (``defer_connect``) on each statement.
"""

from contextlib import contextmanager
from types import SimpleNamespace

import pymysql
import pytest
from app.services import bandit_stats_store
from sqlalchemy.dialects import mysql

from app import router_core, update_nsga_best_params


def _pymysql_bulk_sql(stmt, rows):
    """Final SQL PyMySQL would send for ``executemany(stmt, rows)``."""
    sql = str(stmt.compile(dialect=mysql.dialect(paramstyle="pyformat")))
    conn = pymysql.connections.Connection(defer_connect=True, charset="utf8mb4")
    conn.server_status = 0  # lido ao escapar strings; normalmente preenchido no handshake
    cursor = conn.cursor()
    sent = []
    cursor.execute = lambda query, args=None: sent.append(bytes(query).decode()) or 0
    cursor.executemany(sql, rows)
    assert len(sent) == 1, "deveria virar um único INSERT multi-linha"
    return sent[0]


def _assert_fully_bound(final_sql):
    assert "%(" not in final_sql, final_sql
    assert "ON DUPLICATE KEY UPDATE" in final_sql


def test_bandit_stats_batch_upsert_is_fully_bound():
    rows = [
        {"ctx": "c1", "model": "m1", "avg": 0.5, "count": 3, "var": 0.1, "M2": 0.2},
        {"ctx": "c2", "model": "m2", "avg": 0.7, "count": 5, "var": 0.0, "M2": 0.0},
    ]
    final = _pymysql_bulk_sql(bandit_stats_store._UPSERT_SQL, rows)
    _assert_fully_bound(final)
    assert "avg_reward = VALUES(avg_reward)" in final


def test_ema_batch_upsert_is_fully_bound():
    rows = [
        router_core._ema_row("text", "m1", {"ema_latency": 1.0, "ema_quality": 8.0, "ema_cost": 0.1, "updates": 3}),
        router_core._ema_row("text", "m2", {"ema_latency": 2.0, "ema_quality": 6.0, "ema_cost": 0.2, "updates": 4}),
    ]
    _assert_fully_bound(_pymysql_bulk_sql(router_core._EMA_UPSERT_SQL, rows))


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
    _assert_fully_bound(_pymysql_bulk_sql(stmt, rows))


@pytest.mark.parametrize("tail", ["b = :b", "b = VALUES(b)"])
def test_rewrite_detects_bound_parameters_in_the_update_tail(tail):
    from sqlalchemy import text

    final = _pymysql_bulk_sql(
        text(f"INSERT INTO t (a, b) VALUES (:a, :b) ON DUPLICATE KEY UPDATE {tail}"), [{"a": 1, "b": 2}]
    )
    assert ("%(" in final) == (tail == "b = :b")
