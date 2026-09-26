# Objective: The EMA is written through to ema_history on every update, scoped by semantics and study period, and a
# Redis that lost it is refilled from the table instead of restarting each model from one new sample.
from __future__ import annotations

import json

import fakeredis
import pytest
from app.services import ema_persistencia, ema_store
from app.services.router_state import EMABatchQueue

ENTRADA = {"ema_latency": 2.0, "ema_quality": 8.0, "ema_cost": 0.01, "ema_alignment": 1.0, "updates": 20}


class _Conn:
    def __init__(self, log, linhas=()):
        self.log, self.linhas = log, list(linhas)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.log.append((str(sql), params))
        linhas = self.linhas
        return type("R", (), {"mappings": lambda s: type("M", (), {"all": lambda s2: linhas})()})()


def _engine(monkeypatch, log, linhas=()):
    eng = type("E", (), {"begin": lambda s: _Conn(log), "connect": lambda s: _Conn(log, linhas)})()
    monkeypatch.setattr(ema_persistencia, "_engine", lambda: eng)


@pytest.fixture(autouse=True)
def _periodo(monkeypatch):
    monkeypatch.setattr(ema_persistencia, "_escopo", lambda: {"sem": "formative_v2", "ns": "2026s2"})
    ema_store.reset_ema_snapshots()


def test_every_update_is_written_with_its_scope_and_count(monkeypatch):
    log = []
    _engine(monkeypatch, log)
    assert ema_persistencia.gravar("text", "m", ENTRADA) is True
    (sql, row), (sql_log, _) = log  # updates=20: também entra no log amostrado
    assert "INSERT INTO ema_history" in sql and "ema_history_log" in sql_log
    assert (row["sem"], row["ns"], row["u"], row["q"]) == ("formative_v2", "2026s2", 20, 8.0)


def test_a_failed_write_is_reported_not_raised(monkeypatch):
    monkeypatch.setattr(ema_persistencia, "_engine", lambda: (_ for _ in ()).throw(RuntimeError("db")))
    assert ema_persistencia.gravar("text", "m", ENTRADA) is False
    assert ema_persistencia.carregar("text") == {}


def test_reads_are_scoped_to_the_active_semantics_and_period(monkeypatch):
    log = []
    linha = {"model": "m", "ema_latency": 2, "ema_quality": 8, "ema_cost": 0.01, "ema_alignment": 1, "updates": 20}
    _engine(monkeypatch, log, [linha])
    assert ema_persistencia.carregar("text", "m") == {"m": ENTRADA}
    sql, params = log[0]
    assert "policy_namespace = :ns" in sql and "model = :m" in sql and params["ns"] == "2026s2"


def test_an_empty_redis_is_refilled_from_the_table(monkeypatch):
    rds = fakeredis.FakeRedis()
    monkeypatch.setattr(ema_persistencia, "carregar", lambda modality, model=None: {"m": dict(ENTRADA)})
    snap = ema_store.load_ema_snapshot("text", rds=rds)
    assert snap["m"]["updates"] == 20
    assert json.loads(rds.hget(ema_store._ema_key("text"), "m"))["ema_quality"] == 8.0


def test_a_missing_field_continues_the_durable_history(monkeypatch):
    rds = fakeredis.FakeRedis()
    monkeypatch.setattr(ema_persistencia, "carregar", lambda modality, model=None: {"m": dict(ENTRADA)})
    novo = ema_store.update_shared_ema("text", "m", latency_s=2.0, quality=8.0, cost=0.01, rds=rds)
    assert novo["updates"] == 21  # continua a série, não recomeça em 1


def test_the_batch_queue_keeps_rows_when_the_database_is_down():
    fila = EMABatchQueue()
    fila._persist_batch = lambda items: 0
    fila.add("text", "m", ENTRADA)
    assert fila.flush() == 0 and fila._queue  # nada foi descartado
