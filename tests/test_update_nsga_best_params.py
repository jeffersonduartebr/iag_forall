# Objective: Test coverage for publishing per-model NSGA weights (DB upsert + Redis).
"""update_nsga_best_params.persist_weights: one batched upsert, then the Redis copy."""

import json
from contextlib import contextmanager
from types import SimpleNamespace

import fakeredis
import pytest
from sqlalchemy.exc import SQLAlchemyError

from app import update_nsga_best_params as unb


@pytest.fixture
def stores(monkeypatch):
    calls = []

    class _Conn:
        def execute(self, stmt, params):
            calls.append(params)

    @contextmanager
    def begin():
        yield _Conn()

    rds = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(unb, "engine", SimpleNamespace(begin=begin))
    monkeypatch.setattr(unb, "rds", rds)
    return calls, rds


def test_persist_weights_batches_db_rows_and_publishes(stores):
    calls, rds = stores
    unb.persist_weights("text", {"m1": 0.7, "m2": 0.3})
    assert calls == [[{"mod": "text", "model": "m1", "w": 0.7}, {"mod": "text", "model": "m2", "w": 0.3}]]
    assert json.loads(rds.get("nsga:weights:text")) == {"m1": 0.7, "m2": 0.3}


def test_persist_weights_empty_is_noop(stores):
    calls, rds = stores
    unb.persist_weights("text", {})
    assert calls == [] and rds.keys() == []


def test_db_failure_still_publishes_to_redis(monkeypatch, stores):
    _, rds = stores

    @contextmanager
    def broken():
        raise SQLAlchemyError("db down")
        yield

    monkeypatch.setattr(unb, "engine", SimpleNamespace(begin=broken))
    unb.persist_weights("vision", {"m": 1.0})
    assert json.loads(rds.get("nsga:weights:vision")) == {"m": 1.0}

    monkeypatch.setattr(unb, "rds", SimpleNamespace(set=lambda *a: (_ for _ in ()).throw(ConnectionError())))
    unb.persist_weights("vision", {"m": 1.0})  # erro de Redis só é registrado
    monkeypatch.setattr(unb, "rds", None)
    unb.persist_weights("vision", {"m": 1.0})
