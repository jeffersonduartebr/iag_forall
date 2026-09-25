# Objective: Correlation service runtime — Redis/DB connectivity, bounded DB wait, fetch fallbacks, main loop.
"""Coverage of app.correlation_metrics I/O paths with fake engines, clocks and loop breakers."""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.exc import OperationalError

from app import correlation_metrics as cm


class _Stop(BaseException):
    """Escapes the service's ``while True`` (its handler only catches Exception)."""


class _Clock:
    def __init__(self, stop_after=None):
        self.now, self.sleeps, self.stop_after = 0.0, [], stop_after

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds
        if self.stop_after is not None and len(self.sleeps) >= self.stop_after:
            raise _Stop


class _Engine:
    """Minimal engine: ``connect``/``begin`` yield a connection; the first ``failures`` connects fail."""

    def __init__(self, failures=0):
        self.failures, self.executed = failures, []

    @contextmanager
    def _conn(self):
        if self.failures:
            self.failures -= 1
            raise OperationalError("SELECT 1", {}, Exception("db starting"))
        yield self

    connect = begin = _conn

    def execute(self, statement):
        self.executed.append(str(statement))


def test_connect_redis_success_and_failure(monkeypatch):
    class _Client:
        def __init__(self, **kw):
            self.kw = kw

        def ping(self):
            if self.kw["host"] == "down":
                raise ConnectionError("refused")

    monkeypatch.setattr(cm.redis, "Redis", _Client)
    monkeypatch.setattr(cm, "REDIS_HOST", "up")
    assert cm._connect_redis().kw["decode_responses"] is True
    monkeypatch.setattr(cm, "REDIS_HOST", "down")
    assert cm._connect_redis() is None


def test_wait_for_db_retries_with_capped_backoff(monkeypatch):
    clock = _Clock()
    engine = _Engine(failures=3)
    monkeypatch.setattr(cm, "time", clock)
    monkeypatch.setattr(cm, "db_engine", engine)
    cm.wait_for_db(max_wait_seconds=60)
    assert clock.sleeps == [1.0, 1.5, 2.25]
    assert engine.executed == ["SELECT 1"]


def test_wait_for_db_gives_up_after_deadline(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(cm, "time", clock)
    monkeypatch.setattr(cm, "db_engine", _Engine(failures=10**6))
    with pytest.raises(OperationalError):
        cm.wait_for_db(max_wait_seconds=40)
    assert clock.now > 40 and max(clock.sleeps) == 8.0  # backoff capped at 8 s


def test_ensure_history_table_runs_idempotent_ddl(monkeypatch):
    engine = _Engine()
    monkeypatch.setattr(cm, "db_engine", engine)
    cm.ensure_history_table()
    assert "CREATE TABLE IF NOT EXISTS correlation_history" in engine.executed[0]


def test_fetch_recent_metrics_uses_window_and_degrades_to_empty(monkeypatch):
    seen = []
    monkeypatch.setattr(cm.pd, "read_sql", lambda q, eng: seen.append(q) or pd.DataFrame({"model": ["a"]}))
    assert len(cm.fetch_recent_metrics("2 HOUR")) == 1
    assert "INTERVAL 2 HOUR" in seen[0]
    monkeypatch.setattr(cm.pd, "read_sql", lambda q, eng: pd.DataFrame())
    assert cm.fetch_recent_metrics().empty

    def _missing_table(q, eng):
        raise RuntimeError("Table 'model_metrics' doesn't exist")

    monkeypatch.setattr(cm.pd, "read_sql", _missing_table)
    assert cm.fetch_recent_metrics().empty


def test_correlation_matrix_falls_back_to_identity(monkeypatch):
    group = pd.DataFrame({"latency_ms": [1.0, 2.0, 3.0], "cost_usd": [3.0, 1.0, 2.0], "quality_score": [1.0, 2.0, 3.0],
                          "fitness": [1.0, 2.0, 3.0], "generation": [1.0, 4.0, 2.0]})

    def _boom(*a, **k):
        raise FloatingPointError

    monkeypatch.setattr(cm.np, "corrcoef", _boom)
    out = cm._model_correlations(group)
    assert np.array_equal(out["matrix"], np.eye(3))
    assert out["corr_lq"] == 0.0 and out["generation"] == 4


def test_publish_replaces_nan_matrix_cells_and_skips_r2_when_empty():
    data = {"corr_lq": 0.5, "corr_cq": 0.0, "corr_fw": 0.0, "labels": ["latency", "cost", "quality"],
            "matrix": np.full((3, 3), np.nan), "generation": 1}
    cm.nsga_correlation_r2_mean.set(-1.0)
    cm.publish_metrics({})
    assert cm.nsga_correlation_r2_mean._value.get() == -1.0
    cm.publish_metrics({"nan-model": data})
    assert cm.model_correlation_matrix.labels(model="nan-model", metric_x="cost", metric_y="latency")._value.get() == 0.0
    assert cm.nsga_correlation_r2_mean._value.get() == pytest.approx(0.25 / 3)


def _run_main(monkeypatch, frames, stop_after):
    clock = _Clock(stop_after=stop_after)
    calls = {"published": [], "persisted": [], "http": []}
    monkeypatch.setattr(cm, "time", clock)
    monkeypatch.setattr(cm, "start_http_server", lambda port: calls["http"].append(port))
    monkeypatch.setattr(cm, "wait_for_db", lambda: None)
    monkeypatch.setattr(cm, "ensure_history_table", lambda: None)
    monkeypatch.setattr(cm, "fetch_recent_metrics", lambda window_sql: frames.pop(0))
    monkeypatch.setattr(cm, "publish_metrics", lambda c: calls["published"].append(sorted(c)))
    monkeypatch.setattr(cm, "persist_correlations", lambda c: calls["persisted"].append(sorted(c)))
    with pytest.raises(_Stop):
        cm.main()
    return clock, calls


def test_main_loop_skips_empty_and_thin_windows_then_publishes(monkeypatch):
    rich = pd.DataFrame({"model": ["m"] * 3, "latency_ms": [1, 2, 3], "cost_usd": [1, 2, 3], "quality_score": [3, 2, 1],
                         "fitness": [1, 1, 2], "generation": [1, 2, 3]})
    thin = rich.head(2)
    clock, calls = _run_main(monkeypatch, [pd.DataFrame(), thin, rich], stop_after=3)
    assert calls["http"] == [cm.PROM_PORT]
    assert calls["published"] == [["m"]] and calls["persisted"] == [["m"]]
    assert clock.sleeps == [cm.UPDATE_INTERVAL] * 3


def test_main_loop_survives_iteration_errors(monkeypatch):
    def _boom(window_sql):
        raise RuntimeError("db flapped")

    frames = []
    clock = _Clock(stop_after=2)
    monkeypatch.setattr(cm, "fetch_recent_metrics", _boom)
    for name in ("start_http_server", "wait_for_db", "ensure_history_table"):
        monkeypatch.setattr(cm, name, lambda *a: None)
    monkeypatch.setattr(cm, "time", clock)
    with pytest.raises(_Stop):
        cm.main()
    assert clock.sleeps == [cm.UPDATE_INTERVAL] * 2 and frames == []
