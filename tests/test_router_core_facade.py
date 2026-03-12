"""Focused tests for router_core façade helpers and delegations."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app import router_core as rc


def test_settings_helpers_and_resilience_delegation(monkeypatch):
    """router_core helper wrappers should delegate to extracted resilience helpers."""
    monkeypatch.setattr(rc, "settings", SimpleNamespace(get=lambda key, default=None: {"A": "7"}.get(key, default)))
    assert rc._settings_getter("A", 1) == "7"

    monkeypatch.setattr(rc, "settings", SimpleNamespace(A="8"))
    assert rc._settings_getter("A", 1) == "8"

    calls = {}
    monkeypatch.setattr(rc, "_safe_setting_int_impl", lambda getter, key, default: calls.setdefault("int", (getter("A", 0), key, default)) or 9)
    monkeypatch.setattr(rc, "_safe_setting_float_impl", lambda getter, key, default: calls.setdefault("float", (key, default)) or 1.2)
    monkeypatch.setattr(rc, "_safe_setting_bool_impl", lambda getter, key, default: calls.setdefault("bool", (key, default)) or True)
    monkeypatch.setattr(rc, "settings", SimpleNamespace(get=lambda key, default=None: {"A": "5"}.get(key, default)))

    assert rc._safe_setting_int("A", 1) == calls["int"]
    assert rc._safe_setting_float("F", 1.0) == calls["float"]
    assert rc._safe_setting_bool("B", False) == calls["bool"]

    monkeypatch.setattr(rc, "get_router_redis", lambda: "redis")
    assert rc._get_rds() == "redis"

    monkeypatch.setattr(rc, "_record_dependency_breaker_metrics_impl", lambda: calls.setdefault("metrics", True))
    rc._record_dependency_breaker_metrics()
    assert calls["metrics"] is True

    monkeypatch.setattr(__import__("app.services.router_resilience", fromlist=["error_budget_window"]), "error_budget_window", lambda getter: (11, 0.2, 3))
    assert rc._error_budget_window() == (11, 0.2, 3)

    monkeypatch.setattr(rc, "_record_request_outcome_impl", lambda settings_getter, success: calls.setdefault("outcome", success))
    monkeypatch.setattr(rc, "_is_error_budget_exceeded_impl", lambda settings_getter: True)
    rc._record_request_outcome(False)
    assert calls["outcome"] is False
    assert rc._is_error_budget_exceeded() is True


def test_get_dynamic_strategy_weights_and_persist_ema(monkeypatch):
    """router_core weight and persistence helpers should stay lightweight."""
    monkeypatch.setattr(rc, "settings", SimpleNamespace(NSGA_W_QUALITY=2.0, NSGA_W_LATENCY=3.0, NSGA_W_COST=4.0))
    assert rc.get_dynamic_strategy_weights("text") == {"w_quality": 2.0, "w_latency": 3.0, "w_cost": 4.0}

    monkeypatch.setattr(rc, "settings", SimpleNamespace())
    weights = rc.get_dynamic_strategy_weights("text")
    assert weights == {"w_quality": 1.0, "w_latency": 0.5, "w_cost": 100.0}

    added = []
    monkeypatch.setattr(rc, "EMA_BATCH", SimpleNamespace(add=lambda modality, model, record: added.append((modality, model, record))))
    rc._persist_ema("text", "m1", {"ema_latency": 1})
    assert added == [("text", "m1", {"ema_latency": 1})]


def test_cleanup_helpers_run_single_iteration(monkeypatch):
    """Cleanup loops should execute their body once when the stop event is triggered by wait()."""
    waits = []

    class _Stop:
        def __init__(self):
            self.flag = False

        def is_set(self):
            return self.flag

        def set(self):
            self.flag = True

        def clear(self):
            self.flag = False

        def wait(self, timeout):
            waits.append(timeout)
            self.flag = True
            return True

    stop = _Stop()
    monkeypatch.setattr(rc, "_bg_stop_event", stop)

    ensured = []
    executed = []

    class _Conn:
        def execute(self, stmt, params=None):
            executed.append((str(stmt), params))
            return SimpleNamespace(rowcount=2)

    monkeypatch.setattr(rc, "_get_db_engine", lambda: SimpleNamespace(begin=lambda: _Ctx(_Conn())))
    monkeypatch.setattr(rc, "ensure_query_log", lambda: ensured.append(True))
    monkeypatch.setattr(rc, "LOG_RETENTION_DAYS", 9)
    monkeypatch.setattr(rc, "EMA_LOG_RETENTION_DAYS", 5)
    monkeypatch.setattr(rc, "update_db_pool_metrics", lambda: ensured.append("metrics"))
    monkeypatch.setattr(rc, "EMA_LOG_CLEANUP_ROWS", SimpleNamespace(inc=lambda value: ensured.append(("cleanup", value))))
    monkeypatch.setattr(rc, "EMA_HISTORY", SimpleNamespace(cleanup_expired=lambda: 3, size=lambda: 4))
    monkeypatch.setattr(rc, "ROUTER_HISTORY_ENTRIES", SimpleNamespace(set=lambda value: ensured.append(("history", value))))

    rc._cleanup_old_query_logs()
    stop.flag = False
    rc._cleanup_ema_history_log()
    stop.flag = False
    rc._update_db_pool_metrics()
    stop.flag = False
    rc._cleanup_ema_history()

    assert ensured
    assert executed
    assert waits


def test_start_stop_and_reset_runtime_state(monkeypatch):
    """Background service lifecycle and reset should preserve façade behavior."""
    calls = {"load": 0, "flush": 0}
    monkeypatch.setattr(rc, "_load_ema_from_db", lambda: calls.__setitem__("load", calls["load"] + 1))

    class _Thread:
        def __init__(self):
            self.started = False

        def start(self):
            self.started = True

        def join(self, timeout=None):
            return None

    threads = [_Thread(), _Thread()]
    monkeypatch.setattr(rc, "create_background_threads", lambda **kwargs: threads)
    monkeypatch.setattr(rc, "EMA_BATCH", SimpleNamespace(flush=lambda: calls.__setitem__("flush", calls["flush"] + 1)))
    monkeypatch.setattr(rc, "_bg_threads", [])
    monkeypatch.setattr(rc, "_bg_started", False)
    monkeypatch.setattr(rc, "_bg_stop_event", SimpleNamespace(set=lambda: None, clear=lambda: None))

    rc.start_background_services()
    assert calls["load"] == 1
    assert all(thread.started for thread in threads)

    rc.stop_background_services()
    assert calls["flush"] == 1
    assert rc._bg_started is False

    rc.EMA_HISTORY = "old"
    rc.EMA_BATCH = "old"
    rc._bg_threads = [object()]
    rc._bg_started = True
    stop_calls = []
    monkeypatch.setattr(rc, "_bg_stop_event", SimpleNamespace(set=lambda: stop_calls.append(True)))
    rc.reset_router_runtime_state()
    assert isinstance(rc.EMA_HISTORY, rc.EMAHistoryCache)
    assert isinstance(rc.EMA_BATCH, rc.EMABatchQueue)
    assert rc._bg_threads == []
    assert rc._bg_started is False
    assert stop_calls


def test_router_core_batch_and_maintenance_failure_paths(monkeypatch):
    """router_core maintenance helpers should handle SQL and runtime failures defensively."""
    warnings = []
    monkeypatch.setattr(rc, "logger", SimpleNamespace(info=lambda *a, **k: None, warning=lambda msg: warnings.append(msg), debug=lambda *a, **k: None))

    q = rc.EMABatchQueue(max_size=2, flush_interval=0)
    assert q._persist_batch([]) == 0

    class _Conn:
        def execute(self, stmt, params=None):
            raise rc.SQLAlchemyError("db fail")

    class _Engine:
        def begin(self):
            return _Ctx(_Conn())

    monkeypatch.setattr(rc, "_get_db_engine", lambda: _Engine())
    count = q._persist_batch([(("text", "m1"), {"ema_latency": 1, "ema_quality": 1, "ema_cost": 1, "updates": 10})])
    assert count == 0
    assert warnings

    waits = []

    class _Stop:
        def __init__(self):
            self.flag = False

        def is_set(self):
            return self.flag

        def wait(self, timeout):
            waits.append(timeout)
            self.flag = True
            return True

    stop = _Stop()
    monkeypatch.setattr(rc, "_bg_stop_event", stop)
    monkeypatch.setattr(rc, "EMA_BATCH", SimpleNamespace(should_flush=lambda: (_ for _ in ()).throw(RuntimeError("flush check")), flush=lambda: 0))
    rc._ema_batch_flusher()
    assert waits

    stop.flag = False
    monkeypatch.setattr(rc, "_get_db_engine", lambda: (_ for _ in ()).throw(RuntimeError("db fail")))
    rc._cleanup_ema_history_log()
    stop.flag = False
    monkeypatch.setattr(rc, "update_db_pool_metrics", lambda: (_ for _ in ()).throw(RuntimeError("pool fail")))
    rc._update_db_pool_metrics()
    stop.flag = False
    monkeypatch.setattr(rc, "EMA_HISTORY", SimpleNamespace(cleanup_expired=lambda: (_ for _ in ()).throw(RuntimeError("history fail")), size=lambda: 0))
    rc._cleanup_ema_history()


@pytest.mark.asyncio
async def test_route_and_answer_retry_exhaustion_and_last_error(monkeypatch):
    """route_and_answer should retry non-timeout exceptions and re-raise the last one."""
    monkeypatch.setattr(rc, "_route_and_answer_internal", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(rc, "_record_request_outcome", lambda success: None)
    monkeypatch.setattr(rc, "_safe_setting_int", lambda key, default: 1 if key == "REQUEST_MAX_RETRIES" else 5)
    monkeypatch.setattr(rc, "settings", SimpleNamespace(get=lambda key, default=None: "0"))

    async def _wait_for(awaitable, timeout=None):
        return await awaitable

    monkeypatch.setattr(rc.asyncio, "wait_for", _wait_for)
    monkeypatch.setattr(rc.asyncio, "sleep", lambda delay: _immediate())

    with pytest.raises(RuntimeError, match="boom"):
        await rc.route_and_answer("q", deduplicate=False)


async def _immediate():
    return None


@pytest.mark.asyncio
async def test_internal_and_feedback_wrappers_delegate(monkeypatch):
    """Public façade wrappers should pass the expected deps/state into extracted services."""
    internal_calls = {}

    async def _internal_impl(**kwargs):
        internal_calls.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(rc, "route_and_answer_internal_impl", _internal_impl)
    out = await rc._route_and_answer_internal("q", use_cache=False)
    assert out == {"ok": True}
    assert internal_calls["query"] == "q"
    assert "deps" in internal_calls and "call_model" in internal_calls["deps"]

    feedback_calls = {}

    async def _feedback_impl(**kwargs):
        feedback_calls.update(kwargs)
        return None

    monkeypatch.setattr(rc, "process_background_feedback_impl", _feedback_impl)
    await rc.process_background_feedback("q", "a", "m1", "text", 0.2, 0.01)
    assert feedback_calls["query"] == "q"
    assert feedback_calls["state"]["EMA_HISTORY"] is rc.EMA_HISTORY
    assert "deps" in feedback_calls and "judge_answer" in feedback_calls["deps"]


class _Ctx:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self.conn

    def __exit__(self, exc_type, exc, tb):
        return False
