"""Focused tests for router resilience helpers."""

from __future__ import annotations

from types import SimpleNamespace

from app.services import router_resilience as rr


class _MetricLabel:
    def __init__(self):
        self.values = []
        self.inc_calls = 0

    def set(self, value):
        self.values.append(value)

    def inc(self, value=1):
        self.inc_calls += value


def test_safe_setting_helpers_and_window_clamping():
    """Coercion helpers should parse values and fall back safely."""
    getter = lambda key, default=None: {"I": "7", "F": "1.5", "B": "true", "ERROR_BUDGET_WINDOW_S": "5", "ERROR_BUDGET_THRESHOLD": "4.0", "ERROR_BUDGET_MIN_REQUESTS": "0"}.get(key, default)
    assert rr.safe_setting_int(getter, "I", 2) == 7
    assert rr.safe_setting_int(lambda *_: (_ for _ in ()).throw(ValueError()), "I", 2) == 2
    assert rr.safe_setting_float(getter, "F", 2.0) == 1.5
    assert rr.safe_setting_bool(getter, "B", False) is True
    assert rr.error_budget_window(getter) == (10, 1.0, 1)


def test_record_dependency_breaker_metrics(monkeypatch):
    """Breaker states should map to stable metric values."""
    cache_metric = _MetricLabel()
    uq_metric = _MetricLabel()
    monkeypatch.setattr(rr, "DEPENDENCY_CIRCUIT_STATE", SimpleNamespace(labels=lambda dependency: cache_metric if dependency == "cache" else uq_metric))
    monkeypatch.setattr(rr, "dep_cache_breaker", SimpleNamespace(current_state="closed"))
    monkeypatch.setattr(rr, "dep_uq_breaker", SimpleNamespace(current_state="open"))

    rr.record_dependency_breaker_metrics()
    assert cache_metric.values[-1] == 0
    assert uq_metric.values[-1] == 2


def test_record_request_outcome_and_exceeded(monkeypatch):
    """Error budget should write counters and detect exceeded thresholds."""
    store = {}

    class _Pipe:
        def __init__(self):
            self.key = None

        def hincrby(self, key, field, value):
            self.key = key
            bucket = store.setdefault(key, {"total": 0, "errors": 0})
            bucket[field] += value

        def expire(self, key, ttl):
            return None

        def execute(self):
            return None

    class _Redis:
        def pipeline(self):
            return _Pipe()

        def hgetall(self, key):
            return store.get(key, {})

    monkeypatch.setattr(rr, "get_router_redis", lambda: _Redis())
    monkeypatch.setattr(rr.time, "time", lambda: 100.0)
    getter = lambda key, default=None: {
        "ERROR_BUDGET_ENABLED": "1",
        "ERROR_BUDGET_WINDOW_S": "30",
        "ERROR_BUDGET_THRESHOLD": "0.5",
        "ERROR_BUDGET_MIN_REQUESTS": "2",
    }.get(key, default)

    rr.record_request_outcome(settings_getter=getter, success=False)
    rr.record_request_outcome(settings_getter=getter, success=True)
    assert rr.is_error_budget_exceeded(settings_getter=getter) is True


def test_error_budget_handles_disabled_and_exceptions(monkeypatch):
    """Error budget helpers should short-circuit when disabled and count failures on read errors."""
    failure_metric = _MetricLabel()
    monkeypatch.setattr(rr, "DEPENDENCY_FAILURES", SimpleNamespace(labels=lambda dependency: failure_metric))
    monkeypatch.setattr(rr, "get_router_redis", lambda: None)
    disabled = lambda key, default=None: {"ERROR_BUDGET_ENABLED": "0"}.get(key, default)
    assert rr.is_error_budget_exceeded(settings_getter=disabled) is False

    class _BrokenRedis:
        def hgetall(self, key):
            raise RuntimeError("redis down")

    monkeypatch.setattr(rr, "get_router_redis", lambda: _BrokenRedis())
    enabled = lambda key, default=None: {"ERROR_BUDGET_ENABLED": "1"}.get(key, default)
    assert rr.is_error_budget_exceeded(settings_getter=enabled) is False
    assert failure_metric.inc_calls == 1
