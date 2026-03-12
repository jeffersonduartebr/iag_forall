# Objective: Test coverage for reliability core extra behavior and regressions.
"""Test coverage for reliability core extra behavior and regressions.

This test module verifies expected behavior, regression boundaries, and failure
handling for the corresponding runtime component.
"""


from types import SimpleNamespace

import pybreaker
import pytest

from app import reliability as rel
from app.error_handling import ErrorCategory


def test_circuit_breaker_manager_config_status_reset(monkeypatch):
    """Testa circuit breaker manager config status reset."""
    rel.ModelCircuitBreakerManager._instance = None

    cfg = SimpleNamespace(circuit_breaker_threshold=2, circuit_breaker_timeout=7)
    monkeypatch.setattr(rel, "get_model_registry", lambda: SimpleNamespace(get=lambda name: cfg))

    mgr = rel.get_circuit_breaker_manager()
    br = mgr.get_breaker("openai/gpt-4o")
    assert br.fail_max == 2
    assert br.reset_timeout == 7
    st = mgr.get_status("openai/gpt-4o")
    assert st["state"] == "closed"

    assert mgr.reset_breaker("openai/gpt-4o") is True
    assert mgr.reset_breaker("missing/model") is False
    assert mgr.get_status("unknown/model")["state"] == "not_initialized"


@pytest.mark.asyncio
async def test_request_deduplicator_compute_cleanup_and_stats():
    """Testa request deduplicator compute cleanup and stats."""
    rel.RequestDeduplicator._instance = None
    d = rel.get_request_deduplicator()

    k1 = d._compute_key("q", "m", max_tokens=10, temperature=0.2, system_prompt="s")
    k2 = d._compute_key("q", "m", max_tokens=10, temperature=0.2, system_prompt="s")
    assert k1 == k2

    fut = rel.asyncio.get_event_loop().create_future()
    d._in_flight["old"] = rel.InFlightRequest(future=fut, created_at=0, model="m", query_hash="old")
    await d.cleanup_stale()
    assert "old" not in d._in_flight

    stats = d.get_stats()
    assert "in_flight_count" in stats


@pytest.mark.asyncio
async def test_execute_with_fallback_success_and_fail(monkeypatch):
    """Testa execute with fallback success and fail."""
    class _Breaker:
        """Represent `_Breaker` within this module.

The class groups the state and behavior required for Breaker."""
        async def call_async(self, fn, model):
            """Execute the call async routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            return await fn(model)

    class _Manager:
        """Represent `_Manager` within this module.

The class groups the state and behavior required for Manager."""
        def __init__(self):
            """Initialize the internal state required by this instance.

The constructor keeps setup local to the object so callers can use it without additional bootstrapping."""
            self.avail = {"primary/m": False, "fb/ok": True, "fb/fail": True}

        def is_available(self, model):
            """Return whether available is true for the current input or runtime state."""
            return self.avail.get(model, True)

        def get_breaker(self, model):
            """Return breaker.

This helper centralizes retrieval logic so callers do not have to duplicate lookup behavior."""
            return _Breaker()

    monkeypatch.setattr(rel, "get_circuit_breaker_manager", lambda: _Manager())
    monkeypatch.setattr(
        rel,
        "get_model_registry",
        lambda: SimpleNamespace(get_fallback_chain=lambda m, max_depth=3: [SimpleNamespace(full_name="fb/ok")]),
    )

    async def _exec_ok(model):
        """Execute the exec ok routine.

This helper encapsulates one focused step used by the surrounding workflow."""
        return f"ok-{model}"

    r1 = await rel.execute_with_fallback("primary/m", _exec_ok, max_fallbacks=2)
    assert r1.success is True
    assert r1.model_used == "fb/ok"
    assert r1.errors and r1.errors[0]["category"] == ErrorCategory.CIRCUIT_OPEN.value

    monkeypatch.setattr(
        rel,
        "get_model_registry",
        lambda: SimpleNamespace(get_fallback_chain=lambda m, max_depth=3: [SimpleNamespace(full_name="fb/fail")]),
    )
    monkeypatch.setattr(
        rel,
        "log_provider_error",
        lambda exc, model, operation=None: SimpleNamespace(category=ErrorCategory.PROVIDER_UNAVAILABLE),
    )

    async def _exec_fail(model):
        """Execute the exec fail routine.

This helper encapsulates one focused step used by the surrounding workflow."""
        raise RuntimeError("boom")

    r2 = await rel.execute_with_fallback("primary/m", _exec_fail, max_fallbacks=2)
    assert r2.success is False
    assert r2.errors


@pytest.mark.asyncio
async def test_model_health_helpers(monkeypatch):
    """Testa model health helpers."""
    mgr = SimpleNamespace(
        get_status=lambda model: {"model": model, "state": "closed"},
        is_available=lambda model: model != "m2",
        get_all_statuses=lambda: [{"model": "m1", "state": "closed"}, {"model": "m2", "state": "open"}],
    )
    reg = SimpleNamespace(
        get=lambda m: SimpleNamespace(provider=SimpleNamespace(value="openai"), default_timeout=60, fallback_models=["x"]),
        list_models=lambda: [SimpleNamespace(full_name="m1"), SimpleNamespace(full_name="m2")],
    )
    monkeypatch.setattr(rel, "get_circuit_breaker_manager", lambda: mgr)
    monkeypatch.setattr(rel, "get_model_registry", lambda: reg)

    h = await rel.check_model_health("m1")
    assert h["available"] is True
    all_h = await rel.check_all_models_health()
    assert all_h["total_models"] == 2
    assert all_h["unavailable_models"] == 1


def test_cascade_detector_status_and_warnings(monkeypatch):
    """Testa cascade detector status and warnings."""
    rel.CascadeDetector._instance = None

    mgr = SimpleNamespace(
        get_all_statuses=lambda: [
            {"model": "a", "state": "open"},
            {"model": "b", "state": "open"},
            {"model": "c", "state": "open"},
            {"model": "d", "state": "closed"},
        ],
        is_available=lambda model: model == "ollama/phi4:latest",
    )
    monkeypatch.setattr(rel, "get_circuit_breaker_manager", lambda: mgr)

    d = rel.get_cascade_detector()
    ratio, failed, total = d.get_failed_model_ratio()
    assert ratio == pytest.approx(0.75)
    assert failed == 3 and total == 4
    assert d.get_severity_name() == "critical"
    assert d.is_degraded is True
    assert d.is_emergency_mode is False

    status = d.get_status()
    assert status["severity_name"] == "critical"
    warns = d.check_and_log_warnings()
    assert warns["warnings"]

    # emergency branch + fallback selection
    mgr2 = SimpleNamespace(
        get_all_statuses=lambda: [{"model": "a", "state": "open"} for _ in range(10)],
        is_available=lambda model: model == "ollama/phi4:latest",
    )
    monkeypatch.setattr(rel, "get_circuit_breaker_manager", lambda: mgr2)
    assert d.get_severity_name() == "emergency"
    assert d.get_emergency_fallback() == "ollama/phi4:latest"
