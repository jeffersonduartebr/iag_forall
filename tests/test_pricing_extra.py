# Objective: Test coverage for pricing extra behavior and regressions.
"""Test coverage for pricing extra behavior and regressions.

This test module verifies expected behavior, regression boundaries, and failure
handling for the corresponding runtime component.
"""


import json
import time
from types import SimpleNamespace

from app.utils import pricing as pr


class _Conn:
    """Represent `_Conn` within this module.

The class groups the state and behavior required for Conn."""
    def __init__(self, rows):
        """Initialize the internal state required by this instance.

The constructor keeps setup local to the object so callers can use it without additional bootstrapping."""
        self.rows = rows

    def execute(self, *_a, **_k):
        """Execute the execute routine.

This helper encapsulates one focused step used by the surrounding workflow."""
        return SimpleNamespace(fetchall=lambda: self.rows)


class _Ctx:
    """Represent `_Ctx` within this module.

The class groups the state and behavior required for Ctx."""
    def __init__(self, rows):
        """Initialize the internal state required by this instance.

The constructor keeps setup local to the object so callers can use it without additional bootstrapping."""
        self.rows = rows

    def __enter__(self):
        """Execute the enter routine.

This helper encapsulates one focused step used by the surrounding workflow."""
        return _Conn(self.rows)

    def __exit__(self, exc_type, exc, tb):
        """Execute the exit routine.

This helper encapsulates one focused step used by the surrounding workflow."""
        return False


def test_refresh_pricing_from_db_success_and_error(monkeypatch):
    """Testa refresh pricing from db success and error."""
    monkeypatch.setattr(pr, "get_engine", lambda: SimpleNamespace(connect=lambda: _Ctx([("m1", 0.1, 0.2)])))
    out = pr._refresh_pricing_from_db()
    assert out["m1"]["in"] == 0.1

    monkeypatch.setattr(pr, "get_engine", lambda: (_ for _ in ()).throw(RuntimeError("db")))
    assert pr._refresh_pricing_from_db() == {}


def test_refresh_pricing_redis_hit_and_fallback(monkeypatch):
    """Testa refresh pricing redis hit and fallback."""
    class _R:
        """Represent `_R` within this module.

The class groups the state and behavior required for R."""
        def __init__(self, payload=None):
            """Initialize the internal state required by this instance.

The constructor keeps setup local to the object so callers can use it without additional bootstrapping."""
            self.payload = payload
            self.writes = []

        def get(self, key):
            """Execute the get routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            return self.payload

        def setex(self, key, ttl, value):
            """Execute the setex routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            self.writes.append((key, ttl, value))

        def delete(self, key):
            """Execute the delete routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            self.writes.append(("del", key))

    hit = _R(payload=json.dumps({"x": {"in": 1.0, "out": 2.0}}))
    monkeypatch.setattr(pr, "_get_rds", lambda: hit)
    pr._PRICING_CACHE = {}
    pr._LAST_UPDATE = 0
    pr._refresh_pricing()
    assert pr._PRICING_CACHE["x"]["in"] == 1.0

    miss = _R(payload=None)
    monkeypatch.setattr(pr, "_get_rds", lambda: miss)
    monkeypatch.setattr(pr, "_refresh_pricing_from_db", lambda: {"dbm": {"in": 0.5, "out": 0.6}})
    pr._refresh_pricing()
    assert pr._PRICING_CACHE["dbm"]["out"] == 0.6
    assert miss.writes

    pr.invalidate_pricing_cache()
    assert pr._PRICING_CACHE == {}
    assert pr._LAST_UPDATE == 0


def test_get_model_cost_paths(monkeypatch):
    """Testa get model cost paths."""
    pr._LAST_UPDATE = time.time()
    pr._PRICING_CACHE = {"gpt-4o": {"in": 1.0, "out": 2.0}, "mini": {"in": 0.5, "out": 0.5}}

    c1 = pr.get_model_cost("gpt-4o", 1000, 1000)
    assert c1 == 3.0

    c2 = pr.get_model_cost("openai/mini", 1000, 1000)
    assert c2 == 1.0

    # fallback table branches
    assert pr.get_model_cost("gpt-5-mini", 1000, 1000) > 0
    assert pr.get_model_cost("gpt-5", 1000, 1000) > 0
    assert pr.get_model_cost("gpt-4.1-mini", 1000, 1000) > 0
    assert pr.get_model_cost("gpt-4o-mini", 1000, 1000) > 0
    assert pr.get_model_cost("gpt-4o", 1000, 1000) > 0
    assert pr.get_model_cost("gemini-2.5-flash", 1000, 1000) > 0
    assert pr.get_model_cost("gemini-2.5", 1000, 1000) > 0
    assert pr.get_model_cost("claude-haiku", 1000, 1000) > 0
    assert pr.get_model_cost("claude-sonnet", 1000, 1000) > 0
    assert pr.get_model_cost("claude-opus", 1000, 1000) > 0
    assert pr.get_model_cost("ollama/local", 1000, 1000) == 0.0
    pr.invalidate_pricing_cache()
