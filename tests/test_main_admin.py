# Objective: Test coverage for main admin behavior and regressions.
"""Tests for admin and feedback HTTP handlers after router extraction."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException


def _set_admin_token(monkeypatch, value="secret-token"):
    """Configure admin token for route-level auth tests."""
    from app.api import admin_routes, deps

    monkeypatch.setattr(
        admin_routes.settings,
        "get",
        lambda key, fallback=None: value if key == "ADMIN_TOKEN" else fallback,
    )
    monkeypatch.setattr(
        deps.settings,
        "get",
        lambda key, fallback=None: value if key == "ADMIN_TOKEN" else fallback,
    )


def _set_settings_map(monkeypatch, mapping):
    """Configure multiple dynamic settings for admin route tests."""
    from app.api import admin_routes, deps

    monkeypatch.setattr(admin_routes.settings, "get", lambda key, fallback=None: mapping.get(key, fallback))
    monkeypatch.setattr(deps.settings, "get", lambda key, fallback=None: mapping.get(key, fallback))


def test_safe_parse_json_variants():
    """safe_parse_json should preserve dicts and decode JSON strings."""
    from app.services.query_response_builder import safe_parse_json

    assert safe_parse_json('{"a":1}') == {"a": 1}
    assert safe_parse_json("not-json") == "not-json"
    assert safe_parse_json({"x": 1}) == {"x": 1}


def test_require_admin(monkeypatch):
    """Shared require_admin helper should still be exposed by main."""
    from app.main import _require_admin

    _set_admin_token(monkeypatch, "abc")
    _require_admin("abc")

    with pytest.raises(HTTPException) as exc:
        _require_admin("wrong")
    assert exc.value.status_code == 401


def test_admin_settings_endpoints(monkeypatch):
    """Settings endpoints live in admin_routes and preserve payload behavior."""
    from app.api import admin_routes
    from app.schemas import AdminSettingsUpdateRequest

    _set_admin_token(monkeypatch, "abc")
    monkeypatch.setattr(admin_routes.settings, "snapshot", lambda: {"ok": True})
    monkeypatch.setattr(admin_routes.settings, "keys", lambda domain=None: ["MAX_TOKENS_DEFAULT"])
    monkeypatch.setattr(admin_routes.settings, "metadata", lambda key: {"mutability": "runtime_safe", "domain": "runtime"})
    monkeypatch.setattr(
        admin_routes.settings,
        "validate_runtime_updates",
        lambda payload: {"runtime_safe": list(payload.keys()), "requires_restart": [], "unknown": []},
    )

    calls = []

    def _set(k, v, actor=None, source=None):
        calls.append((k, v, actor, source))

    monkeypatch.setattr(admin_routes.settings, "set", _set)

    got = admin_routes.get_settings("abc")
    assert got == {"ok": True}
    catalog = admin_routes.get_settings_catalog("abc")
    assert catalog["settings"]["MAX_TOKENS_DEFAULT"]["mutability"] == "runtime_safe"

    out = admin_routes.update_settings(AdminSettingsUpdateRequest(settings={"a": 1, "b": {"k": "v"}}), "abc")
    assert out["status"] == "updated"
    assert out["applied"] == ["a", "b"]
    assert calls[0] == ("a", "1", "api", "admin")
    assert calls[1] == ("b", '{"k": "v"}', "api", "admin")


def test_admin_settings_reject_unknown_or_restart_required(monkeypatch):
    """Admin settings updates should reject unknown or restart-only keys."""
    from app.api import admin_routes
    from app.schemas import AdminSettingsUpdateRequest

    _set_admin_token(monkeypatch, "abc")

    monkeypatch.setattr(
        admin_routes.settings,
        "validate_runtime_updates",
        lambda payload: {"runtime_safe": [], "requires_restart": ["OLLAMA_HOST"], "unknown": []},
    )
    with pytest.raises(HTTPException) as exc:
        admin_routes.update_settings(AdminSettingsUpdateRequest(settings={"OLLAMA_HOST": "http://x"}), "abc")
    assert exc.value.status_code == 409

    monkeypatch.setattr(
        admin_routes.settings,
        "validate_runtime_updates",
        lambda payload: {"runtime_safe": [], "requires_restart": [], "unknown": ["CUSTOM_X"]},
    )
    with pytest.raises(HTTPException) as exc:
        admin_routes.update_settings(AdminSettingsUpdateRequest(settings={"CUSTOM_X": "1"}), "abc")
    assert exc.value.status_code == 400


def test_circuit_breaker_admin(monkeypatch):
    """Circuit breaker endpoints use the reliability manager directly."""
    from app.api import admin_routes

    _set_admin_token(monkeypatch, "abc")
    manager = SimpleNamespace(
        get_all_statuses=lambda: [{"model": "m1", "state": "closed"}],
        reset_breaker=lambda model: model == "m1",
    )
    monkeypatch.setattr(admin_routes, "get_circuit_breaker_manager", lambda: manager)

    status = admin_routes.get_circuit_breakers("abc")
    assert status["circuit_breakers"][0]["model"] == "m1"

    ok = admin_routes.reset_circuit_breaker("m1", "abc")
    assert ok["status"] == "reset"

    with pytest.raises(HTTPException) as exc:
        admin_routes.reset_circuit_breaker("missing", "abc")
    assert exc.value.status_code == 404


def test_cascade_status_admin(monkeypatch):
    """Cascade status endpoint delegates to the detector."""
    from app.api import admin_routes

    _set_admin_token(monkeypatch, "abc")
    monkeypatch.setattr(admin_routes, "get_cascade_detector", lambda: SimpleNamespace(get_status=lambda: {"status": "ok"}))
    out = admin_routes.get_cascade_status("abc")
    assert out["status"] == "ok"


def test_runtime_reset_admin(monkeypatch):
    """Runtime reset endpoint should trigger the reset hook."""
    from app.api import admin_routes

    _set_admin_token(monkeypatch, "abc")
    seen = {"called": False}
    monkeypatch.setattr(admin_routes, "reset_runtime_state", lambda: seen.__setitem__("called", True))
    out = admin_routes.reset_runtime("abc")
    assert out == {"status": "reset"}
    assert seen["called"] is True


def test_feedback_endpoints(monkeypatch):
    """Feedback routes preserve success and validation failures."""
    from app.api import feedback_routes

    result = SimpleNamespace(user_quality=8.0, blended_quality=7.2, model="m1", reward=0.73)
    monkeypatch.setattr(feedback_routes, "process_feedback", lambda req: result)
    monkeypatch.setattr(feedback_routes, "get_feedback_stats", lambda model=None, hours=24: {"hours": hours, "model": model})
    monkeypatch.setattr(feedback_routes, "require_admin", lambda token: None)

    out = feedback_routes.submit_feedback(SimpleNamespace())
    assert out["status"] == "accepted"
    assert out["model"] == "m1"

    monkeypatch.setattr(feedback_routes, "process_feedback", lambda req: (_ for _ in ()).throw(ValueError("bad")))
    with pytest.raises(HTTPException) as exc:
        feedback_routes.submit_feedback(SimpleNamespace())
    assert exc.value.status_code == 400

    stats = feedback_routes.feedback_stats(model="m1", hours=12, x_admin_token="abc")
    assert stats["hours"] == 12


def test_experiment_admin_endpoints(monkeypatch):
    """Experiment endpoints remain in admin_routes."""
    from app.api import admin_routes

    _set_settings_map(monkeypatch, {"ADMIN_TOKEN": "abc", "AB_TESTING_ENABLED": True})

    exp = SimpleNamespace(id="e1", to_dict=lambda: {"id": "e1"})
    manager = SimpleNamespace(
        list_experiments=lambda status=None: [exp],
        create_experiment=lambda request: exp,
        get_experiment=lambda eid: exp if eid == "e1" else None,
        start_experiment=lambda eid: exp if eid == "e1" else (_ for _ in ()).throw(KeyError(eid)),
        pause_experiment=lambda eid: exp if eid == "e1" else (_ for _ in ()).throw(KeyError(eid)),
        complete_experiment=lambda eid: exp if eid == "e1" else (_ for _ in ()).throw(KeyError(eid)),
        get_experiment_results=lambda eid: {"id": eid, "results": []},
        delete_experiment=lambda eid: eid == "e1",
    )
    monkeypatch.setattr(admin_routes, "get_ab_test_manager", lambda: manager)

    listed = admin_routes.list_experiments(status=None, x_admin_token="abc")
    assert listed["total"] == 1

    created = admin_routes.create_experiment(SimpleNamespace(), "abc")
    assert created["status"] == "created"

    got = admin_routes.get_experiment("e1", "abc")
    assert got["id"] == "e1"

    with pytest.raises(HTTPException) as exc:
        admin_routes.get_experiment("missing", "abc")
    assert exc.value.status_code == 404

    assert admin_routes.start_experiment("e1", "abc")["status"] == "start"
    assert admin_routes.pause_experiment("e1", "abc")["status"] == "pause"
    assert admin_routes.complete_experiment("e1", "abc")["status"] == "complete"
    assert admin_routes.get_experiment_results("e1", "abc")["id"] == "e1"
    assert admin_routes.delete_experiment("e1", "abc")["status"] == "deleted"

    with pytest.raises(HTTPException) as exc:
        admin_routes.start_experiment("missing", "abc")
    assert exc.value.status_code == 404

    with pytest.raises(HTTPException) as exc:
        admin_routes.delete_experiment("missing", "abc")
    assert exc.value.status_code == 404


def test_experiments_disabled_paths(monkeypatch):
    """Disabled A/B setting should short-circuit experiment endpoints."""
    from app.api import admin_routes

    _set_settings_map(monkeypatch, {"ADMIN_TOKEN": "abc", "AB_TESTING_ENABLED": False})

    out = admin_routes.list_experiments(status=None, x_admin_token="abc")
    assert out["error"] == "A/B testing is disabled"

    with pytest.raises(HTTPException) as exc:
        admin_routes.create_experiment(SimpleNamespace(), "abc")
    assert exc.value.status_code == 400
