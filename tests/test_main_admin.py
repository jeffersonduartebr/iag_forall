"""Módulo `tests/test_main_admin.py`: descreve responsabilidades e integrações deste arquivo."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException


def _set_admin_token(monkeypatch, value="secret-token"):
    """Executa set admin token."""
    from app import main

    monkeypatch.setattr(
        main.settings,
        "get",
        lambda key, fallback=None: value if key == "ADMIN_TOKEN" else fallback,
    )


def _set_settings_map(monkeypatch, mapping):
    """Executa set settings map."""
    from app import main

    monkeypatch.setattr(main.settings, "get", lambda key, fallback=None: mapping.get(key, fallback))


def test_safe_parse_json_variants():
    """Testa safe parse json variants."""
    from app.main import safe_parse_json

    assert safe_parse_json('{"a":1}') == {"a": 1}
    assert safe_parse_json("not-json") == "not-json"
    assert safe_parse_json({"x": 1}) == {"x": 1}


def test_require_admin(monkeypatch):
    """Testa require admin."""
    from app.main import _require_admin

    _set_admin_token(monkeypatch, "abc")
    _require_admin("abc")

    with pytest.raises(HTTPException) as exc:
        _require_admin("wrong")
    assert exc.value.status_code == 401


def test_admin_settings_endpoints(monkeypatch):
    """Testa admin settings endpoints."""
    from app import main

    _set_admin_token(monkeypatch, "abc")
    monkeypatch.setattr(main.settings, "snapshot", lambda: {"ok": True})

    calls = []

    def _set(k, v, actor=None, source=None):
        """Executa set."""
        calls.append((k, v, actor, source))

    monkeypatch.setattr(main.settings, "set", _set)

    got = main.get_settings("abc")
    assert got == {"ok": True}

    out = main.update_settings({"a": 1, "b": {"k": "v"}}, "abc")
    assert out["status"] == "updated"
    assert calls[0][0] == "a" and calls[0][1] == "1"
    assert calls[1][0] == "b" and calls[1][1] == '{"k": "v"}'


def test_circuit_breaker_admin(monkeypatch):
    """Testa circuit breaker admin."""
    from app import main

    _set_admin_token(monkeypatch, "abc")
    manager = SimpleNamespace(
        get_all_statuses=lambda: [{"model": "m1", "state": "closed"}],
        reset_breaker=lambda model: model == "m1",
    )
    monkeypatch.setattr(main, "get_circuit_breaker_manager", lambda: manager)

    status = main.get_circuit_breakers("abc")
    assert "circuit_breakers" in status
    assert status["circuit_breakers"][0]["model"] == "m1"

    ok = main.reset_circuit_breaker("m1", "abc")
    assert ok["status"] == "reset"

    with pytest.raises(HTTPException) as exc:
        main.reset_circuit_breaker("missing", "abc")
    assert exc.value.status_code == 404


def test_cascade_status_admin(monkeypatch):
    """Testa cascade status admin."""
    from app import main

    _set_admin_token(monkeypatch, "abc")
    monkeypatch.setattr(main, "get_cascade_detector", lambda: SimpleNamespace(get_status=lambda: {"status": "ok"}))
    out = main.get_cascade_status("abc")
    assert out["status"] == "ok"


def test_feedback_endpoints(monkeypatch):
    """Testa feedback endpoints."""
    from app import main

    result = SimpleNamespace(user_quality=8.0, blended_quality=7.2, model="m1", reward=0.73)
    monkeypatch.setattr(main, "process_feedback", lambda req: result)
    monkeypatch.setattr(main, "get_feedback_stats", lambda model=None, hours=24: {"hours": hours, "model": model})

    out = main.submit_feedback(SimpleNamespace())
    assert out["status"] == "accepted"
    assert out["model"] == "m1"

    monkeypatch.setattr(main, "process_feedback", lambda req: (_ for _ in ()).throw(ValueError("bad")))
    with pytest.raises(HTTPException) as exc:
        main.submit_feedback(SimpleNamespace())
    assert exc.value.status_code == 400

    stats = main.feedback_stats(model="m1", hours=12)
    assert stats["hours"] == 12


def test_experiment_admin_endpoints(monkeypatch):
    """Testa experiment admin endpoints."""
    from app import main

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
    monkeypatch.setattr(main, "get_ab_test_manager", lambda: manager)

    listed = main.list_experiments(status=None, x_admin_token="abc")
    assert listed["total"] == 1

    created = main.create_experiment(SimpleNamespace(), "abc")
    assert created["status"] == "created"

    got = main.get_experiment("e1", "abc")
    assert got["id"] == "e1"

    with pytest.raises(HTTPException) as exc:
        main.get_experiment("missing", "abc")
    assert exc.value.status_code == 404

    assert main.start_experiment("e1", "abc")["status"] == "started"
    assert main.pause_experiment("e1", "abc")["status"] == "paused"
    assert main.complete_experiment("e1", "abc")["status"] == "completed"
    assert main.get_experiment_results("e1", "abc")["id"] == "e1"
    assert main.delete_experiment("e1", "abc")["status"] == "deleted"

    with pytest.raises(HTTPException) as exc:
        main.start_experiment("missing", "abc")
    assert exc.value.status_code == 404

    with pytest.raises(HTTPException) as exc:
        main.delete_experiment("missing", "abc")
    assert exc.value.status_code == 404


def test_experiments_disabled_paths(monkeypatch):
    """Testa experiments disabled paths."""
    from app import main

    _set_settings_map(monkeypatch, {"ADMIN_TOKEN": "abc", "AB_TESTING_ENABLED": False})

    out = main.list_experiments(status=None, x_admin_token="abc")
    assert out["error"] == "A/B testing is disabled"

    with pytest.raises(HTTPException) as exc:
        main.create_experiment(SimpleNamespace(), "abc")
    assert exc.value.status_code == 400
