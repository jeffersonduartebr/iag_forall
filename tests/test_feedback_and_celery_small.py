# Objective: Test small API and Celery modules with low-cost branch coverage.
"""Small branch-coverage tests for feedback routes and celery configuration."""

from importlib import reload
from types import SimpleNamespace

import pytest
from app.api import feedback_routes as fr
from fastapi import HTTPException


def test_feedback_routes_success_and_errors(monkeypatch):
    """Feedback routes should map success, validation, and generic failures correctly."""
    monkeypatch.setattr(fr, "process_feedback", lambda request: SimpleNamespace(user_quality=8.0, blended_quality=7.5, model="ollama/x", reward=0.12345))
    out = fr.submit_feedback(object())
    assert out["status"] == "accepted"
    assert out["reward"] == 0.1235

    monkeypatch.setattr(fr, "process_feedback", lambda request: (_ for _ in ()).throw(ValueError("bad payload")))
    with pytest.raises(HTTPException) as exc:
        fr.submit_feedback(object())
    assert exc.value.status_code == 400

    monkeypatch.setattr(fr, "process_feedback", lambda request: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(HTTPException) as exc:
        fr.submit_feedback(object())
    assert exc.value.status_code == 500

    monkeypatch.setattr(fr, "get_feedback_stats", lambda model=None, hours=24: {"model": model, "hours": hours})
    monkeypatch.setattr(fr, "require_admin", lambda token: None)
    assert fr.feedback_stats(model="ollama/x", hours=12, x_admin_token="any") == {"model": "ollama/x", "hours": 12}


def test_celery_app_builds_urls_from_env(monkeypatch):
    """Celery configuration should build broker URLs with and without Redis auth."""
    import app.celery_app as ca

    monkeypatch.setenv("REDIS_HOST", "redis")
    monkeypatch.setenv("REDIS_PORT", "6379")
    monkeypatch.delenv("REDIS_PASSWORD", raising=False)
    reloaded = reload(ca)
    assert reloaded.BROKER_URL == "redis://redis:6379/0"
    assert reloaded.RESULT_BACKEND == "redis://redis:6379/1"

    monkeypatch.setenv("REDIS_PASSWORD", "secret")
    reloaded = reload(ca)
    assert reloaded.BROKER_URL == "redis://:secret@redis:6379/0"
    assert reloaded.RESULT_BACKEND == "redis://:secret@redis:6379/1"
