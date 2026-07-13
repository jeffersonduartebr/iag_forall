# Objective: Test coverage for admin model management routes.
"""Tests for admin model candidate and health endpoints."""

from types import SimpleNamespace

import pytest
from app.api import admin_models_routes as routes
from app.api.admin_models_routes import ModelCandidatesUpdate
from fastapi import HTTPException


def _auth(monkeypatch):
    monkeypatch.setattr(routes, "resolve_admin_session", lambda **kwargs: {"username": "jefferson.silva"})


def _fake_settings(**kwargs):
    base = {
        "CANDIDATE_MODELS_LIST": [],
        "CANDIDATE_VISION_MODELS_LIST": [],
        "CANDIDATE_MULTIMODAL_MODELS_LIST": [],
    }
    base.update(kwargs)
    ns = SimpleNamespace(**base)
    ns.set = lambda key, value: base.__setitem__(key, value)
    return ns


def test_list_models(monkeypatch):
    """List models should merge registry and candidate lists."""
    _auth(monkeypatch)
    monkeypatch.setattr(
        routes,
        "settings",
        _fake_settings(CANDIDATE_MODELS_LIST=["ollama/phi4:latest"]),
    )
    monkeypatch.setattr(
        routes,
        "get_model_registry",
        lambda: SimpleNamespace(
            list_models=lambda configured_only=False: [
                SimpleNamespace(
                    full_name="ollama/phi4:latest",
                    display_name="phi4",
                    provider=SimpleNamespace(value="ollama"),
                    supports_vision=False,
                    priority=1,
                )
            ]
        ),
    )
    monkeypatch.setattr(routes, "is_model_configured", lambda name: True)
    monkeypatch.setattr(routes, "filter_configured_model_names", lambda names: list(names))

    out = routes.list_models()
    assert out["candidates"]["text"] == ["ollama/phi4:latest"]
    assert out["registered"][0]["full_name"] == "ollama/phi4:latest"


def test_update_model_candidates(monkeypatch):
    """PUT candidates should persist via settings.set."""
    _auth(monkeypatch)
    fake = _fake_settings()
    calls = []
    fake.set = lambda key, value: calls.append((key, value))
    monkeypatch.setattr(routes, "settings", fake)

    out = routes.update_model_candidates(ModelCandidatesUpdate(text=["openai/gpt-4o"]))
    assert out["status"] == "updated"
    assert ("CANDIDATE_MODELS_LIST", ["openai/gpt-4o"]) in calls


def test_models_health(monkeypatch):
    """Health endpoint should report circuit breaker state."""
    _auth(monkeypatch)
    monkeypatch.setattr(routes, "settings", _fake_settings(CANDIDATE_MODELS_LIST=["m1"]))
    monkeypatch.setattr(
        routes,
        "get_circuit_breaker_manager",
        lambda: SimpleNamespace(get_all_statuses=lambda: [{"model": "m1", "state": "closed", "fail_counter": 0}]),
    )
    monkeypatch.setattr(routes, "is_provider_temporarily_unavailable", lambda model: False)
    monkeypatch.setattr(routes, "is_model_configured", lambda model: True)

    out = routes.models_health()
    assert out["items"][0]["model"] == "m1"
    assert out["items"][0]["circuit_state"] == "closed"


def test_models_pricing_db_error(monkeypatch):
    """Pricing endpoint should return 503 when DB is unavailable."""
    _auth(monkeypatch)

    class _BrokenEngine:
        def connect(self):
            raise RuntimeError("db down")

    monkeypatch.setattr(routes, "get_engine", lambda: _BrokenEngine())
    with pytest.raises(HTTPException) as exc:
        routes.models_pricing()
    assert exc.value.status_code == 503


def test_models_requires_auth(monkeypatch):
    """Unauthorized access should be rejected."""

    def _deny(**kwargs):
        raise HTTPException(status_code=401, detail="Não autorizado.")

    monkeypatch.setattr(routes, "resolve_admin_session", _deny)
    with pytest.raises(HTTPException) as exc:
        routes.list_models()
    assert exc.value.status_code == 401


def test_update_openrouter_exploration(monkeypatch):
    """PUT exploration should persist runtime settings."""
    _auth(monkeypatch)
    calls = []
    fake = _fake_settings()
    fake.set = lambda key, value: calls.append((key, value))
    monkeypatch.setattr(routes, "settings", fake)

    from app.api.admin_models_routes import OpenRouterExplorationUpdate

    out = routes.update_openrouter_exploration(
        OpenRouterExplorationUpdate(enabled=True, rate=0.15, max_per_day=50),
    )
    assert out["status"] == "updated"
    assert ("OPENROUTER_EXPLORATION_ENABLED", "1") in calls
    assert ("OPENROUTER_EXPLORATION_RATE", "0.15") in calls
    assert ("OPENROUTER_EXPLORATION_MAX_PER_DAY", "50") in calls


def test_openrouter_credentials_roundtrip(monkeypatch):
    """Credentials endpoints should store and mask OpenRouter API key."""
    _auth(monkeypatch)
    store = {}
    fake = _fake_settings()

    def _set(key, value, actor="system", source="internal"):
        store[key] = value

    fake.set = _set
    monkeypatch.setattr(routes, "settings", fake)
    monkeypatch.setattr(routes, "get_openrouter_api_key", lambda: store.get("OPENROUTER_API_KEY", ""))
    monkeypatch.setattr(routes, "invalidate_openrouter_catalog_cache", lambda: None)
    from app.providers_async import ProviderFactory

    monkeypatch.setattr(ProviderFactory, "_instances", {})

    from app.api.admin_models_routes import OpenRouterCredentialsUpdate

    before = routes.get_openrouter_credentials()
    assert before["configured"] is False

    out = routes.update_openrouter_credentials(OpenRouterCredentialsUpdate(api_key="sk-or-v1-test-secret"))
    assert out["status"] == "updated"
    assert out["configured"] is True
    assert "..." in out["masked_key"]

    after = routes.get_openrouter_credentials()
    assert after["configured"] is True
    assert store["OPENROUTER_API_KEY"] == "sk-or-v1-test-secret"
