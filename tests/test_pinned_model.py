# Objective: Test coverage for pinned_model (measurement-instrument calls that bypass the policy under study).
"""``pinned_model``: only instrument/admin callers, exactly that model, and nothing flows back into learning."""

from types import SimpleNamespace

import pytest
from app.api.auth import AuthContext
from app.schemas import QueryRequest
from app.services import query_profile as qp
from app.services import query_runtime as qr
from app.services.query_http import _real_stream_eligible
from app.services.router_execution import _choose_route
from app.services.router_stages import RouteContext
from app.services.tenant_context import bind_tenant_to_request
from fastapi import HTTPException


def _req(**kw):
    return QueryRequest(query="Avalie a aderência à rubrica.", **kw)


@pytest.mark.parametrize("roles", [["instrument"], ["admin"]])
def test_pinned_model_allowed_for_instrument_and_admin(roles):
    auth = AuthContext(authenticated=True, method="jwt", tenant_id="t", roles=roles)
    assert bind_tenant_to_request(_req(pinned_model="openai/gpt-x"), auth).tenant_id == "t"


@pytest.mark.parametrize("auth", [None, AuthContext(authenticated=True, method="api_key"),
                                  AuthContext(authenticated=False, roles=["instrument"])])
def test_pinned_model_forbidden_otherwise(auth):
    with pytest.raises(HTTPException) as exc:
        bind_tenant_to_request(_req(pinned_model="openai/gpt-x"), auth)
    assert exc.value.status_code == 403
    assert bind_tenant_to_request(_req(), auth) is not None


class _ProviderCallError(Exception):
    def __init__(self, model, message, category, retryable):
        super().__init__(message)
        self.model, self.category = model, category


def _ctx(hints, models=("openai/gpt-x", "ollama/qwen")):
    settings = SimpleNamespace(
        CANDIDATE_MODELS_LIST=list(models), CANDIDATE_VISION_MODELS_LIST=[], CANDIDATE_MULTIMODAL_MODELS_LIST=[]
    )
    deps = {"settings": settings, "ProviderCallError": _ProviderCallError, "logger": SimpleNamespace(info=print),
            "BLOCKED_PREFIXES": ("blocked/",)}
    return RouteContext(
        deps=deps, query="q", system_prompt="", use_rag=False, max_tokens=64, temperature=0.1, modality="text",
        image_b64=None, rag_modality="text", use_cache=False, runtime_hints=hints,
    )


@pytest.mark.asyncio
async def test_pinned_choice_skips_selection_and_disables_hedging(monkeypatch):
    import app.services.router_stages as rs

    monkeypatch.setattr(rs, "filter_configured_model_names", list)
    choice = await _choose_route(_ctx({"pinned_model": "openai/gpt-x"}), 0.9)
    assert (choice.chosen, choice.top2, choice.exploration_mode) == ("openai/gpt-x", ["openai/gpt-x"], False)


@pytest.mark.asyncio
async def test_pinned_model_outside_configured_candidates_is_rejected(monkeypatch):
    import app.services.router_stages as rs

    monkeypatch.setattr(rs, "filter_configured_model_names", list)
    with pytest.raises(_ProviderCallError) as exc:
        await _choose_route(_ctx({"pinned_model": "anthropic/nope"}), 0.1)
    assert exc.value.category == "pinned_model_unavailable"


def test_pinned_unavailable_maps_to_422():
    err = qr.ProviderCallError("anthropic/nope", "x", category="pinned_model_unavailable")
    http = qr._routing_http_error(err, "text")
    assert (http.status_code, http.detail["category"]) == (422, "pinned_model_unavailable")


def test_profile_hints_pin_the_model_without_fallbacks(monkeypatch):
    monkeypatch.setattr(qp.settings, "get", lambda key, fallback=None: fallback)
    hints = qp.apply_query_runtime_profile(_req(pinned_model="openai/gpt-x"), "text", None)["runtime_hints"]
    assert (hints["pinned_model"], hints["max_fallbacks"]) == ("openai/gpt-x", 0)
    assert "pinned_model" not in qp.apply_query_runtime_profile(_req(), "text", None)["runtime_hints"]


def test_pinned_call_never_streams_through_the_policy():
    assert _real_stream_eligible(_req(pinned_model="openai/gpt-x")) is False


def test_pinned_call_sends_no_learning_feedback_but_still_bills(monkeypatch):
    sent, usage = [], []
    monkeypatch.setattr(qr.task_process_feedback, "delay", lambda **kw: sent.append(kw))
    monkeypatch.setattr(qr, "record_tool_turn", lambda **kw: sent.append(kw))
    monkeypatch.setattr(qr, "record_tenant_usage", lambda **kw: usage.append(kw))
    result = {"answer": "{}", "model": "openai/gpt-x", "modality": "text", "latency_s": 0.1,
              "estimated_cost_usd": 0.002, "metadata": {"prompt_tokens": 5, "completion_tokens": 2}}
    qr.record_query_side_effects(_req(pinned_model="openai/gpt-x", tenant_id="t"), result, None)
    assert sent == []
    assert usage[0]["tenant_id"] == "t"


@pytest.mark.asyncio
async def test_pinned_and_scoped_calls_bypass_semantic_cache(monkeypatch):
    seen = {}

    async def _route(**kwargs):
        seen["use_cache"] = kwargs["use_cache"]
        return {}

    monkeypatch.setattr(qr, "route_and_answer", _route)
    await qr._route(_req(pinned_model="openai/gpt-x"), "text", None, {"use_rag": False, "max_tokens": 64, "runtime_hints": {}})
    assert seen["use_cache"] is False
