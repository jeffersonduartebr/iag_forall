# -*- coding: utf-8 -*-
"""Testes da governança de tool calling por policy/tenant (roadmap item #11)."""

from __future__ import annotations

import pytest
from app.services import tool_governance as tg


def _tool(name: str):
    return {"type": "function", "function": {"name": name, "parameters": {}}}


def _policy(gov: dict):
    return {"version": "v1", "config": {"tool_governance": gov}}


def test_no_tools_is_allowed():
    assert tg.evaluate_tool_policy(None, _policy({"allowed_tools": []})).allowed is True
    assert tg.evaluate_tool_policy([], _policy({"denied_tools": ["x"]})).allowed is True


def test_no_governance_config_allows_all():
    assert tg.evaluate_tool_policy([_tool("anything")], {"version": "v1"}).allowed is True
    assert tg.evaluate_tool_policy([_tool("anything")], None).allowed is True


def test_allowlist_permits_and_rejects():
    policy = _policy({"allowed_tools": ["get_weather", "search"]})
    assert tg.evaluate_tool_policy([_tool("get_weather")], policy).allowed is True
    decision = tg.evaluate_tool_policy([_tool("delete_db")], policy)
    assert decision.allowed is False
    assert decision.offending_tool == "delete_db"
    assert "allowlist" in (decision.reason or "")


def test_denylist_precedes_allowlist():
    policy = _policy({"allowed_tools": ["danger"], "denied_tools": ["danger"]})
    decision = tg.evaluate_tool_policy([_tool("danger")], policy)
    assert decision.allowed is False
    assert "denied" in (decision.reason or "")


def test_max_tools_cap():
    policy = _policy({"max_tools_per_request": 1})
    assert tg.evaluate_tool_policy([_tool("a")], policy).allowed is True
    decision = tg.evaluate_tool_policy([_tool("a"), _tool("b")], policy)
    assert decision.allowed is False
    assert "too many tools" in (decision.reason or "")


def test_max_tools_cap_ignores_bool():
    # True é int em Python; não deve ser tratado como um teto de 1.
    policy = _policy({"max_tools_per_request": True})
    assert tg.evaluate_tool_policy([_tool("a"), _tool("b")], policy).allowed is True


def test_per_tenant_override_tightens_allowlist():
    policy = _policy(
        {
            "allowed_tools": ["get_weather", "search"],
            "per_tenant": {"acme": {"allowed_tools": ["get_weather"]}},
        }
    )
    # tenant global pode usar search; acme não.
    assert tg.evaluate_tool_policy([_tool("search")], policy, tenant_id="globex").allowed is True
    assert tg.evaluate_tool_policy([_tool("search")], policy, tenant_id="acme").allowed is False
    assert tg.evaluate_tool_policy([_tool("get_weather")], policy, tenant_id="acme").allowed is True


def test_audit_tool_denial_is_best_effort(monkeypatch):
    captured = {}

    def fake_log(**kwargs):
        captured.update(kwargs)

    import app.roadmap_features as rf

    monkeypatch.setattr(rf, "log_audit_event", fake_log)
    decision = tg.ToolPolicyDecision(False, "tool 'x' is denied by policy", "x")
    tg.audit_tool_denial("acme", decision)
    assert captured["action"] == "tool_call_denied"
    assert captured["resource"] == "x"
    assert captured["tenant_id"] == "acme"


def test_audit_never_raises(monkeypatch):
    import app.roadmap_features as rf

    def boom(**kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(rf, "log_audit_event", boom)
    # Não deve propagar.
    tg.audit_tool_denial("acme", tg.ToolPolicyDecision(False, "r", "x"))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
