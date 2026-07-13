# -*- coding: utf-8 -*-
# Objective: Per-tenant/policy governance for tool calling (allowlist, cost cap, audit).
"""Governança de tool calling por *policy* / *tenant*.

Turnos de tool dão ao cliente o poder de fazer o modelo pedir a execução de funções
arbitrárias. Em ambiente multi-tenant isso precisa de trava: qual tenant pode pedir
quais tools, um teto de fan-out (controle de custo) e trilha de auditoria de
violações. A configuração vem do ``config`` da policy ativa (ver ``roadmap_features``):

    config["tool_governance"] = {
        "allowed_tools": ["get_weather", "search"],   # ausente/None => todas liberadas
        "denied_tools": ["delete_account"],           # sempre bloqueadas (precede allow)
        "max_tools_per_request": 32,                   # teto de tools por request (custo)
        "per_tenant": {                                # overrides por tenant
            "acme": {"allowed_tools": ["get_weather"]}
        },
    }

A avaliação é pura (``evaluate_tool_policy``); a auditoria de negações é best-effort
e nunca quebra o caminho de request.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class ToolPolicyDecision:
    """Outcome of evaluating requested tools against the active policy."""

    allowed: bool
    reason: Optional[str] = None
    offending_tool: Optional[str] = None


def _tool_names(tools: Any) -> List[str]:
    """Extract function names from an OpenAI-style ``tools`` array (best-effort)."""
    names: List[str] = []
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes)):
        return names
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("name"), str) and fn["name"]:
            names.append(fn["name"])
    return names


def _resolve_config(active_policy: Any, tenant_id: Any) -> Dict[str, Any]:
    """Merge the policy-level tool_governance config with any per-tenant override."""
    cfg: Dict[str, Any] = {}
    if isinstance(active_policy, dict):
        raw = active_policy.get("config")
        if isinstance(raw, dict):
            gov = raw.get("tool_governance")
            if isinstance(gov, dict):
                cfg = dict(gov)
    per_tenant = cfg.get("per_tenant")
    if tenant_id and isinstance(per_tenant, dict):
        override = per_tenant.get(str(tenant_id))
        if isinstance(override, dict):
            cfg = {**cfg, **override}
    return cfg


def evaluate_tool_policy(
    tools: Any,
    active_policy: Any = None,
    tenant_id: Any = None,
) -> ToolPolicyDecision:
    """Decide whether the requested tools are permitted for this tenant/policy.

    Precedence: fan-out cap → denylist → allowlist. A request with no tools, or no
    configured governance, is always allowed (opt-in enforcement).
    """
    names = _tool_names(tools)
    if not names:
        return ToolPolicyDecision(allowed=True)

    cfg = _resolve_config(active_policy, tenant_id)
    if not cfg:
        return ToolPolicyDecision(allowed=True)

    max_tools = cfg.get("max_tools_per_request")
    if isinstance(max_tools, int) and not isinstance(max_tools, bool) and max_tools >= 0 and len(names) > max_tools:
        return ToolPolicyDecision(False, f"too many tools requested: {len(names)} > {max_tools}")

    denied = cfg.get("denied_tools")
    if isinstance(denied, (list, tuple)):
        deny_set = {str(d) for d in denied}
        for name in names:
            if name in deny_set:
                return ToolPolicyDecision(False, f"tool '{name}' is denied by policy", name)

    allowed = cfg.get("allowed_tools")
    if isinstance(allowed, (list, tuple)):
        allow_set = {str(a) for a in allowed}
        for name in names:
            if name not in allow_set:
                return ToolPolicyDecision(False, f"tool '{name}' is not in the tenant allowlist", name)

    return ToolPolicyDecision(allowed=True)


def audit_tool_denial(tenant_id: Any, decision: ToolPolicyDecision, model: str = "tool_governance") -> None:
    """Persist an audit event for a denied tool request (best-effort, security trail)."""
    try:
        from app.roadmap_features import log_audit_event

        log_audit_event(
            actor=str(tenant_id or "anon"),
            action="tool_call_denied",
            resource=decision.offending_tool or "tool_policy",
            tenant_id=str(tenant_id) if tenant_id else None,
            metadata={"reason": decision.reason, "model": model},
        )
    except Exception as exc:  # pragma: no cover - auditoria jamais quebra o request
        logger.warning("[tool-gov] audit of tool denial failed: %s", exc)
