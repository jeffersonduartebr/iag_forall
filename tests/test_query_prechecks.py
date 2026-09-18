# Objective: Test coverage for the concurrent query pre-checks.
"""process_query_request: cheap guardrails first, then tenant budget and active policy concurrently."""

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.services import query_runtime as qr


def _req(**overrides):
    base = dict(query="pergunta", tenant_id="t1", modality="text", image_b64=None, images=None, policy_version=None)
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.asyncio
async def test_budget_and_policy_reads_overlap(monkeypatch):
    budget_started = asyncio.Event()

    async def guardrails(query):
        return SimpleNamespace(allowed=True, reasons=[])

    async def budget(tenant_id):
        # Só termina se a leitura da política começar enquanto o orçamento espera.
        await asyncio.wait_for(budget_started.wait(), timeout=2.0)
        return SimpleNamespace(
            allowed=False, reason="daily", daily_spent=2.0, monthly_spent=2.0, daily_limit=1.0, monthly_limit=10.0
        )

    async def policy():
        budget_started.set()
        return None

    monkeypatch.setattr(qr, "check_input_guardrails_async", guardrails)
    monkeypatch.setattr(qr, "check_tenant_budget", budget)
    monkeypatch.setattr(qr, "get_active_policy", policy)

    with pytest.raises(HTTPException) as exc:
        await qr.process_query_request(_req())
    assert exc.value.status_code == 429


@pytest.mark.asyncio
async def test_blocked_guardrail_skips_budget_and_policy(monkeypatch):
    async def guardrails(query):
        return SimpleNamespace(allowed=False, reasons=["prompt_injection_signal"])

    async def must_not_run(*_a):
        raise AssertionError("não deveria consultar orçamento/política")

    monkeypatch.setattr(qr, "check_input_guardrails_async", guardrails)
    monkeypatch.setattr(qr, "check_tenant_budget", must_not_run)
    monkeypatch.setattr(qr, "get_active_policy", must_not_run)

    with pytest.raises(HTTPException) as exc:
        await qr.process_query_request(_req())
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_budget_rejection_wins_over_policy_failure(monkeypatch):
    async def guardrails(query):
        return SimpleNamespace(allowed=True, reasons=[])

    async def budget(tenant_id):
        return SimpleNamespace(
            allowed=False, reason="daily", daily_spent=2.0, monthly_spent=2.0, daily_limit=1.0, monthly_limit=10.0
        )

    async def policy():
        raise RuntimeError("redis fora")

    monkeypatch.setattr(qr, "check_input_guardrails_async", guardrails)
    monkeypatch.setattr(qr, "check_tenant_budget", budget)
    monkeypatch.setattr(qr, "get_active_policy", policy)

    with pytest.raises(HTTPException) as exc:
        await qr.process_query_request(_req())
    assert exc.value.status_code == 429


@pytest.mark.asyncio
async def test_budget_rejection_after_guardrails_pass(monkeypatch):
    async def guardrails(query):
        return SimpleNamespace(allowed=True, reasons=[])

    async def budget(tenant_id):
        return SimpleNamespace(
            allowed=False, reason="daily", daily_spent=2.0, monthly_spent=2.0, daily_limit=1.0, monthly_limit=10.0
        )

    async def policy():
        return None

    monkeypatch.setattr(qr, "check_input_guardrails_async", guardrails)
    monkeypatch.setattr(qr, "check_tenant_budget", budget)
    monkeypatch.setattr(qr, "get_active_policy", policy)

    with pytest.raises(HTTPException) as exc:
        await qr.process_query_request(_req())
    assert exc.value.status_code == 429 and exc.value.detail["category"] == "tenant_budget_exceeded"
