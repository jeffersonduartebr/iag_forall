# -*- coding: utf-8 -*-
"""Testes do endpoint /query/stream: fast-path real vs fallback (roadmap item #1)."""

from __future__ import annotations

import asyncio

import pytest
from app.schemas import QueryRequest
from app.services import query_http


async def _collect_sse(response) -> str:
    chunks = []
    async for piece in response.body_iterator:
        chunks.append(piece if isinstance(piece, str) else piece.decode())
    return "".join(chunks)


def _events(sse_text: str) -> list[str]:
    return [line.split("event: ", 1)[1] for line in sse_text.splitlines() if line.startswith("event: ")]


def test_real_stream_fast_path(monkeypatch):
    # Seleciona um modelo streamável e injeta tokens reais (sem rede).
    async def fake_select(query, modality="text"):
        return "openai/gpt-4o"

    async def fake_astream(model, prompt, **kwargs):
        for piece in ["Hel", "lo!"]:
            yield query_http_stream_event("delta", piece)
        yield query_http_stream_event("final", "", finish="stop", pt=5, ct=2)

    async def ok_guardrail(_q):
        return type("D", (), {"allowed": True, "reasons": []})()

    async def ok_budget(_t):
        return type("B", (), {"allowed": True, "reason": None})()

    monkeypatch.setattr("app.providers_stream.select_stream_model", fake_select)
    monkeypatch.setattr("app.providers_stream.astream_model", fake_astream)
    monkeypatch.setattr("app.services.hot_path_runtime.check_input_guardrails_async", ok_guardrail)
    monkeypatch.setattr("app.services.governance_runtime.check_runtime_budget_async", ok_budget)
    # Não bater no Celery/DB nos efeitos colaterais.
    monkeypatch.setattr(query_http, "_record_stream_side_effects", lambda *a, **k: None)

    req = QueryRequest(query="hi there", modality="text")
    resp = asyncio.run(query_http.execute_query_stream(req))
    sse = asyncio.run(_collect_sse(resp))

    assert _events(sse) == ["meta", "token", "token", "done"]
    assert "Hel" in sse and "lo!" in sse
    assert "streamed" in sse and "completed" in sse


def test_fallback_when_tools_present(monkeypatch):
    # Requests com tools NÃO fazem streaming real: caem no pseudo-stream completo.
    class FakeMain:
        async def process_query_request(self, req):
            return {"result": {"model": "m", "modality": "text", "answer": "final answer text", "metadata": {}}}

    monkeypatch.setattr(query_http, "_main", lambda: FakeMain())
    # Se por engano tentasse o fast-path, isto falharia o teste:
    monkeypatch.setattr(
        "app.providers_stream.select_stream_model",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not select")),
    )

    req = QueryRequest(query="use a tool", modality="text", tools=[{"type": "function", "function": {"name": "f"}}])
    resp = asyncio.run(query_http.execute_query_stream(req))
    sse = asyncio.run(_collect_sse(resp))
    assert _events(sse)[0] == "meta"
    assert "token" in _events(sse)
    assert "final" in sse or "answer" in sse


def test_guardrail_block_emits_error(monkeypatch):
    async def fake_select(query, modality="text"):
        return "openai/gpt-4o"

    async def block_guardrail(_q):
        return type("D", (), {"allowed": False, "reasons": ["blocked"]})()

    monkeypatch.setattr("app.providers_stream.select_stream_model", fake_select)
    monkeypatch.setattr("app.services.hot_path_runtime.check_input_guardrails_async", block_guardrail)

    req = QueryRequest(query="bad", modality="text")
    resp = asyncio.run(query_http.execute_query_stream(req))
    sse = asyncio.run(_collect_sse(resp))
    assert _events(sse) == ["error", "done"]
    assert "guardrail_block" in sse


def test_not_eligible_when_rag_enabled(monkeypatch):
    class FakeMain:
        async def process_query_request(self, req):
            return {"result": {"model": "m", "modality": "text", "answer": "rag answer", "metadata": {}}}

    monkeypatch.setattr(query_http, "_main", lambda: FakeMain())
    req = QueryRequest(query="q", modality="text", enable_rag_for_answer=True)
    assert query_http._real_stream_eligible(req) is False


def query_http_stream_event(kind, text, finish=None, pt=0, ct=0):
    from app.providers_stream import StreamEvent

    return StreamEvent(kind, text=text, finish_reason=finish, prompt_tokens=pt, completion_tokens=ct)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
