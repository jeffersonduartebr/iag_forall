# -*- coding: utf-8 -*-
"""Testes do streaming real de tokens do provedor (roadmap item #1)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app import providers_stream as ps


def test_provider_family_and_streamable():
    assert ps.provider_family("openai/gpt-4o") == "openai"
    assert ps.provider_family("gemma3:4b") == "ollama"  # sem prefixo => ollama
    assert ps.is_streamable("openai/gpt-4o") is True
    assert ps.is_streamable("anthropic/claude-3-5-sonnet") is True
    assert ps.is_streamable("ollama/gemma3:4b") is True
    assert ps.is_streamable("gemini/gemini-1.5-flash") is False
    assert ps.is_streamable(None) is False


def test_astream_unsupported_provider_raises():
    async def _run():
        gen = ps.astream_model("gemini/gemini-1.5-flash", "hi")
        return [ev async for ev in gen]

    import asyncio

    with pytest.raises(ps.StreamingUnsupportedError):
        asyncio.run(_run())


class _FakeDelta:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content, finish=None):
        self.delta = _FakeDelta(content)
        self.finish_reason = finish


class _FakeChunk:
    def __init__(self, content=None, finish=None, usage=None):
        self.choices = [_FakeChoice(content, finish)] if content is not None or finish else []
        self.usage = usage


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        async def _gen():
            for c in self._chunks:
                yield c

        return _gen()


class _FakeCompletions:
    def __init__(self, chunks):
        self._chunks = chunks
        self.captured = {}

    async def create(self, **kwargs):
        self.captured.update(kwargs)
        return _FakeStream(self._chunks)


def test_stream_openai_yields_deltas_and_final(monkeypatch):
    usage = SimpleNamespace(prompt_tokens=11, completion_tokens=7)
    chunks = [
        _FakeChunk(content="Hello"),
        _FakeChunk(content=" world"),
        _FakeChunk(content=None, finish="stop", usage=usage),
    ]
    completions = _FakeCompletions(chunks)
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    import app.providers_async as pa

    monkeypatch.setattr(pa.ProviderFactory, "get_provider", classmethod(lambda cls, m: SimpleNamespace(client=fake_client)))

    async def _run():
        return [ev async for ev in ps.astream_model("openai/gpt-4o", "hi", temperature=0.3, max_tokens=64)]

    import asyncio

    events = asyncio.run(_run())
    deltas = [e.text for e in events if e.type == "delta"]
    finals = [e for e in events if e.type == "final"]
    assert deltas == ["Hello", " world"]
    assert len(finals) == 1
    assert finals[0].prompt_tokens == 11 and finals[0].completion_tokens == 7
    assert finals[0].finish_reason == "stop"
    # request usou stream=True e o model sem prefixo de provedor
    assert completions.captured["stream"] is True
    assert completions.captured["model"] == "gpt-4o"
    assert completions.captured["max_tokens"] == 64


def test_stream_openai_gpt5_uses_max_completion_tokens(monkeypatch):
    completions = _FakeCompletions([_FakeChunk(content="ok", finish="stop")])
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    import app.providers_async as pa

    monkeypatch.setattr(pa.ProviderFactory, "get_provider", classmethod(lambda cls, m: SimpleNamespace(client=fake_client)))

    import asyncio

    asyncio.run(_drain(ps.astream_model("openai/gpt-5-mini", "hi", max_tokens=99)))
    assert completions.captured.get("max_completion_tokens") == 99
    assert "max_tokens" not in completions.captured


async def _drain(gen):
    return [ev async for ev in gen]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_stream_ollama_parses_ndjson(monkeypatch):
    import json

    import httpx

    seen = {}
    lines = [
        {"response": "Olá"},
        "",
        "{nao-json",
        {"response": ", mundo"},
        {"response": "", "done": True, "prompt_eval_count": 7, "eval_count": 3},
    ]

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        body = "\n".join(line if isinstance(line, str) else json.dumps(line) for line in lines)
        return httpx.Response(200, text=body)

    async def fake_client():
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    monkeypatch.setattr("app.providers_async.get_http_client", fake_client)
    monkeypatch.setattr("app.providers_async.OLLAMA_HOST", "http://ollama:11434")

    async def _run():
        return [ev async for ev in ps._stream_ollama("ollama/gemma3:4b", "oi", "sys", 0.3, 32, 0.2)]

    import asyncio

    events = asyncio.run(_run())
    assert [e.text for e in events if e.type == "delta"] == ["Olá", ", mundo"]
    final = events[-1]
    assert (final.type, final.prompt_tokens, final.completion_tokens) == ("final", 7, 3)
    assert seen["url"] == "http://ollama:11434/api/generate"
    assert seen["body"]["model"] == "gemma3:4b" and seen["body"]["system"] == "sys"
    assert seen["body"]["options"] == {"temperature": 0.3, "num_predict": 32, "num_ctx": 4096}


def test_stream_ollama_raises_on_http_error(monkeypatch):
    import httpx

    async def fake_client():
        return httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(503)))

    monkeypatch.setattr("app.providers_async.get_http_client", fake_client)

    async def _run():
        return [ev async for ev in ps._stream_ollama("gemma3:4b", "oi", "", 0.3, 32, None)]

    import asyncio

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(_run())


def test_warmup_and_calls_share_one_context_size():
    """A warm-up with a different num_ctx made Ollama reload the model on every switch (prod, 2026-09-24)."""
    import inspect

    import app.providers_async  # noqa: F401  (ordem de import dos provedores)
    from app.providers import _ollama, _ollama_adapter
    from app.providers._infra import OLLAMA_NUM_CTX

    assert "OLLAMA_NUM_CTX" in inspect.getsource(_ollama.warm_ollama_model_runtime)
    assert "OLLAMA_NUM_CTX" in inspect.getsource(_ollama_adapter)
    assert OLLAMA_NUM_CTX == 4096
