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
