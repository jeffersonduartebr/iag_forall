# Objective: Behavioural coverage for the admin log stream (Loki polling and line parsing).
"""admin_logs: line parsing, Loki response handling and cursor advancement, with no network."""

import json
from types import SimpleNamespace

import httpx
import pytest


@pytest.fixture
def al(fake_clock):
    from app.services import admin_logs

    fake_clock(admin_logs)
    return admin_logs


def test_parse_empty_line(al):
    assert al._parse_log_line("  ") == {"message": "", "level": "info", "timestamp": al.time.time()}
    assert al._parse_log_line(None)["message"] == ""


def test_parse_structured_line_prefers_known_keys(al):
    raw = json.dumps({"ts": 5, "levelname": "WARNING", "msg": "lento", "correlation_id": "c1"})
    assert al._parse_log_line(raw) == {
        "timestamp": 5,
        "level": "WARNING",
        "event": "lento",
        "correlation_id": "c1",
        "message": raw,
    }
    minimal = al._parse_log_line("{}")
    assert (minimal["level"], minimal["event"], minimal["correlation_id"]) == ("info", "", None)


@pytest.mark.parametrize("raw", ["texto simples", "[1, 2]", "{meio json"])
def test_parse_unstructured_line_is_the_event(al, raw):
    assert al._parse_log_line(raw) == {"timestamp": al.time.time(), "level": "info", "event": raw, "message": raw}


def _client_factory(handler, seen):
    real = httpx.AsyncClient

    def factory(**kwargs):
        seen.append(kwargs)
        return real(transport=httpx.MockTransport(handler), **kwargs)

    return SimpleNamespace(AsyncClient=factory)


@pytest.mark.asyncio
async def test_fetch_merges_streams_sorted_by_timestamp(al, monkeypatch):
    requests, seen = [], []
    payload = {
        "data": {
            "result": [
                {"stream": {"container": "api"}, "values": [["30", "c"], ["10", '{"level": "error", "event": "a"}']]},
                {"stream": {"container_name": "celery"}, "values": [["20", "b"]]},
                {"values": None},
            ]
        }
    }

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=payload)

    monkeypatch.setattr(al, "httpx", _client_factory(handler, seen))
    out = await al._fetch_loki_logs(query="{x}", start_ns=1, end_ns=2, limit=7)
    assert [(e["ts_ns"], e["container"], e["event"]) for e in out] == [
        (10, "api", "a"),
        (20, "celery", "b"),
        (30, "api", "c"),
    ]
    assert out[0]["level"] == "error"
    params = dict(requests[0].url.params)
    assert params == {"query": "{x}", "start": "1", "end": "2", "limit": "7", "direction": "forward"}
    assert requests[0].url.path == "/loki/api/v1/query_range" and seen[0]["timeout"] == 5.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "respond",
    [
        lambda r: httpx.Response(503, text="indisponível"),
        lambda r: httpx.Response(200, json={"data": {}}),
        lambda r: httpx.Response(200, text="nao-json"),
    ],
)
async def test_fetch_degrades_to_empty(al, monkeypatch, respond):
    monkeypatch.setattr(al, "httpx", _client_factory(respond, []))
    assert await al._fetch_loki_logs(query="q", start_ns=0, end_ns=1) == []


@pytest.mark.asyncio
async def test_fetch_network_error_is_empty(al, monkeypatch):
    def handler(request):
        raise httpx.ConnectError("sem rota", request=request)

    monkeypatch.setattr(al, "httpx", _client_factory(handler, []))
    assert await al._fetch_loki_logs(query="q", start_ns=0, end_ns=1) == []


class _StopError(Exception):
    pass


@pytest.mark.asyncio
async def test_stream_advances_cursor_and_respects_min_poll(al, monkeypatch):
    now_ns = int(al.time.time() * 1_000_000_000)
    batches = iter([[{"ts_ns": now_ns + 5, "event": "a"}, {"ts_ns": None, "event": "b"}], []])
    windows, sleeps = [], []

    async def fake_fetch(*, query, start_ns, end_ns, limit):
        windows.append((query, start_ns, end_ns, limit))
        return next(batches)

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 2:
            raise _StopError

    monkeypatch.setattr(al, "_fetch_loki_logs", fake_fetch)
    monkeypatch.setattr(al, "asyncio", SimpleNamespace(sleep=fake_sleep))
    got = []
    with pytest.raises(_StopError):
        async for entry in al.stream_logs(query="{q}", poll_s=0.1):
            got.append(entry["event"])
    assert got == ["a", "b"]
    assert sleeps == [0.5, 0.5]  # nunca abaixo de 0,5 s
    assert windows[0][1] == now_ns - 30 * 1_000_000_000  # começa 30 s atrás
    assert windows[1][1] == now_ns + 5 and windows[1][3] == 100


@pytest.mark.asyncio
async def test_stream_resumes_from_given_cursor(al, monkeypatch):
    windows = []

    async def fake_fetch(*, query, start_ns, end_ns, limit):
        windows.append(start_ns)
        return []

    async def fake_sleep(seconds):
        raise _StopError

    monkeypatch.setattr(al, "_fetch_loki_logs", fake_fetch)
    monkeypatch.setattr(al, "asyncio", SimpleNamespace(sleep=fake_sleep))
    with pytest.raises(_StopError):
        async for _ in al.stream_logs(last_ts_ns=123):
            pass
    assert windows == [123]
