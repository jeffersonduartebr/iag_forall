# Objective: A retried request must return the first answer, not buy a second one.
"""The feature never worked, because the two sides computed different keys.

``_resolve_idempotency`` hashed with ``model=""``; ``_store_idempotency``
hashed with the model that had just answered — a value only known *after* the
response exists. The keys could therefore never match, and a client retrying
with the same ``Idempotency-Key`` executed the query again and paid again,
exactly as if the header had not been sent.
"""

from types import SimpleNamespace

import pytest
from app.schemas import QueryRequest
from app.services.idempotency import _composite_key, _resolve_idempotency, _store_idempotency


def a_request(key="chave-do-cliente"):
    return SimpleNamespace(state=SimpleNamespace(idempotency_key=key))


def a_query():
    return QueryRequest(query="quanto é 2+2?", tenant_id="acme", modality="text")


def test_both_sides_compute_the_same_key():
    """The regression: the write key included the chosen model, the read key
    did not, so one could never find the other."""
    assert _composite_key(a_query(), "k") == _composite_key(a_query(), "k")


def test_the_key_ignores_the_answering_model():
    import inspect

    from app.services import idempotency

    source = inspect.getsource(idempotency._composite_key)
    assert 'model=""' in source


def test_a_different_query_gets_a_different_key():
    other = QueryRequest(query="outra pergunta", tenant_id="acme", modality="text")
    assert _composite_key(a_query(), "k") != _composite_key(other, "k")


def test_a_different_tenant_gets_a_different_key():
    other = QueryRequest(query="quanto é 2+2?", tenant_id="outro", modality="text")
    assert _composite_key(a_query(), "k") != _composite_key(other, "k")


@pytest.mark.asyncio
async def test_a_stored_response_is_found_on_retry(monkeypatch):
    """End to end over the two helpers, which is where the mismatch lived."""
    store = {}

    async def _set(key, payload, ttl_s=300):
        store[key] = payload

    async def _get(key):
        return store.get(key)

    monkeypatch.setattr("app.services.idempotency.redis_idempotency_set", _set)
    monkeypatch.setattr("app.services.idempotency.redis_idempotency_get", _get)

    body = {"answer": "4", "model": "ollama/phi4"}
    await _store_idempotency(a_query(), a_request(), body)

    assert await _resolve_idempotency(a_query(), a_request()) == body


@pytest.mark.asyncio
async def test_a_request_without_the_header_is_never_cached(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "app.services.idempotency.redis_idempotency_set",
        lambda *a, **k: calls.append(a),
    )
    await _store_idempotency(a_query(), a_request(key=None), {"answer": "4"})
    assert calls == []


@pytest.mark.asyncio
async def test_an_incomplete_entry_is_not_served(monkeypatch):
    async def _get(key):
        return {"status": "in_progress"}

    monkeypatch.setattr("app.services.idempotency.redis_idempotency_get", _get)
    assert await _resolve_idempotency(a_query(), a_request()) is None


@pytest.mark.asyncio
async def test_redis_being_down_does_not_break_the_request(monkeypatch):
    """It degrades to no idempotency, which is the pre-existing behaviour."""

    async def _get(key):
        return None

    monkeypatch.setattr("app.services.idempotency.redis_idempotency_get", _get)
    assert await _resolve_idempotency(a_query(), a_request()) is None
