# Objective: An escaped exception must say which kind of failure it was.
"""There was no ``@app.exception_handler`` anywhere in the project.

Anything escaping a route handler became Starlette's
``500 {"detail":"Internal Server Error"}``: no correlation id, no error class,
nothing an operator could act on and nothing a client could distinguish from a
genuine defect in the router.
"""

import json

import pytest
from fastapi import Request
from sqlalchemy.exc import OperationalError

from app.services.error_responses import handle_unexpected


def a_request():
    return Request({"type": "http", "method": "POST", "path": "/query", "headers": []})


async def run(exc):
    response = await handle_unexpected(a_request(), exc)
    return response, json.loads(response.body)


@pytest.mark.asyncio
async def test_a_database_outage_is_a_503_with_retry_after():
    """The client should retry, and the response says when."""
    response, body = await run(OperationalError("SELECT 1", {}, Exception("mariadb em baixo")))
    assert response.status_code == 503
    assert body["error"] == "dependency_unavailable"
    assert response.headers["Retry-After"]


@pytest.mark.asyncio
@pytest.mark.parametrize("exc", [ConnectionError("redis"), TimeoutError("chroma"), OSError("socket")])
async def test_every_infrastructure_failure_is_a_503(exc):
    response, _ = await run(exc)
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_a_programming_error_stays_a_500():
    """Pretending a defect is transient makes the retry loop pay for it twice."""
    response, body = await run(ValueError("índice fora de alcance"))
    assert response.status_code == 500
    assert body["error"] == "internal_error"
    assert "Retry-After" not in response.headers


@pytest.mark.asyncio
async def test_the_response_carries_a_correlation_id_field():
    _, body = await run(ValueError("x"))
    assert "correlation_id" in body


@pytest.mark.asyncio
async def test_the_handler_survives_a_broken_correlation_context(monkeypatch):
    """It is the last line of defence; it must not raise on its way out."""
    monkeypatch.setattr(
        "app.services.error_responses.get_correlation_id",
        lambda: (_ for _ in ()).throw(RuntimeError("contexto perdido")),
    )
    response, body = await run(ValueError("x"))
    assert response.status_code == 500
    assert body["correlation_id"] is None


def test_the_handler_is_registered_on_the_app():
    from app.services.app_wiring import install_middleware

    registered = {}

    class _App:
        def add_middleware(self, cls, **kwargs):
            registered.setdefault("middleware", []).append(cls)

        def add_exception_handler(self, exc, handler):
            registered["handler"] = (exc, handler)

    install_middleware(_App(), (), gzip_min_size=500)
    assert registered["handler"][0] is Exception
    assert registered["handler"][1] is handle_unexpected


def test_the_middleware_order_is_preserved():
    """Last added runs first, so the order the stack is given is the contract."""
    from app.services.app_wiring import install_middleware

    added = []

    class _App:
        def add_middleware(self, cls, **kwargs):
            added.append(cls)

        def add_exception_handler(self, exc, handler):
            pass

    class A:
        pass

    class B:
        pass

    install_middleware(_App(), (A, B), gzip_min_size=500)
    assert added[:2] == [A, B]
