# Objective: Turn an unhandled exception into a structured, correlated response.
"""The project had no ``@app.exception_handler`` at all.

Anything that escaped a route handler became Starlette's default
``500 {"detail":"Internal Server Error"}``: no correlation id, no error class,
nothing an operator could act on and nothing a client could distinguish from a
genuine bug in the router.

The distinction this draws is between a *dependency* being unavailable and the
application being wrong. A database, cache or provider outage is a 503 — the
client should retry, and the response says so. Everything else stays a 500,
because pretending a programming error is transient just makes the retry loop
pay for it twice.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter
from sqlalchemy.exc import SQLAlchemyError

from app.correlation import get_correlation_id
from app.observability import registry

logger = logging.getLogger(__name__)

UNHANDLED_ERRORS_TOTAL = Counter(
    "unhandled_errors_total",
    "Exceptions that reached the application-wide handler",
    ["kind"],
    registry=registry,
)

#: Falhas de infraestrutura: o cliente deve tentar outra vez, e o Retry-After
#: diz-lhe quando. Um erro de programação não entra aqui de propósito.
_UNAVAILABLE = (SQLAlchemyError, ConnectionError, TimeoutError, OSError)

RETRY_AFTER_SECONDS = "5"


async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
    """Map an escaped exception onto 503 (dependency) or 500 (defect)."""
    correlation_id = _correlation_id()
    unavailable = isinstance(exc, _UNAVAILABLE)
    kind = "dependency_unavailable" if unavailable else "internal_error"
    UNHANDLED_ERRORS_TOTAL.labels(kind=kind).inc()
    logger.exception(
        "[api] %s em %s %s (correlation_id=%s): %s",
        kind,
        request.method,
        request.url.path,
        correlation_id,
        exc,
    )
    body: dict[str, Any] = {
        "detail": (
            "Uma dependência do serviço está indisponível. Tente novamente."
            if unavailable
            else "Erro interno."
        ),
        "error": kind,
        "correlation_id": correlation_id,
    }
    headers = {"Retry-After": RETRY_AFTER_SECONDS} if unavailable else None
    return JSONResponse(status_code=503 if unavailable else 500, content=body, headers=headers)


def _correlation_id() -> str | None:
    """Never let the handler itself raise; it is the last line of defence."""
    try:
        return get_correlation_id()
    except Exception:
        return None
