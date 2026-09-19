# Objective: Property-based API contract tests (schemathesis) for the public endpoints.
"""Fuzz the public OpenAPI operations and check the app never answers 5xx and honors its schema.

Providers, judges, RAG and background work are stubbed; the HTTP layer
(validation, routing, auth, response models, error mapping) runs for real.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import requests
import schemathesis
from hypothesis import HealthCheck, reject, settings
from schemathesis.checks import not_a_server_error
from schemathesis.specs.openapi.checks import response_schema_conformance

pytestmark = pytest.mark.contract

PUBLIC_PATHS = r"^/(query|v1/query|health|healthz|v1/health|feedback|feedback/stats|v1/chat/completions)$"


def _routed_result():
    return {
        "result": {
            "answer": "resposta",
            "model": "ollama/gemma3:4b",
            "modality": "text",
            "image_output_b64": None,
            "latency_s": 0.5,
            "cost_per_1k": 0.0,
            "metadata": {"prompt_tokens": 3, "completion_tokens": 5},
            "route": {"chosen_model": "ollama/gemma3:4b", "modality_selected": "text", "fallback": {"used": False}},
            "candidates": [],
        },
        "image_input": None,
        "selected_policy": None,
        "assigned_variant": None,
        "modality": "text",
    }


def _without_lifespan(app):
    """Skip startup/shutdown (DB, vector store, background workers): each case would restart them."""

    async def asgi(scope, receive, send):
        if scope["type"] != "lifespan":
            return await app(scope, receive, send)
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return

    return asgi


@pytest.fixture
def api_schema():
    stubs = [
        patch("app.main.get_redis", return_value=MagicMock()),
        patch("app.main.init_vectorstore"),
        patch("app.main._ensure_model_metrics_table"),
        patch("app.main.preload_ollama_models", new_callable=AsyncMock),
        patch("app.main.vs_add_document", new_callable=AsyncMock),
        patch("app.main.record_query_side_effects"),
        patch("app.main.process_query_request", new_callable=AsyncMock, return_value=_routed_result()),
    ]
    for stub in stubs:
        stub.start()
    try:
        from app.main import app

        yield schemathesis.openapi.from_asgi("/openapi.json", _without_lifespan(app))
    finally:
        for stub in reversed(stubs):
            stub.stop()


schema = schemathesis.pytest.from_fixture("api_schema").include(path_regex=PUBLIC_PATHS)


@schema.parametrize()
@settings(
    max_examples=15,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow, HealthCheck.filter_too_much],
)
def test_public_api_contract(case):
    try:
        case.call_and_validate(checks=[not_a_server_error, response_schema_conformance])
    except requests.exceptions.InvalidHeader:
        reject()  # valor de header que o cliente HTTP se recusa a enviar: nunca chega ao servidor
