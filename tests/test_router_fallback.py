# Objective: The fallback chain must be reachable, enabled, and actually able to run.
"""A fallback chain that is written, tested and switched off protects nobody.

``REQUEST_FALLBACK_ENABLED`` was read with ``default=False`` and was absent from
the settings catalog, so a failed model reached the user as 502 with healthy
alternatives configured. Turning it on exposed a second defect: the chain called
``pybreaker.CircuitBreaker.call_async``, which is Tornado-only and raises
``NameError`` — so the chain could never have succeeded either.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from app.services.router_execution import route_and_answer_internal_impl
from router_fakes import deps_for_execution


def test_the_fallback_setting_is_declared_and_enabled():
    """An undeclared setting silently defaults to off, whatever the code intends.

    ``REQUEST_FALLBACK_ENABLED`` was read with ``default=False`` and was absent
    from the catalog and the compose file — so the whole fallback chain, which
    is written and tested, never ran on the request path. There was not even a
    way to discover the key existed by reading the configuration.
    """
    from app.config.settings_catalog import SETTINGS_BY_DOMAIN

    catalog = {k: v for domain in SETTINGS_BY_DOMAIN.values() for k, v in domain.items()}
    assert catalog["REQUEST_FALLBACK_ENABLED"] == "1"
    assert catalog["REQUEST_MAX_FALLBACKS"] == "2"


@pytest.mark.asyncio
async def test_with_the_shipped_defaults_a_dead_primary_falls_back():
    """Resolve the flag from the real catalog, not from a fake that says True."""
    from app.config.settings_catalog import SETTINGS_BY_DOMAIN

    catalog = {k: v for domain in SETTINGS_BY_DOMAIN.values() for k, v in domain.items()}

    def _catalog_bool(key, default=False):
        return str(catalog.get(key, "1" if default else "0")).strip() == "1"

    used_fallback = {"value": False}

    async def _fallback(**kwargs):
        used_fallback["value"] = True
        return SimpleNamespace(
            success=True,
            result=("resposta do suplente", {"prompt_tokens": 1, "completion_tokens": 1, "cost_per_1k": 0.1}),
            model_used="openai/gpt-4o",
            models_tried=["ollama/quebrado", "openai/gpt-4o"],
            errors=["ollama/quebrado: modelo em baixo"],
        )

    deps = deps_for_execution()
    deps["check_cache"] = lambda *a, **k: None
    deps["call_model"] = lambda **kwargs: ("ignored", {})
    deps["execute_with_fallback"] = _fallback
    deps["_safe_setting_bool"] = _catalog_bool

    out = await route_and_answer_internal_impl(
        deps=deps,
        query="pergunta",
        system_prompt="SYS",
        use_rag=False,
        max_tokens=64,
        temperature=0.7,
        modality="text",
        image_b64=None,
        rag_modality="text",
        use_cache=False,
    )

    assert used_fallback["value"], "a cadeia de fallback não correu com os defaults de fábrica"
    assert out["answer"] == "resposta do suplente"
