# Objective: What every OpenRouter request carries (prompt review against the official docs, 2026-09-25).
from __future__ import annotations

import app.providers_async  # noqa: F401  (ordem de import dos provedores)
from app.providers._openai import OpenRouterProvider


def _args(model: str, max_tokens: int = 512) -> dict:
    return OpenRouterProvider._reasoning_args(object.__new__(OpenRouterProvider), model, max_tokens)


def test_student_data_is_never_sent_to_providers_that_retain_prompts():
    """provider.data_collection=deny: the traffic carries minors' submissions (LGPD)."""
    assert _args("deepseek/deepseek-v4.1-flash")["extra_body"]["provider"] == {"data_collection": "deny"}


def test_openai_models_get_effort_and_the_others_a_token_budget():
    """OpenAI models only accept reasoning.effort; reasoning counts inside max_tokens for every model."""
    openai, other = _args("openai/gpt-5.6-luna"), _args("anthropic/claude-sonnet-5")
    assert openai["extra_body"]["reasoning"] == {"effort": "medium"}
    assert other["extra_body"]["reasoning"] == {"max_tokens": 4096}
    assert openai["max_tokens"] == other["max_tokens"] == 512 + 4096


def test_zero_budget_keeps_the_privacy_policy(monkeypatch):
    from app.providers import _openai

    monkeypatch.setattr(_openai, "orcamento_raciocinio", lambda: 0)
    assert _args("x/y") == {"extra_body": {"provider": {"data_collection": "deny"}}}
