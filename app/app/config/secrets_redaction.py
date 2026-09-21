# Objective: Redact secret values from settings snapshots.
"""Helpers to mask sensitive configuration in admin responses."""

from __future__ import annotations

from typing import Any, Dict

SECRET_SETTING_KEYS = frozenset({
    "ADMIN_TOKEN",
    "ADMIN_TOKEN_PREVIOUS",
    "API_KEYS",
    "JWT_SECRET",
    "REDIS_PASSWORD",
    "DB_PASS",
    "MYSQL_ROOT_PASSWORD",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
    "METRICS_TOKEN",
})


#: Sufixos que denunciam uma credencial. A convenção é que a chave *termina*
#: no marcador — `ADMIN_TOKEN`, `JWT_SECRET`, `REDIS_PASSWORD`, `DB_PASS`,
#: `OPENROUTER_API_KEY`, `API_KEYS`. O plural só existe para `KEYS`, porque
#: `..._TOKENS` é sempre uma contagem de tokens e nunca uma credencial.
SECRET_SUFFIXES = ("KEY", "KEYS", "TOKEN", "SECRET", "PASSWORD", "PASS")


def is_secret_key(key: str) -> bool:
    """Whether this setting holds a credential.

    A regra era `any(marker in key)` — substring, não sufixo — e isso escondia
    do operador seis definições perfeitamente operacionais como se fossem
    segredos: `MAX_TOKENS_DEFAULT`, os três `RAG_*_CONTEXT_TOKEN_BUDGET`,
    `REWARD_LATENCY_TOKENS_PER_S` e — o melhor — `RAG_SIMPLE_QUERY_BYPASS_ENABLED`,
    porque "BYPASS" contém "PASS". Nenhuma delas é uma credencial, e nenhuma
    era visível em `/admin/settings`.

    A lista explícita continua a ser a autoridade; o sufixo é a rede para uma
    chave nova que ninguém se lembre de lá pôr.
    """
    upper = key.upper()
    return upper in SECRET_SETTING_KEYS or any(upper.endswith(s) for s in SECRET_SUFFIXES)


def redact_secrets(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Mask secret values in a settings snapshot."""
    redacted = dict(snapshot)
    for key in list(redacted.keys()):
        if is_secret_key(key) and redacted.get(key) not in (None, ""):
            redacted[key] = "***REDACTED***"
    return redacted
