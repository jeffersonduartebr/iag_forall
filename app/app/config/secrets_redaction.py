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


def redact_secrets(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Mask secret values in a settings snapshot."""
    redacted = dict(snapshot)
    for key in list(redacted.keys()):
        upper = key.upper()
        if upper in SECRET_SETTING_KEYS or any(token in upper for token in ("PASSWORD", "SECRET", "TOKEN", "API_KEY")):
            value = redacted.get(key)
            if value not in (None, ""):
                redacted[key] = "***REDACTED***"
    return redacted
