# Objective: Test coverage for settings config behavior and regressions.
"""Tests for settings catalog and coercion helpers."""

from app.config.settings_catalog import (
    SETTING_METADATA,
    SETTINGS_BY_DOMAIN,
    SETTINGS_DEFAULTS,
    is_known_setting,
    is_runtime_mutable,
    known_setting_keys,
    metadata_for,
)
from app.config.settings_sources import decode_redis_value, resolve_setting_value
from app.config.settings_types import as_bool, as_float, as_int, as_list
from app.settings_dynamic import DynamicSettings


def test_settings_catalog_is_flattened_from_domains():
    """Flattened defaults should preserve domain keys and metadata."""
    assert "runtime" in SETTINGS_BY_DOMAIN
    assert SETTINGS_DEFAULTS["MAX_TOKENS_DEFAULT"] == "2000"
    assert SETTING_METADATA["MAX_TOKENS_DEFAULT"]["domain"] == "runtime"
    assert "OLLAMA_HOST" in known_setting_keys()


def test_settings_catalog_marks_restart_sensitive_keys():
    """Some infra/provider keys should be marked as restart-sensitive."""
    assert SETTING_METADATA["OLLAMA_HOST"]["mutability"] == "requires_restart"
    assert SETTING_METADATA["MAX_TOKENS_DEFAULT"]["mutability"] == "runtime_safe"
    assert is_runtime_mutable("MAX_TOKENS_DEFAULT") is True
    assert is_runtime_mutable("OLLAMA_HOST") is False
    assert is_known_setting("MAX_TOKENS_DEFAULT") is True
    assert metadata_for("OLLAMA_HOST")["domain"] == "providers"


def test_decode_redis_value_normalizes_supported_types():
    """Redis decoding should normalize bytes and scalar values to strings."""
    assert decode_redis_value(b"abc") == "abc"
    assert decode_redis_value(7) == "7"
    assert decode_redis_value(True) == "True"
    assert decode_redis_value(None) is None
    assert decode_redis_value(object()) is None


def test_resolve_setting_value_uses_cache_then_sources():
    """Source resolution should prefer cache, then redis, DB, env, and defaults."""
    cache = {}
    value = resolve_setting_value(
        key="A",
        fallback="fallback",
        defaults={"A": "default"},
        cache_get=lambda key: cache.get(key),
        cache_set=lambda key, value: cache.__setitem__(key, value),
        redis_get=lambda key: None,
        db_get=lambda key: None,
        env_get=lambda key: None,
    )
    assert value == "default"
    assert cache["A"] == "default"
    assert resolve_setting_value(
        key="A",
        fallback="fallback",
        defaults={"A": "default"},
        cache_get=lambda key: "cached",
        cache_set=lambda key, value: None,
        redis_get=lambda key: "redis",
        db_get=lambda key: "db",
        env_get=lambda key: "env",
    ) == "cached"


def test_as_list_bool_int_float_cover_fallback_paths():
    """Type coercion helpers should cover JSON, csv, and fallback parsing."""
    assert as_list('["a", "b"]') == ["a", "b"]
    assert as_list("a, b") == ["a", "b"]
    assert as_list(None) == []
    assert as_bool("1") is True
    assert as_bool("true") is True
    assert as_bool("0") is False
    assert as_int("4", 9) == 4
    assert as_int("x", 9) == 9
    assert as_float("1.5", 2.0) == 1.5
    assert as_float("x", 2.0) == 2.0


def test_dynamic_settings_validate_runtime_updates():
    """Runtime update validation should split safe, restart-only, and unknown keys."""
    settings = DynamicSettings()
    validation = settings.validate_runtime_updates(
        {
            "MAX_TOKENS_DEFAULT": "1024",
            "OLLAMA_HOST": "http://localhost:11434",
            "CUSTOM_KEY": "x",
        }
    )
    assert validation["runtime_safe"] == ["MAX_TOKENS_DEFAULT"]
    assert validation["requires_restart"] == ["OLLAMA_HOST"]
    assert validation["unknown"] == ["CUSTOM_KEY"]
