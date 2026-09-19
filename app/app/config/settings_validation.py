# Objective: Validation of the critical runtime settings checked at startup.
"""Checks for timeouts, NSGA weights and production hardening.

Extracted from ``settings_dynamic`` (which keeps ``validate_critical_settings``
bound to the global settings object).
"""

from __future__ import annotations

from typing import Any, Callable, List

from .settings_types import as_float, as_int

_TRUTHY = {"1", "true", "yes", "on"}


def _read_setting(cfg: Any, name: str, default: Any) -> Any:
    """Attribute first (typed property), then ``cfg.get``; the default when both fail."""
    try:
        if hasattr(cfg, name):
            return getattr(cfg, name)
    except Exception:
        pass
    try:
        getter = getattr(cfg, "get", None)
        if callable(getter):
            return getter(name, default)
    except Exception:
        pass
    return default


def _timeout_errors(read: Callable[[str, Any], Any]) -> List[str]:
    min_timeout = as_int(read("MIN_TIMEOUT", 30), 30)
    max_timeout = as_int(read("MAX_TIMEOUT", 1200), 1200)
    errors = []
    if min_timeout <= 0:
        errors.append("MIN_TIMEOUT must be > 0")
    if max_timeout <= 0:
        errors.append("MAX_TIMEOUT must be > 0")
    if min_timeout > max_timeout:
        errors.append("MIN_TIMEOUT must be <= MAX_TIMEOUT")
    return errors


def _nsga_weight_errors(read: Callable[[str, Any], Any]) -> List[str]:
    weights = (
        as_float(read("NSGA_W_QUALITY", 1.0), 1.0),
        as_float(read("NSGA_W_LATENCY", 0.5), 0.5),
        as_float(read("NSGA_W_COST", 100.0), 100.0),
    )
    errors = []
    if min(weights) < 0:
        errors.append("NSGA weights must be non-negative")
    if sum(weights) <= 0:
        errors.append("NSGA weights sum must be > 0")
    return errors


def _production_errors(read: Callable[[str, Any], Any]) -> List[str]:
    if str(read("ENV", "development") or "development").lower() not in {"production", "prod"}:
        return []

    def _text(name: str) -> str:
        return str(read(name, "") or "").strip()

    errors = []
    if str(read("REQUIRE_API_AUTH", "0")).strip().lower() not in _TRUTHY:
        errors.append("REQUIRE_API_AUTH must be enabled in production")
    if not _text("API_KEYS") and not _text("JWT_SECRET"):
        errors.append("production requires API_KEYS or JWT_SECRET")
    if not _text("METRICS_TOKEN"):
        errors.append("METRICS_TOKEN must be set in production")
    if str(read("ROADMAP_AUTO_DDL", "0")).strip().lower() in _TRUTHY:
        errors.append("ROADMAP_AUTO_DDL must be disabled in production")
    return errors


def validate_critical_settings(cfg: Any) -> List[str]:
    """
    Validate critical runtime settings.
    Returns a list of validation errors (empty when valid).
    """

    def read(name: str, default: Any) -> Any:
        return _read_setting(cfg, name, default)

    errors: List[str] = []
    for check, fallback in (
        (_timeout_errors, "Timeout settings are invalid"),
        (_nsga_weight_errors, "NSGA weights are invalid"),
    ):
        try:
            errors += check(read)
        except Exception:
            errors.append(fallback)
    return errors + _production_errors(read)
