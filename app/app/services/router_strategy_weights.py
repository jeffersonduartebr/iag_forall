# Objective: NSGA strategy weight resolution for routing decisions.
"""Dynamic NSGA objective weights loaded from settings."""

from __future__ import annotations

from typing import Any, Dict


def _resolve_settings_getter():
    from ..settings_dynamic import settings

    getter = getattr(settings, "get", None)
    if callable(getter):
        return getter
    return lambda name, default: default


def _weights_from_getter(settings_getter, settings_obj: Any = None) -> Dict[str, float]:
    obj = settings_obj

    def _safe_attr(name: str, default: float) -> float:
        if obj is not None:
            try:
                return float(getattr(obj, name))
            except Exception:
                pass
        try:
            return float(settings_getter(name, default))
        except Exception:
            return float(default)

    return {
        "w_quality": _safe_attr("NSGA_W_QUALITY", 1.0),
        "w_latency": _safe_attr("NSGA_W_LATENCY", 0.5),
        "w_cost": _safe_attr("NSGA_W_COST", 100.0),
    }


def get_dynamic_strategy_weights(modality: str) -> Dict[str, float]:
    """Return NSGA strategy weights for one modality."""
    _ = modality
    from ..settings_dynamic import settings

    return _weights_from_getter(_resolve_settings_getter(), settings_obj=settings)


async def get_dynamic_strategy_weights_async(modality: str) -> Dict[str, float]:
    """Load NSGA strategy weights without blocking on Redis settings I/O."""
    _ = modality
    from ..settings_dynamic import settings

    async_getter = getattr(settings, "get_async", None)
    if not callable(async_getter):
        return get_dynamic_strategy_weights(modality)

    async def _safe_attr(name: str, default: float) -> float:
        try:
            return float(getattr(settings, name))
        except Exception:
            pass
        try:
            return float(await async_getter(name, default))
        except Exception:
            return float(default)

    return {
        "w_quality": await _safe_attr("NSGA_W_QUALITY", 1.0),
        "w_latency": await _safe_attr("NSGA_W_LATENCY", 0.5),
        "w_cost": await _safe_attr("NSGA_W_COST", 100.0),
    }
