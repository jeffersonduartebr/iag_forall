"""Typed coercion helpers for dynamic settings."""

from __future__ import annotations

import json
from typing import Any, List, Optional


def as_list(raw: Optional[str]) -> List[str]:
    """Parse a setting value as a JSON list or comma-separated string."""
    if raw is None:
        return []
    raw = str(raw).strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return [item.strip() for item in raw.split(",") if item.strip()]


def as_bool(value: Any, default: bool = False) -> bool:
    """Coerce a dynamic setting to bool."""
    if value is None:
        return default
    return str(value).strip() in ("1", "true", "True")


def as_int(value: Any, default: int) -> int:
    """Coerce a dynamic setting to int with fallback."""
    try:
        return int(value)
    except Exception:
        return int(default)


def as_float(value: Any, default: float) -> float:
    """Coerce a dynamic setting to float with fallback."""
    try:
        return float(value)
    except Exception:
        return float(default)
