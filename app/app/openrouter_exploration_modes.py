# Objective: Preset bundles for OpenRouter exploration modes.
"""Exploration mode presets applied on top of base settings."""

from __future__ import annotations

from typing import Any, Dict

EXPLORATION_MODE_PRESETS: Dict[str, Dict[str, Any]] = {
    "conservative": {
        "OPENROUTER_EXPLORATION_RATE": "0.05",
        "OPENROUTER_EXPLORATION_MAX_PER_DAY": "50",
        "OPENROUTER_EXPLORATION_MAX_USD_PER_DAY": "1.0",
        "OPENROUTER_EXPLORATION_POOL_SIZE": "25",
        "OPENROUTER_EXPLORATION_MAX_PRICE_PROMPT_1K": "0.008",
        "OPENROUTER_EXPLORATION_MAX_PRICE_COMPLETION_1K": "0.02",
        "OPENROUTER_EXPLORATION_SHADOW_COMPARE_RATE": "0.03",
    },
    "balanced": {
        "OPENROUTER_EXPLORATION_RATE": "0.10",
        "OPENROUTER_EXPLORATION_MAX_PER_DAY": "100",
        "OPENROUTER_EXPLORATION_MAX_USD_PER_DAY": "3.0",
        "OPENROUTER_EXPLORATION_POOL_SIZE": "40",
        "OPENROUTER_EXPLORATION_MAX_PRICE_PROMPT_1K": "0.01",
        "OPENROUTER_EXPLORATION_MAX_PRICE_COMPLETION_1K": "0.03",
        "OPENROUTER_EXPLORATION_SHADOW_COMPARE_RATE": "0.05",
    },
    "aggressive": {
        "OPENROUTER_EXPLORATION_RATE": "0.20",
        "OPENROUTER_EXPLORATION_MAX_PER_DAY": "250",
        "OPENROUTER_EXPLORATION_MAX_USD_PER_DAY": "10.0",
        "OPENROUTER_EXPLORATION_POOL_SIZE": "60",
        "OPENROUTER_EXPLORATION_MAX_PRICE_PROMPT_1K": "0.02",
        "OPENROUTER_EXPLORATION_MAX_PRICE_COMPLETION_1K": "0.06",
        "OPENROUTER_EXPLORATION_SHADOW_COMPARE_RATE": "0.08",
    },
    "cost_hunt": {
        "OPENROUTER_EXPLORATION_RATE": "0.15",
        "OPENROUTER_EXPLORATION_MAX_PER_DAY": "150",
        "OPENROUTER_EXPLORATION_MAX_USD_PER_DAY": "2.0",
        "OPENROUTER_EXPLORATION_POOL_SIZE": "50",
        "OPENROUTER_EXPLORATION_MAX_PRICE_PROMPT_1K": "0.004",
        "OPENROUTER_EXPLORATION_MAX_PRICE_COMPLETION_1K": "0.012",
        "OPENROUTER_EXPLORATION_SHADOW_COMPARE_RATE": "0.05",
    },
}


def apply_mode_preset(mode: str) -> Dict[str, str]:
    """Return setting key/value pairs for a named exploration mode."""
    key = (mode or "balanced").strip().lower()
    preset = EXPLORATION_MODE_PRESETS.get(key)
    if not preset:
        raise ValueError(f"Unknown exploration mode: {mode}")
    return dict(preset)
