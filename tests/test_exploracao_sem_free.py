# Objective: OpenRouter ":free" variants never enter the exploration pool (rate-limited, data-retaining).
from __future__ import annotations

from app import openrouter_explorer as ore


def test_free_variants_are_never_explored(monkeypatch):
    """Production 2026-09-25: an exploration landed on z-ai/glm-5.2:free and hit 429 in a loop."""
    monkeypatch.setattr(ore, "_provider_allowed", lambda slug, cfg: True)
    monkeypatch.setattr(ore, "_price_within_budget", lambda slug, cfg: True)
    catalogo = [{"full_name": "openrouter/z-ai/glm-5.2:free"}, {"full_name": "openrouter/z-ai/glm-5.2"}]
    assert ore._eligible_catalog_models(catalogo, set(), set(), cfg=None) == ["openrouter/z-ai/glm-5.2"]
