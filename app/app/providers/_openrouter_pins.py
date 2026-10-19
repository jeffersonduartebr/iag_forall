# Objective: Pin each OpenRouter model to one provider endpoint, so the served version cannot change mid-window.
"""Provider pin per OpenRouter model (``OPENROUTER_PROVIDER_PINS``).

OpenRouter sends a call to any provider that serves the model, and providers differ in snapshot and
quantization. The field protocol freezes model versions, so a pinned model goes out with ``only: [tag]``
and ``allow_fallbacks: false``: the call fails instead of silently landing on another endpoint, and the
fallback chain of the router takes over (https://openrouter.ai/docs/features/provider-routing).

The tag is the endpoint slug of the catalog (``anthropic``, ``deepinfra/fp4``, ``xiaomi/fp8``), keyed by
the OpenRouter model id without the ``openrouter/`` route (``anthropic/claude-opus-5.5``).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from app.settings_dynamic import settings

logger = logging.getLogger(__name__)


def provider_pins() -> Dict[str, str]:
    """Model id -> provider endpoint tag; an invalid setting pins nothing (and says so)."""
    raw = settings.get("OPENROUTER_PROVIDER_PINS", "{}")
    try:
        data = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
    except (TypeError, ValueError):
        logger.warning("OPENROUTER_PROVIDER_PINS inválido: nenhum provedor fixado")
        return {}
    if not isinstance(data, dict):
        logger.warning("OPENROUTER_PROVIDER_PINS não é um objeto: nenhum provedor fixado")
        return {}
    return {str(k): str(v) for k, v in data.items() if k and v}


def pin_provider(provider: Dict[str, Any], model: str) -> Dict[str, Any]:
    """Restrict ``provider`` to the pinned endpoint of ``model``, with no fallback; unpinned models pass as is."""
    tag = provider_pins().get(model)
    if tag:
        provider["only"] = [tag]
        provider["allow_fallbacks"] = False
    return provider
