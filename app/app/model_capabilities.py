# Objective: Single source of truth for per-model capability checks (tools, vision).
"""Checagem de capacidade por modelo — fonte única de verdade.

Antes esta lógica estava espalhada: marcadores de nome de visão duplicados em
``router_strategy``, e helpers de tool em ``model_registry``. Este módulo
centraliza tudo. ``model_registry`` reexporta estes nomes para compatibilidade.

A resolução usa, nesta ordem: a capacidade declarada no registry e, como
heurística para modelos desconhecidos/locais, marcadores no nome do modelo (mais,
para OpenRouter, o ``supported_parameters`` autoritativo do catálogo).
"""

from __future__ import annotations

from typing import List

# Marcadores de nome usados como heurística quando o modelo não está no registry.
VISION_CAPABLE_MARKERS: tuple[str, ...] = ("vision", "vl", "llava", "moondream", "gpt-4o", "gemini", "claude")
# Modelos estritamente de visão que devem sair do roteamento de texto (VRAM/custo).
VISION_ONLY_MARKERS: tuple[str, ...] = ("vision", "vl", "llava", "moondream")


def model_supports_tools(model_name: str) -> bool:
    """Return whether one model supports tool/function calling.

    1. Capacidade declarada no registry (``Capability.FUNCTION_CALLING``).
    2. Para modelos OpenRouter, o ``supported_parameters`` do catálogo.

    Desconhecidos/locais sem sinal explícito retornam ``False`` (estrito); habilite
    via ``CANDIDATE_TOOL_MODELS_LIST``.
    """
    if not isinstance(model_name, str) or not model_name:
        return False
    from app.model_registry import get_model_registry  # lazy: evita ciclo de import

    cfg = get_model_registry().get(model_name)
    if cfg is not None and cfg.supports_tools:
        return True
    if model_name.startswith("openrouter/"):
        try:
            from app.openrouter_catalog import openrouter_supports_tools

            return bool(openrouter_supports_tools(model_name.split("/", 1)[1]))
        except Exception:
            return False
    return False


def filter_tool_capable_model_names(model_names: List[str]) -> List[str]:
    """Keep only model names that support tool/function calling."""
    return [m for m in model_names if model_supports_tools(m)]


def model_supports_vision(model_name: str) -> bool:
    """Return whether one model accepts image input (registry capability then markers)."""
    if not isinstance(model_name, str) or not model_name:
        return False
    from app.model_registry import get_model_registry  # lazy: evita ciclo de import

    cfg = get_model_registry().get(model_name)
    if cfg is not None and cfg.supports_vision:
        return True
    return any(tag in model_name.lower() for tag in VISION_CAPABLE_MARKERS)


def is_vision_only_model(model_name: str) -> bool:
    """Return whether a model is vision-only (excluído do roteamento de texto)."""
    return isinstance(model_name, str) and any(tag in model_name.lower() for tag in VISION_ONLY_MARKERS)


def filter_vision_capable_model_names(model_names: List[str]) -> List[str]:
    """Keep only model names that can accept image input."""
    return [m for m in model_names if model_supports_vision(m)]
