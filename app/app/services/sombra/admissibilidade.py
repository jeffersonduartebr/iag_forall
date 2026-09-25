# Objective: Decide, before any call, whether a configuration may take part in the shadow (R6).
"""Admissible = provider prefix in ``SHADOW_PROVIDER_ALLOWLIST``; when ``SHADOW_REQUIRED_CLOUD_REGION`` is set, a
cloud model must have a *verifiable* region equal to it (read from the effective client configuration, not from the
model name); the model must support the request modality. Everything else is refused with a recorded reason.
"""

from __future__ import annotations

from typing import Optional

from .config import ConfigSombra

LOCAL = "local"


def regiao_efetiva(model: str) -> Optional[str]:
    """Region where a call to ``model`` is served, from the live client config; ``None`` when unknowable.

    Gemini goes through the google-genai client: with Vertex AI (``vertexai=True``) the location is the one the
    client was built with (``global`` or a region); with an AI Studio key there is no region. OpenRouter routes to
    upstream providers without a region guarantee. Local models run on our own hosts.
    """
    if model.startswith("ollama/"):
        return LOCAL
    if not model.startswith("gemini/"):
        return None
    try:
        from app.providers._gemini import _cliente_genai

        api = getattr(_cliente_genai(), "_api_client", None)
    except Exception:
        return None
    if api is None or not getattr(api, "vertexai", False):
        return None
    return str(getattr(api, "location", None) or "global")


def _suporta_modalidade(model: str, modalidade: str) -> bool:
    if modalidade not in ("vision", "multimodal"):
        return True
    from app.settings_dynamic import settings

    visao = list(getattr(settings, "CANDIDATE_VISION_MODELS_LIST", []) or [])
    return model in visao


def motivo_recusa(model: str, cfg: ConfigSombra, modalidade: str = "text") -> Optional[str]:
    """``None`` when admissible; otherwise the status recorded for the refused configuration."""
    if not any(model.startswith(prefixo) for prefixo in cfg.provedores):
        return "recusado_allowlist"
    if cfg.regiao and not model.startswith("ollama/") and regiao_efetiva(model) != cfg.regiao:
        return "recusado_regiao"
    if not _suporta_modalidade(model, modalidade):
        return "recusado_modalidade"
    return None
