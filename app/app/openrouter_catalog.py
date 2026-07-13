# Objective: Fetch and cache OpenRouter model catalog for routing and admin UI.
"""OpenRouter /api/v1/models catalog helper."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
_CACHE: Dict[str, Any] = {"data": [], "expires_at": 0.0}
_CACHE_TTL_S = 3600


def _api_key() -> str:
    return get_openrouter_api_key()


def get_openrouter_api_key() -> str:
    """Resolve OpenRouter API key: dynamic settings (DB/Redis) then env."""
    try:
        from app.settings_dynamic import settings

        value = str(settings.get("OPENROUTER_API_KEY", "") or "").strip()
        if value:
            return value
    except Exception:
        pass
    return (os.getenv("OPENROUTER_API_KEY", "") or "").strip()


def mask_openrouter_api_key(key: str) -> str:
    """Return a masked representation safe for admin UI."""
    text = (key or "").strip()
    if len(text) < 10:
        return ""
    return f"{text[:8]}...{text[-4:]}"


def _headers() -> Dict[str, str]:
    headers: Dict[str, str] = {}
    key = _api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    referer = (os.getenv("OPENROUTER_HTTP_REFERER", "") or "").strip()
    if referer:
        headers["HTTP-Referer"] = referer
    title = (os.getenv("OPENROUTER_APP_NAME", "") or "").strip()
    if title:
        headers["X-Title"] = title
    return headers


async def fetch_openrouter_models(*, force_refresh: bool = False) -> List[Dict[str, Any]]:
    """Return OpenRouter model metadata (cached 1h)."""
    now = time.time()
    if not force_refresh and _CACHE["expires_at"] > now and _CACHE["data"]:
        return list(_CACHE["data"])

    try:
        import httpx

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(OPENROUTER_MODELS_URL, headers=_headers())
            resp.raise_for_status()
            payload = resp.json()
    except Exception as exc:
        logger.warning("[openrouter] catalog fetch failed: %s", exc)
        return list(_CACHE["data"]) if _CACHE["data"] else []

    models = payload.get("data") if isinstance(payload, dict) else []
    if not isinstance(models, list):
        models = []

    normalized: List[Dict[str, Any]] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id") or "").strip()
        if not model_id:
            continue
        pricing = item.get("pricing") or {}
        normalized.append(
            {
                "id": model_id,
                "full_name": f"openrouter/{model_id}",
                "name": model_id,
                "description": item.get("description"),
                "context_length": item.get("context_length"),
                "pricing": pricing,
                "architecture": item.get("architecture"),
                # Sinal autoritativo de capacidades (inclui "tools"/"tool_choice"
                # quando o modelo suporta function calling).
                "supported_parameters": item.get("supported_parameters") or [],
            }
        )

    _CACHE["data"] = normalized
    _CACHE["expires_at"] = now + _CACHE_TTL_S
    return list(normalized)


def openrouter_model_ids(models: Optional[List[Dict[str, Any]]] = None) -> List[str]:
    """Return fully qualified openrouter/* model names."""
    rows = models or []
    return [str(m.get("full_name") or f"openrouter/{m.get('id')}") for m in rows if m.get("id")]


def invalidate_openrouter_catalog_cache() -> None:
    """Clear cached OpenRouter catalog."""
    _CACHE["data"] = []
    _CACHE["expires_at"] = 0.0


def openrouter_supports_tools(model_slug: str) -> bool:
    """Return whether an OpenRouter model advertises tool/function calling.

    Lê o catálogo já em cache (síncrono) e verifica se ``supported_parameters``
    contém ``"tools"``. Retorna ``False`` quando o modelo não está no cache
    (conservador) — o catálogo é populado de forma assíncrona em background.
    """
    slug = (model_slug or "").strip()
    if not slug:
        return False
    for item in _CACHE.get("data") or []:
        if str(item.get("id") or "") != slug:
            continue
        params = item.get("supported_parameters") or []
        try:
            return "tools" in params or "tool_choice" in params
        except TypeError:
            return False
    return False


def get_openrouter_pricing_per_1k(model_slug: str) -> Optional[Dict[str, float]]:
    """Return USD per-1K token pricing from the cached OpenRouter catalog."""
    slug = (model_slug or "").strip()
    if not slug:
        return None
    for item in _CACHE.get("data") or []:
        if str(item.get("id") or "") != slug:
            continue
        pricing = item.get("pricing") or {}
        try:
            prompt = float(pricing.get("prompt") or 0.0)
            completion = float(pricing.get("completion") or 0.0)
        except (TypeError, ValueError):
            return None
        return {"in": prompt * 1000.0, "out": completion * 1000.0}
    return None
