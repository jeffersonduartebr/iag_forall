# Objective: Admin model management API routes.
"""On-the-fly model candidate and health management."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import text

from ..api.admin_auth_routes import resolve_admin_session
from ..db import get_engine
from ..model_registry import filter_configured_model_names, get_model_registry, is_model_configured
from ..openrouter_catalog import (
    fetch_openrouter_models,
    get_openrouter_api_key,
    invalidate_openrouter_catalog_cache,
    mask_openrouter_api_key,
)
from ..openrouter_exploration_modes import EXPLORATION_MODE_PRESETS, apply_mode_preset
from ..openrouter_explorer import (
    blocklist_model,
    get_exploration_status,
    unblocklist_model,
)
from ..providers_async import is_provider_temporarily_unavailable
from ..reliability import get_circuit_breaker_manager
from ..settings_dynamic import settings

router = APIRouter()


class ModelCandidatesUpdate(BaseModel):
    text: List[str] = Field(default_factory=list)
    vision: List[str] = Field(default_factory=list)
    multimodal: List[str] = Field(default_factory=list)


@router.get("/admin/models", tags=["AdminModels"])
def list_models(
    x_admin_token: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """List registry models and active candidate lists."""
    resolve_admin_session(x_admin_token=x_admin_token, authorization=authorization)
    registry = get_model_registry()
    registered = [
        {
            "full_name": m.full_name,
            "display_name": m.display_name,
            "provider": m.provider.value,
            "supports_vision": m.supports_vision,
            "priority": m.priority,
            "configured": is_model_configured(m.full_name),
        }
        for m in registry.list_models(configured_only=False)
    ]
    candidates = {
        "text": list(settings.CANDIDATE_MODELS_LIST or []),
        "vision": list(settings.CANDIDATE_VISION_MODELS_LIST or []),
        "multimodal": list(settings.CANDIDATE_MULTIMODAL_MODELS_LIST or []),
    }
    configured = filter_configured_model_names(candidates["text"] + candidates["vision"] + candidates["multimodal"])
    return {"registered": registered, "candidates": candidates, "configured_candidates": configured}


@router.put("/admin/models/candidates", tags=["AdminModels"])
def update_model_candidates(
    payload: ModelCandidatesUpdate,
    x_admin_token: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """Update runtime candidate model lists."""
    resolve_admin_session(x_admin_token=x_admin_token, authorization=authorization)
    updates: Dict[str, Any] = {}
    if payload.text is not None:
        settings.set("CANDIDATE_MODELS_LIST", payload.text)  # type: ignore[arg-type]  # DynamicSettings.set aceita list em runtime (idem openrouter_explorer)
        updates["CANDIDATE_MODELS_LIST"] = payload.text
    if payload.vision is not None:
        settings.set("CANDIDATE_VISION_MODELS_LIST", payload.vision)  # type: ignore[arg-type]  # DynamicSettings.set aceita list em runtime (idem openrouter_explorer)
        updates["CANDIDATE_VISION_MODELS_LIST"] = payload.vision
    if payload.multimodal is not None:
        settings.set("CANDIDATE_MULTIMODAL_MODELS_LIST", payload.multimodal)  # type: ignore[arg-type]  # DynamicSettings.set aceita list em runtime (idem openrouter_explorer)
        updates["CANDIDATE_MULTIMODAL_MODELS_LIST"] = payload.multimodal
    return {"status": "updated", "updates": updates}


@router.get("/admin/models/openrouter", tags=["AdminModels"])
async def list_openrouter_models(
    refresh: bool = False,
    x_admin_token: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """List models available via OpenRouter (cached catalog)."""
    resolve_admin_session(x_admin_token=x_admin_token, authorization=authorization)
    models = await fetch_openrouter_models(force_refresh=refresh)
    return {"items": models, "count": len(models)}


@router.post("/admin/models/openrouter/refresh", tags=["AdminModels"])
async def refresh_openrouter_catalog(
    x_admin_token: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """Force-refresh the OpenRouter model catalog cache."""
    resolve_admin_session(x_admin_token=x_admin_token, authorization=authorization)
    invalidate_openrouter_catalog_cache()
    models = await fetch_openrouter_models(force_refresh=True)
    return {"status": "refreshed", "count": len(models)}


class OpenRouterCredentialsUpdate(BaseModel):
    api_key: str = Field(..., min_length=8)


@router.get("/admin/models/openrouter/credentials", tags=["AdminModels"])
def get_openrouter_credentials(
    x_admin_token: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """Return whether OpenRouter credentials are configured (masked)."""
    resolve_admin_session(x_admin_token=x_admin_token, authorization=authorization)
    key = get_openrouter_api_key()
    return {
        "configured": bool(key),
        "masked_key": mask_openrouter_api_key(key),
        "source": "settings" if key else "none",
    }


@router.put("/admin/models/openrouter/credentials", tags=["AdminModels"])
def update_openrouter_credentials(
    payload: OpenRouterCredentialsUpdate,
    x_admin_token: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """Persist OpenRouter API key for runtime use (DB/Redis + cache invalidation)."""
    resolve_admin_session(x_admin_token=x_admin_token, authorization=authorization)
    api_key = payload.api_key.strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="api_key is required")

    settings.set("OPENROUTER_API_KEY", api_key, actor="admin", source="admin_ui")
    invalidate_openrouter_catalog_cache()

    from ..providers_async import ProviderFactory

    ProviderFactory._instances.pop("openrouter", None)

    return {
        "status": "updated",
        "configured": True,
        "masked_key": mask_openrouter_api_key(api_key),
    }


@router.get("/admin/models/openrouter/exploration", tags=["AdminModels"])
async def openrouter_exploration_status(
    x_admin_token: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """Return OpenRouter exploration stats, suggestions, and daily usage."""
    resolve_admin_session(x_admin_token=x_admin_token, authorization=authorization)
    return await get_exploration_status(settings)


class OpenRouterExplorationUpdate(BaseModel):
    enabled: Optional[bool] = None
    mode: Optional[str] = None
    rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_per_day: Optional[int] = Field(default=None, ge=1)
    max_usd_per_day: Optional[float] = Field(default=None, ge=0.0)
    max_price_prompt_per_1k: Optional[float] = Field(default=None, ge=0.0)
    max_price_completion_per_1k: Optional[float] = Field(default=None, ge=0.0)
    promote_min_samples: Optional[int] = Field(default=None, ge=1)
    promote_min_reward: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    promote_max_latency_s: Optional[float] = Field(default=None, ge=1.0)
    promote_max_cost_usd_per_1k: Optional[float] = Field(default=None, ge=0.0)
    promote_max_failure_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    provider_allowlist: Optional[List[str]] = None
    pool_size: Optional[int] = Field(default=None, ge=5)
    pool_cache_ttl_s: Optional[int] = Field(default=None, ge=60)
    adaptive_rate_enabled: Optional[bool] = None
    auto_promote_enabled: Optional[bool] = None
    shadow_compare_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    consecutive_failure_block: Optional[int] = Field(default=None, ge=1)


class OpenRouterExplorationModeUpdate(BaseModel):
    mode: str = Field(..., min_length=3)


class OpenRouterExplorationBlocklistUpdate(BaseModel):
    model: str = Field(..., min_length=8)


@router.put("/admin/models/openrouter/exploration", tags=["AdminModels"])
def update_openrouter_exploration(
    payload: OpenRouterExplorationUpdate,
    x_admin_token: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """Update OpenRouter exploration settings at runtime."""
    resolve_admin_session(x_admin_token=x_admin_token, authorization=authorization)
    applied: Dict[str, Any] = {}

    if payload.enabled is not None:
        value = "1" if payload.enabled else "0"
        settings.set("OPENROUTER_EXPLORATION_ENABLED", value)
        applied["OPENROUTER_EXPLORATION_ENABLED"] = value

    if payload.mode is not None:
        preset = apply_mode_preset(payload.mode)
        for key, value in preset.items():
            settings.set(key, value)
            applied[key] = value
        settings.set("OPENROUTER_EXPLORATION_MODE", payload.mode.strip().lower())
        applied["OPENROUTER_EXPLORATION_MODE"] = payload.mode.strip().lower()

    bool_fields = {
        "adaptive_rate_enabled": "OPENROUTER_EXPLORATION_ADAPTIVE_RATE_ENABLED",
        "auto_promote_enabled": "OPENROUTER_EXPLORATION_AUTO_PROMOTE_ENABLED",
    }
    payload_dict = payload.model_dump(exclude_none=True)
    for key, setting_key in bool_fields.items():
        if key in payload_dict:
            value = "1" if payload_dict[key] else "0"
            settings.set(setting_key, value)
            applied[setting_key] = value

    field_map = {
        "rate": ("OPENROUTER_EXPLORATION_RATE", lambda v: str(v)),
        "max_per_day": ("OPENROUTER_EXPLORATION_MAX_PER_DAY", lambda v: str(int(v))),
        "max_usd_per_day": ("OPENROUTER_EXPLORATION_MAX_USD_PER_DAY", lambda v: str(v)),
        "max_price_prompt_per_1k": ("OPENROUTER_EXPLORATION_MAX_PRICE_PROMPT_1K", lambda v: str(v)),
        "max_price_completion_per_1k": ("OPENROUTER_EXPLORATION_MAX_PRICE_COMPLETION_1K", lambda v: str(v)),
        "promote_min_samples": ("OPENROUTER_EXPLORATION_PROMOTE_MIN_SAMPLES", lambda v: str(int(v))),
        "promote_min_reward": ("OPENROUTER_EXPLORATION_PROMOTE_MIN_REWARD", lambda v: str(v)),
        "promote_max_latency_s": ("OPENROUTER_EXPLORATION_PROMOTE_MAX_LATENCY_S", lambda v: str(v)),
        "promote_max_cost_usd_per_1k": ("OPENROUTER_EXPLORATION_PROMOTE_MAX_COST_USD_PER_1K", lambda v: str(v)),
        "promote_max_failure_rate": ("OPENROUTER_EXPLORATION_PROMOTE_MAX_FAILURE_RATE", lambda v: str(v)),
        "pool_size": ("OPENROUTER_EXPLORATION_POOL_SIZE", lambda v: str(int(v))),
        "pool_cache_ttl_s": ("OPENROUTER_EXPLORATION_POOL_CACHE_TTL_S", lambda v: str(int(v))),
        "shadow_compare_rate": ("OPENROUTER_EXPLORATION_SHADOW_COMPARE_RATE", lambda v: str(v)),
        "consecutive_failure_block": ("OPENROUTER_EXPLORATION_CONSECUTIVE_FAILURE_BLOCK", lambda v: str(int(v))),
    }
    for key, (setting_key, fmt) in field_map.items():
        if key in payload_dict:
            value = fmt(payload_dict[key])
            settings.set(setting_key, value)
            applied[setting_key] = value

    if payload.provider_allowlist is not None:
        value = json.dumps(payload.provider_allowlist)
        settings.set("OPENROUTER_EXPLORATION_PROVIDER_ALLOWLIST", value)
        applied["OPENROUTER_EXPLORATION_PROVIDER_ALLOWLIST"] = payload.provider_allowlist

    if not applied:
        raise HTTPException(status_code=400, detail="Nenhum campo de exploração informado.")

    return {"status": "updated", "applied": applied}


@router.get("/admin/models/openrouter/exploration/modes", tags=["AdminModels"])
def openrouter_exploration_modes(
    x_admin_token: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """List available exploration mode presets."""
    resolve_admin_session(x_admin_token=x_admin_token, authorization=authorization)
    return {"modes": list(EXPLORATION_MODE_PRESETS.keys())}


@router.put("/admin/models/openrouter/exploration/mode", tags=["AdminModels"])
def set_openrouter_exploration_mode(
    payload: OpenRouterExplorationModeUpdate,
    x_admin_token: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """Apply a named exploration mode preset."""
    resolve_admin_session(x_admin_token=x_admin_token, authorization=authorization)
    try:
        preset = apply_mode_preset(payload.mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    for key, value in preset.items():
        settings.set(key, value)
    settings.set("OPENROUTER_EXPLORATION_MODE", payload.mode.strip().lower())
    return {"status": "updated", "mode": payload.mode.strip().lower(), "applied": preset}


@router.post("/admin/models/openrouter/exploration/blocklist", tags=["AdminModels"])
async def add_openrouter_exploration_blocklist(
    payload: OpenRouterExplorationBlocklistUpdate,
    x_admin_token: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """Block a model from the exploration pool."""
    resolve_admin_session(x_admin_token=x_admin_token, authorization=authorization)
    await blocklist_model(payload.model.strip())
    return {"status": "blocked", "model": payload.model.strip()}


@router.delete("/admin/models/openrouter/exploration/blocklist", tags=["AdminModels"])
async def remove_openrouter_exploration_blocklist(
    model: str,
    x_admin_token: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """Remove a model from the exploration blocklist."""
    resolve_admin_session(x_admin_token=x_admin_token, authorization=authorization)
    await unblocklist_model(model.strip())
    return {"status": "unblocked", "model": model.strip()}


@router.get("/admin/models/health", tags=["AdminModels"])
def models_health(
    x_admin_token: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """Return circuit breaker and availability status per model."""
    resolve_admin_session(x_admin_token=x_admin_token, authorization=authorization)
    breakers_list = get_circuit_breaker_manager().get_all_statuses()
    breaker_map = {b.get("model"): b for b in breakers_list if b.get("model")}
    models = set(settings.CANDIDATE_MODELS_LIST or [])
    models.update(settings.CANDIDATE_VISION_MODELS_LIST or [])
    models.update(settings.CANDIDATE_MULTIMODAL_MODELS_LIST or [])
    items = []
    for model in sorted(models):
        st = breaker_map.get(model, {})
        items.append(
            {
                "model": model,
                "circuit_state": st.get("state", "unknown"),
                "failures": st.get("fail_counter", 0),
                "temporarily_unavailable": is_provider_temporarily_unavailable(model),
                "configured": is_model_configured(model),
            }
        )
    return {"items": items}


@router.get("/admin/models/pricing", tags=["AdminModels"])
def models_pricing(
    x_admin_token: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """Read model pricing table (read-only MVP)."""
    resolve_admin_session(x_admin_token=x_admin_token, authorization=authorization)
    try:
        with get_engine().connect() as conn:
            rows = conn.execute(text("SELECT model, cost_input_1k, cost_output_1k FROM model_pricing ORDER BY model")).mappings().all()
        return {"items": [dict(r) for r in rows]}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Pricing unavailable: {exc}") from exc
