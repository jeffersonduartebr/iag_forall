# Objective: Freeze routing policy during academic evaluation runs.
"""Frozen policy mode — disable exploration and online tuning during eval."""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)

REDIS_FROZEN_PREFIX = "eval:frozen:"


def _redis_client():
    try:
        from app.utils.redis_client import get_redis

        return get_redis()
    except Exception:
        return None


def build_frozen_snapshot() -> Dict[str, Any]:
    """Capture current routing settings for reproducible eval execution."""
    from app.config.constants import DEFAULT_UNCERTAINTY_THRESHOLD
    from app.settings_dynamic import settings

    return {
        "NSGA_W_QUALITY": float(settings.NSGA_W_QUALITY),
        "NSGA_W_LATENCY": float(settings.NSGA_W_LATENCY),
        "NSGA_W_COST": float(settings.NSGA_W_COST),
        "BANDIT_EPSILON": float(settings.get("BANDIT_EPSILON", 0.0) or 0.0),
        "UNCERTAINTY_THRESHOLD": float(settings.get("UNCERTAINTY_THRESHOLD", DEFAULT_UNCERTAINTY_THRESHOLD)),
        "OPENROUTER_EXPLORATION_ENABLED": bool(settings.get("OPENROUTER_EXPLORATION_ENABLED", False)),
        "OPENROUTER_EXPLORATION_RATE": float(settings.get("OPENROUTER_EXPLORATION_RATE", 0.0) or 0.0),
    }


def activate_frozen_policy(run_id: str, snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Store frozen policy state in Redis for the duration of one eval run."""
    # ``{}`` explícito é um snapshot (vazio), não a ausência dele.
    payload = dict(snapshot if snapshot is not None else build_frozen_snapshot())
    payload["run_id"] = run_id
    payload["active"] = True
    rds = _redis_client()
    if rds:
        try:
            rds.set(f"{REDIS_FROZEN_PREFIX}{run_id}", json.dumps(payload, ensure_ascii=False), ex=86400)
            rds.set(f"{REDIS_FROZEN_PREFIX}active", run_id, ex=86400)
        except Exception as exc:
            logger.debug("[frozen_policy] redis activate skipped: %s", exc)
    return payload


def deactivate_frozen_policy(run_id: str) -> None:
    """Clear frozen policy markers after eval completes."""
    rds = _redis_client()
    if not rds:
        return
    try:
        rds.delete(f"{REDIS_FROZEN_PREFIX}{run_id}")
        active = rds.get(f"{REDIS_FROZEN_PREFIX}active")
        active_text = active.decode() if isinstance(active, bytes) else str(active or "")
        if active_text == run_id:
            rds.delete(f"{REDIS_FROZEN_PREFIX}active")
    except Exception as exc:
        logger.debug("[frozen_policy] redis deactivate skipped: %s", exc)


def get_frozen_policy(run_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Return frozen snapshot for a run or the currently active frozen run."""
    rds = _redis_client()
    if not rds:
        return None
    try:
        if not run_id:
            active = rds.get(f"{REDIS_FROZEN_PREFIX}active")
            run_id = active.decode() if isinstance(active, bytes) else str(active or "")
        if not run_id:
            return None
        raw = rds.get(f"{REDIS_FROZEN_PREFIX}{run_id}")
        if not raw:
            return None
        text = raw.decode() if isinstance(raw, bytes) else str(raw)
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def is_frozen_policy_active(run_id: Optional[str] = None) -> bool:
    """Check whether frozen policy mode is active."""
    return bool(get_frozen_policy(run_id))


def frozen_bandit_epsilon(default: float) -> float:
    """Return zero exploration when frozen policy is active."""
    if is_frozen_policy_active():
        frozen = get_frozen_policy()
        if frozen:
            return 0.0
    return default


def frozen_exploration_enabled(default: bool) -> bool:
    """Disable OpenRouter exploration under frozen policy."""
    return False if is_frozen_policy_active() else default


def frozen_exploration_rate(default: float) -> float:
    """Return zero exploration rate under frozen policy."""
    return 0.0 if is_frozen_policy_active() else default


def should_skip_eval_feedback(metadata: Optional[Dict[str, Any]]) -> bool:
    """Skip online tuning when eval was run under frozen policy."""
    meta = dict(metadata or {})
    if meta.get("frozen_policy"):
        return True
    manifest = meta.get("experiment_manifest") or {}
    return bool(manifest.get("frozen_policy"))


@contextmanager
def frozen_policy_context(run_id: str, snapshot: Optional[Dict[str, Any]] = None) -> Iterator[Dict[str, Any]]:
    """Context manager used by Celery eval tasks."""
    payload = activate_frozen_policy(run_id, snapshot=snapshot)
    try:
        yield payload
    finally:
        deactivate_frozen_policy(run_id)


@contextmanager
def optional_frozen_policy(enabled: bool, prefix: str = "preview") -> Iterator[Optional[Dict[str, Any]]]:
    """Freeze the policy around one ad-hoc call (e.g. the expert preview) when ``enabled``.

    An already active freeze (an eval run) is reused rather than re-activated:
    activating would overwrite its ``active`` marker and deactivating would then
    delete it, unfreezing the eval run for the rest of its execution.
    """
    if not enabled:
        yield None
        return
    current = get_frozen_policy()
    if current:
        yield current
        return
    with frozen_policy_context(f"{prefix}:{uuid.uuid4().hex[:12]}") as payload:
        yield payload
