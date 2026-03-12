"""Administrative and A/B testing endpoints."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, Header, HTTPException

from ..ab_testing import ExperimentCreateRequest, ExperimentStatus, get_ab_test_manager
from ..api.deps import require_admin
from ..reliability import get_cascade_detector, get_circuit_breaker_manager
from ..runtime_state import reset_runtime_state
from ..settings_dynamic import settings

router = APIRouter()


@router.get("/admin/settings", tags=["Admin"])
def get_settings(x_admin_token: Optional[str] = Header(None)):
    """Get current dynamic settings snapshot."""
    require_admin(x_admin_token)
    return settings.snapshot()


@router.get("/admin/settings/catalog", tags=["Admin"])
def get_settings_catalog(x_admin_token: Optional[str] = Header(None)):
    """Expose dynamic settings metadata for operational tooling."""
    require_admin(x_admin_token)
    return {
        "settings": {
            key: settings.metadata(key)
            for key in settings.keys()
        }
    }


@router.put("/admin/settings", tags=["Admin"])
def update_settings(payload: Dict[str, Any], x_admin_token: Optional[str] = Header(None)):
    """Update dynamic settings."""
    require_admin(x_admin_token)
    validation = settings.validate_runtime_updates(payload)
    if validation["unknown"]:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "unknown_settings",
                "keys": validation["unknown"],
            },
        )
    if validation["requires_restart"]:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "requires_restart",
                "keys": validation["requires_restart"],
            },
        )
    for key, value in payload.items():
        serialized = value if isinstance(value, str) else json.dumps(value) if isinstance(value, (list, dict)) else str(value)
        settings.set(key, serialized, actor="api", source="admin")
    return {"status": "updated", "applied": validation["runtime_safe"]}


@router.get("/admin/circuit-breakers", tags=["Admin"])
def get_circuit_breakers(x_admin_token: Optional[str] = Header(None)):
    """Get status of all circuit breakers."""
    require_admin(x_admin_token)
    manager = get_circuit_breaker_manager()
    return {"circuit_breakers": manager.get_all_statuses(), "timestamp": time.time()}


@router.post("/admin/circuit-breakers/{model_name}/reset", tags=["Admin"])
def reset_circuit_breaker(model_name: str, x_admin_token: Optional[str] = Header(None)):
    """Reset a specific circuit breaker."""
    require_admin(x_admin_token)
    manager = get_circuit_breaker_manager()
    success = manager.reset_breaker(model_name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Circuit breaker for '{model_name}' not found")
    return {"status": "reset", "model": model_name}


@router.get("/admin/cascade-status", tags=["Admin"])
def get_cascade_status(x_admin_token: Optional[str] = Header(None)):
    """Get cascade failure detection status."""
    require_admin(x_admin_token)
    detector = get_cascade_detector()
    return detector.get_status()


@router.post("/admin/runtime/reset", tags=["Admin"])
def reset_runtime(x_admin_token: Optional[str] = Header(None)):
    """Reset internal runtime/singleton state for operational recovery."""
    require_admin(x_admin_token)
    reset_runtime_state()
    return {"status": "reset"}


@router.get("/admin/experiments", tags=["A/B Testing"])
def list_experiments(status: Optional[str] = None, x_admin_token: Optional[str] = Header(None)):
    """List all A/B experiments."""
    require_admin(x_admin_token)
    if not settings.AB_TESTING_ENABLED:
        return {"error": "A/B testing is disabled", "experiments": []}
    manager = get_ab_test_manager()
    status_filter = ExperimentStatus(status) if status else None
    experiments = manager.list_experiments(status=status_filter)
    return {"experiments": [exp.to_dict() for exp in experiments], "total": len(experiments)}


@router.post("/admin/experiments", tags=["A/B Testing"])
def create_experiment(request: ExperimentCreateRequest, x_admin_token: Optional[str] = Header(None)):
    """Create a new A/B experiment."""
    require_admin(x_admin_token)
    if not settings.AB_TESTING_ENABLED:
        raise HTTPException(status_code=400, detail="A/B testing is disabled")
    try:
        manager = get_ab_test_manager()
        experiment = manager.create_experiment(request)
        return {"status": "created", "experiment": experiment.to_dict()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/admin/experiments/{experiment_id}", tags=["A/B Testing"])
def get_experiment(experiment_id: str, x_admin_token: Optional[str] = Header(None)):
    """Get a specific experiment."""
    require_admin(x_admin_token)
    manager = get_ab_test_manager()
    experiment = manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail=f"Experiment not found: {experiment_id}")
    return experiment.to_dict()


def _change_experiment_state(action: str, experiment_id: str, x_admin_token: Optional[str]):
    require_admin(x_admin_token)
    manager = get_ab_test_manager()
    try:
        experiment = getattr(manager, f"{action}_experiment")(experiment_id)
        return {"status": action.replace("_", ""), "experiment": experiment.to_dict()}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Experiment not found: {experiment_id}") from exc


@router.post("/admin/experiments/{experiment_id}/start", tags=["A/B Testing"])
def start_experiment(experiment_id: str, x_admin_token: Optional[str] = Header(None)):
    """Start an experiment."""
    return _change_experiment_state("start", experiment_id, x_admin_token)


@router.post("/admin/experiments/{experiment_id}/pause", tags=["A/B Testing"])
def pause_experiment(experiment_id: str, x_admin_token: Optional[str] = Header(None)):
    """Pause an experiment."""
    return _change_experiment_state("pause", experiment_id, x_admin_token)


@router.post("/admin/experiments/{experiment_id}/complete", tags=["A/B Testing"])
def complete_experiment(experiment_id: str, x_admin_token: Optional[str] = Header(None)):
    """Complete an experiment."""
    return _change_experiment_state("complete", experiment_id, x_admin_token)


@router.get("/admin/experiments/{experiment_id}/results", tags=["A/B Testing"])
def get_experiment_results(experiment_id: str, x_admin_token: Optional[str] = Header(None)):
    """Get aggregated results for an experiment."""
    require_admin(x_admin_token)
    manager = get_ab_test_manager()
    return manager.get_experiment_results(experiment_id)


@router.delete("/admin/experiments/{experiment_id}", tags=["A/B Testing"])
def delete_experiment(experiment_id: str, x_admin_token: Optional[str] = Header(None)):
    """Delete an experiment."""
    require_admin(x_admin_token)
    manager = get_ab_test_manager()
    success = manager.delete_experiment(experiment_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Experiment not found: {experiment_id}")
    return {"status": "deleted", "experiment_id": experiment_id}
