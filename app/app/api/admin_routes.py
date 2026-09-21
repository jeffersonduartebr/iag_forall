# Objective: API layer code for admin routes.
"""Administrative and A/B testing endpoints."""

from __future__ import annotations

import json
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from ..ab_testing import ExperimentCreateRequest, ExperimentStatus, get_ab_test_manager
from ..config.settings_catalog import split_by_security
from ..reliability import get_cascade_detector, get_circuit_breaker_manager
from ..runtime_state import reset_runtime_state
from ..schemas import AdminSettingsUpdateRequest
from ..settings_dynamic import settings
from .dependencies import admin_session, admin_token_only

router = APIRouter(
    # Ao nível do router, não por handler: a linha `_auth(...)` que cada um
    # tinha de repetir era o que tornava possível esquecê-la — e o que fazia a
    # validação do corpo correr antes das credenciais.
    dependencies=[Depends(admin_session)],
)


@router.get("/admin/settings", tags=["Admin"])
def get_settings():
    """Get current dynamic settings snapshot."""
    return settings.snapshot()


@router.get("/admin/settings/catalog", tags=["Admin"])
def get_settings_catalog():
    """Expose dynamic settings metadata for operational tooling."""
    return {"settings": {key: settings.metadata(key) for key in settings.keys()}}


def _apply_settings(updates: dict) -> dict:
    """Validate and persist one batch of settings; shared by both endpoints.

    A divisão entre os dois é de *autorização*, não de comportamento: o que
    conta como conhecido, o que exige reinício e como o valor é serializado é
    exactamente o mesmo dos dois lados.
    """
    validation = settings.validate_runtime_updates(updates)
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
    for key, value in updates.items():
        serialized = (
            value if isinstance(value, str) else json.dumps(value) if isinstance(value, (list, dict)) else str(value)
        )
        settings.set(key, serialized, actor="api", source="admin")
    return {"status": "updated", "applied": validation["runtime_safe"]}


@router.put("/admin/settings", tags=["Admin"])
def update_settings(payload: AdminSettingsUpdateRequest):
    """Update operational settings. Security keys are refused here.

    Este endpoint aceita um token de sessão de admin, e é por isso que o
    browser o pode chamar. As 13 chaves que decidem *quem entra* —
    `REQUIRE_API_AUTH`, `TRUST_HEADER_ROLES`, `JWT_SECRET`,
    `ADMIN_UI_CORS_ORIGINS`, as credenciais — não passam por aqui: quem tiver
    uma sessão de admin podia desligar a autenticação da instalação inteira,
    e uma sessão roubada no browser bastava para isso.
    """
    updates = dict(payload.settings or {})
    operational, security = split_by_security(updates)
    if security:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "security_settings_refused",
                "keys": security,
                "message": "Use PUT /admin/settings/security, que exige o token de admin mestre.",
            },
        )
    return _apply_settings({key: updates[key] for key in operational})


@router.put(
    "/admin/settings/security",
    tags=["Admin"],
    dependencies=[Depends(admin_token_only)],
)
def update_security_settings(payload: AdminSettingsUpdateRequest):
    """Update the settings that decide who may get in.

    Exige o `ADMIN_TOKEN` mestre e **não** aceita a sessão de admin — a mesma
    distinção que as rotas de RBAC já fazem, e pela mesma razão: o browser não
    deve guardar a credencial que permite mudar a política de acesso.

    Recusa chaves operacionais de propósito. Um batch que misturasse as duas
    ou precisaria do token mestre para tudo, ou aplicaria metade — e a segunda
    hipótese é a pior das três.
    """
    updates = dict(payload.settings or {})
    operational, security = split_by_security(updates)
    if operational:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "operational_settings_refused",
                "keys": operational,
                "message": "Use PUT /admin/settings para as chaves operacionais.",
            },
        )
    return _apply_settings({key: updates[key] for key in security})


@router.get("/admin/circuit-breakers", tags=["Admin"])
def get_circuit_breakers():
    """Get status of all circuit breakers."""
    manager = get_circuit_breaker_manager()
    return {"circuit_breakers": manager.get_all_statuses(), "timestamp": time.time()}


@router.post("/admin/circuit-breakers/{model_name}/reset", tags=["Admin"])
def reset_circuit_breaker(
    model_name: str,
):
    """Reset a specific circuit breaker."""
    manager = get_circuit_breaker_manager()
    success = manager.reset_breaker(model_name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Circuit breaker for '{model_name}' not found")
    return {"status": "reset", "model": model_name}


@router.get("/admin/cascade-status", tags=["Admin"])
def get_cascade_status():
    """Get cascade failure detection status."""
    detector = get_cascade_detector()
    return detector.get_status()


@router.post("/admin/runtime/reset", tags=["Admin"])
def reset_runtime():
    """Reset internal runtime/singleton state for operational recovery."""
    reset_runtime_state()
    return {"status": "reset"}


@router.get("/admin/experiments", tags=["A/B Testing"])
def list_experiments(
    status: Optional[str] = None,
):
    """List all A/B experiments."""
    if not settings.AB_TESTING_ENABLED:
        return {"error": "A/B testing is disabled", "experiments": []}
    manager = get_ab_test_manager()
    status_filter = ExperimentStatus(status) if status else None
    experiments = manager.list_experiments(status=status_filter)
    return {"experiments": [exp.to_dict() for exp in experiments], "total": len(experiments)}


@router.post("/admin/experiments", tags=["A/B Testing"])
def create_experiment(
    request: ExperimentCreateRequest,
):
    """Create a new A/B experiment."""
    if not settings.AB_TESTING_ENABLED:
        raise HTTPException(status_code=400, detail="A/B testing is disabled")
    try:
        manager = get_ab_test_manager()
        experiment = manager.create_experiment(request)
        return {"status": "created", "experiment": experiment.to_dict()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/admin/experiments/{experiment_id}", tags=["A/B Testing"])
def get_experiment(
    experiment_id: str,
):
    """Get a specific experiment."""
    manager = get_ab_test_manager()
    experiment = manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail=f"Experiment not found: {experiment_id}")
    return experiment.to_dict()


def _change_experiment_state(action: str, experiment_id: str):
    manager = get_ab_test_manager()
    try:
        experiment = getattr(manager, f"{action}_experiment")(experiment_id)
        return {"status": action.replace("_", ""), "experiment": experiment.to_dict()}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Experiment not found: {experiment_id}") from exc


@router.post("/admin/experiments/{experiment_id}/start", tags=["A/B Testing"])
def start_experiment(
    experiment_id: str,
):
    """Start an experiment."""
    return _change_experiment_state("start", experiment_id)


@router.post("/admin/experiments/{experiment_id}/pause", tags=["A/B Testing"])
def pause_experiment(
    experiment_id: str,
):
    """Pause an experiment."""
    return _change_experiment_state("pause", experiment_id)


@router.post("/admin/experiments/{experiment_id}/complete", tags=["A/B Testing"])
def complete_experiment(
    experiment_id: str,
):
    """Complete an experiment."""
    return _change_experiment_state("complete", experiment_id)


@router.get("/admin/experiments/{experiment_id}/results", tags=["A/B Testing"])
def get_experiment_results(
    experiment_id: str,
):
    """Get aggregated results for an experiment."""
    manager = get_ab_test_manager()
    return manager.get_experiment_results(experiment_id)


@router.delete("/admin/experiments/{experiment_id}", tags=["A/B Testing"])
def delete_experiment(
    experiment_id: str,
):
    """Delete an experiment."""
    manager = get_ab_test_manager()
    success = manager.delete_experiment(experiment_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Experiment not found: {experiment_id}")
    return {"status": "deleted", "experiment_id": experiment_id}
