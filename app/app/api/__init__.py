# Objective: Package initialization and import surface for api.
"""API routers and dependencies for the FastAPI application."""

from __future__ import annotations

from typing import Any

__all__ = [
    "admin_router",
    "admin_auth_router",
    "admin_dashboard_router",
    "admin_models_router",
    "eval_router",
    "expert_router",
    "feedback_router",
    "governance_router",
    "ops_router",
]


def __getattr__(name: str) -> Any:
    """Lazy-load routers to avoid heavy imports during unit tests."""
    if name == "admin_router":
        from .admin_routes import router as admin_router

        return admin_router
    if name == "admin_auth_router":
        from .admin_auth_routes import router as admin_auth_router

        return admin_auth_router
    if name == "admin_dashboard_router":
        from .admin_dashboard_routes import router as admin_dashboard_router

        return admin_dashboard_router
    if name == "admin_models_router":
        from .admin_models_routes import router as admin_models_router

        return admin_models_router
    if name == "eval_router":
        from .eval_routes import router as eval_router

        return eval_router
    if name == "expert_router":
        from .expert_routes import router as expert_router

        return expert_router
    if name == "feedback_router":
        from .feedback_routes import router as feedback_router

        return feedback_router
    if name == "governance_router":
        from .governance_routes import router as governance_router

        return governance_router
    if name == "ops_router":
        from .ops_routes import router as ops_router

        return ops_router
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
