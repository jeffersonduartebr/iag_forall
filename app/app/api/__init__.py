"""API routers and dependencies for the FastAPI application."""

from .admin_routes import router as admin_router
from .eval_routes import router as eval_router
from .feedback_routes import router as feedback_router
from .governance_routes import router as governance_router
from .ops_routes import router as ops_router

__all__ = ["admin_router", "eval_router", "feedback_router", "governance_router", "ops_router"]
