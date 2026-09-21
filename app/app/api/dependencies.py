# Objective: Authorization as a FastAPI dependency, resolved before the handler runs.
"""Admin and role authorization, moved out of the handlers' first line.

Every protected route used to authenticate itself:

    def get_settings(x_admin_token=Header(None), authorization=Header(None)):
        _auth(x_admin_token, authorization)      # ← a linha que não pode faltar
        return settings.snapshot()

That works, and it is wrong in two ways that only show up when you ask who
reaches the line below.

**A route that forgets the call is public.** Nothing in the type system, the
router or the review diff points at it, and static analysis cannot settle it
either — some handlers delegate to a helper that authenticates, so grepping for
the call produces false positives.

**Validation runs before the handler.** FastAPI parses and validates the body,
the query and the path before the first statement executes, so a POST with an
invalid body answered 422 without ever reaching the credentials check. An
anonymous caller could therefore enumerate the required fields of every admin
route.

A dependency fixes both. FastAPI solves dependencies *before* raising the
accumulated validation errors, so an ``HTTPException`` raised here propagates
first — verified: the same route answers 401 as a dependency and 422 as a first
line. And a dependency declared on the router cannot be forgotten by a route
added later, because there is nothing to remember.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from fastapi import Header

from .deps import require_admin_or_role


def admin_session(
    x_admin_token: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
) -> Dict[str, Any]:
    """Admin token or admin JWT. Raises 401 before anything else runs.

    Use it as ``dependencies=[Depends(admin_session)]`` on a router when the
    handler does not need the result, and as a parameter when it does.
    """
    from .admin_auth_routes import resolve_admin_session

    return resolve_admin_session(x_admin_token=x_admin_token, authorization=authorization)


def admin_token_only(x_admin_token: Optional[str] = Header(None)) -> None:
    """The master admin token, deliberately **not** the admin JWT.

    Four routes authenticate with ``require_admin(token)`` and pass no
    ``authorization``, which rejects a JWT. That is narrower than the rest of
    the admin surface and it is on purpose — they are the RBAC primitives and
    the raw feedback statistics — so it gets its own dependency instead of
    being quietly widened to accept a session token.
    """
    from .deps import require_admin

    require_admin(x_admin_token, None)


def require_roles(*roles: str) -> Callable[..., Dict[str, Any]]:
    """Build a dependency that admits the admin token, an admin JWT, or RBAC.

    The roles differ per route — eval, governance and the expert portal each
    have their own — so this is a factory rather than one dependency. The
    returned callable declares the same headers the handlers used to declare
    themselves, which is what lets them stop declaring them.
    """
    required: List[str] = list(roles)

    def _dependency(
        x_admin_token: Optional[str] = Header(None),
        x_user_id: Optional[str] = Header(None),
        x_user_roles: Optional[str] = Header(None),
        authorization: Optional[str] = Header(None),
    ) -> Dict[str, Any]:
        return require_admin_or_role(
            admin_token=x_admin_token,
            user_id=x_user_id,
            user_roles_header=x_user_roles,
            authorization=authorization,
            required_roles=required,
        )

    _dependency.__name__ = "require_" + "_or_".join(required or ["admin"])
    #: Exposto para os testes e para o relatório da superfície: permite
    #: perguntar a uma rota que papéis exige sem ler o corpo do handler.
    _dependency.required_roles = required  # type: ignore[attr-defined]
    return _dependency


#: Conjuntos de papéis usados em mais do que um sítio. Nomeá-los evita que duas
#: rotas do mesmo recurso divirjam por uma lista mal copiada.
EVAL_READER = ("eval_viewer", "eval_admin", "researcher", "platform_admin")
EVAL_WRITER = ("eval_admin", "researcher", "platform_admin")
EXPERT_REVIEWER = ("expert_reviewer", "audit_viewer", "eval_admin", "researcher", "platform_admin")
EXPERT_MANAGER = ("platform_admin", "eval_admin", "governance_admin", "admin")
