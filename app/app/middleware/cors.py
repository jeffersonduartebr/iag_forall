# Objective: Resolve the allowed CORS origins per request instead of at import.
"""CORS whose allowed origins can change without restarting the process.

Starlette's ``CORSMiddleware`` reads ``allow_origins`` once, in its
constructor, and the middleware stack is built while ``main`` is being
imported. ``ADMIN_UI_CORS_ORIGINS`` was therefore frozen at boot: changing it
through ``/admin/settings`` persisted the value, invalidated the caches,
published the reload — and had no effect whatsoever until someone restarted the
container. The catalog marked it ``requires_restart``, which was honest but
also an admission.

Only one decision actually depends on the list: ``is_allowed_origin``. Starlette
calls it on every response when the middleware was built with an explicit
origin list, for both the simple and the preflight paths. Overriding that one
method is enough, and it keeps everything else — ``Vary: Origin``, the
credentials handling, the preflight header precomputation — exactly as the
library wrote it.

The base class is deliberately constructed with an **empty** origin list. That
makes ``allow_all_origins`` false, which is what routes every request through
``is_allowed_origin`` and sets ``Vary: Origin`` — the header that stops a shared
cache from serving one origin's response to another. A dynamic allowlist
without it would be a cache-poisoning bug.
"""

from __future__ import annotations

import logging
from typing import Sequence

from starlette.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

SETTING_KEY = "ADMIN_UI_CORS_ORIGINS"


def parse_origins(raw: object) -> list[str]:
    """Split a comma-separated origin list, dropping blanks."""
    return [origin.strip() for origin in str(raw or "").split(",") if origin.strip()]


class DynamicCORSMiddleware(CORSMiddleware):
    """``CORSMiddleware`` that re-reads its allowlist on every request."""

    def __init__(self, app: ASGIApp, *, fallback_origins: Sequence[str] = (), **kwargs) -> None:
        #: Usado quando a leitura da definição falha. É o valor que estava em
        #: vigor no arranque, e não uma lista vazia: um Redis em baixo não pode
        #: transformar-se em "nenhuma origem é permitida", que partiria o
        #: frontend inteiro por causa de uma falha de infraestrutura.
        self._fallback = list(fallback_origins)
        super().__init__(app, allow_origins=[], **kwargs)

    def current_origins(self) -> list[str]:
        """The allowlist as configured right now."""
        from app.settings_dynamic import settings

        try:
            configured = settings.get(SETTING_KEY, None)
        except Exception as exc:
            logger.warning(f"[cors] Falha ao ler {SETTING_KEY}; a usar o valor de arranque: {exc}")
            return self._fallback
        origins = parse_origins(configured)
        return origins or self._fallback

    def is_allowed_origin(self, origin: str) -> bool:
        """The single decision Starlette delegates, resolved per request."""
        allowed = self.current_origins()
        return "*" in allowed or origin in allowed
