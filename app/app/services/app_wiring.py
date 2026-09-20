# Objective: Register the middleware stack and the application-wide error handler.
"""Wiring that was inline in ``main.py``.

``docs/SLOC_REFACTOR_ROADMAP.md`` already listed moving the lifespan and the
middleware out of ``main.py``; this is the middleware half. It also keeps the
one thing that must not be forgotten next to the stack it protects: the
application-wide exception handler, without which anything escaping a route
becomes Starlette's bare ``500 {"detail":"Internal Server Error"}`` — no
correlation id, no error class, indistinguishable from a bug in the router.
"""

from __future__ import annotations

from typing import Iterable, Type

from fastapi import FastAPI
from starlette.middleware.gzip import GZipMiddleware

from .error_responses import handle_unexpected


def install_middleware(app: FastAPI, stack: Iterable[Type], *, gzip_min_size: int) -> None:
    """Add the middleware stack, then GZip, then the catch-all handler.

    Order matters and is preserved: the last added runs first, so ``stack`` is
    registered outermost-last exactly as it was written inline.
    """
    for middleware in stack:
        app.add_middleware(middleware)
    app.add_middleware(GZipMiddleware, minimum_size=gzip_min_size)
    app.add_exception_handler(Exception, handle_unexpected)
