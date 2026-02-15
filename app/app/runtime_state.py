# -*- coding: utf-8 -*-
"""Global runtime state reset helpers for tests/dev."""

from __future__ import annotations

from .providers_async import reset_provider_runtime_state
from .reliability import reset_reliability_runtime_state
from .router_core import reset_router_runtime_state
from .bandits import reset_bandits_runtime_state
from .vectorstore import reset_vectorstore_runtime_state


def reset_runtime_state() -> None:
    """Reset all known global runtime/singleton states."""
    reset_provider_runtime_state()
    reset_reliability_runtime_state()
    reset_router_runtime_state()
    reset_bandits_runtime_state()
    reset_vectorstore_runtime_state()
