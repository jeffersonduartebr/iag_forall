# Objective: Quality semantics flag and the state namespace that keeps bandit/EMA history from mixing.
"""Which meaning the stored quality has, and how learned state is kept apart.

``Q`` used to be one number: the three-dimension rubric mean. With the formative
score it becomes a different quantity — ``Q_calibrado = Q_tech * (1 - p^3)`` —
and the two are not comparable. That matters far beyond reporting, because the
reward derived from quality feeds the bandit's Beta posteriors and the shared
EMAs. Switching the meaning without separating the state would mean continuing
to learn from a prior accumulated under another distribution: the numbers would
keep moving, nothing would error, and months of exploration would be silently
wrong.

So the state is **namespaced, not reset**. Turning the flag on starts a fresh
namespace and leaves the old one readable, which makes the change reversible and
lets both semantics run side by side as experimental conditions.

The single most important property here is that :func:`namespaced` is the
**identity** under the default semantics. If it were not, merely deploying this
module would orphan every existing key — the exact failure it exists to prevent.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.settings_dynamic import settings

logger = logging.getLogger(__name__)

#: The rubric mean, as stored since the beginning.
SEMANTICS_RUBRIC_V1 = "rubric_v1"

#: Q_tech times the annihilation term; see ``services.judge_usurpation``.
SEMANTICS_FORMATIVE_V2 = "formative_v2"

VALID_SEMANTICS = (SEMANTICS_RUBRIC_V1, SEMANTICS_FORMATIVE_V2)

DEFAULT_SEMANTICS = SEMANTICS_RUBRIC_V1

SETTING_KEY = "JUDGE_QUALITY_SEMANTICS"


def current_semantics() -> str:
    """The active quality semantics, falling back to the legacy one.

    An unrecognised value falls back rather than raising: a typo in a setting
    must not take the routing path down, and silently learning under an
    unnamed semantics would be worse than staying on the known one.
    """
    try:
        value = str(settings.get(SETTING_KEY, DEFAULT_SEMANTICS) or DEFAULT_SEMANTICS).strip()
    except Exception:
        return DEFAULT_SEMANTICS
    if value not in VALID_SEMANTICS:
        logger.warning("[semantics] valor desconhecido %r; a usar %s", value, DEFAULT_SEMANTICS)
        return DEFAULT_SEMANTICS
    return value


def state_namespace() -> Optional[str]:
    """The prefix for learned state, or ``None`` under the legacy semantics.

    ``None`` is what makes the default a no-op: every existing Redis key and
    every existing row keeps the identifier it already has.
    """
    semantics = current_semantics()
    return None if semantics == DEFAULT_SEMANTICS else semantics


def namespaced(value: str) -> str:
    """Qualify one state identifier with the active semantics.

    Under ``rubric_v1`` this returns ``value`` unchanged, so deploying the
    namespacing changes nothing until the flag is turned on.
    """
    prefix = state_namespace()
    return value if prefix is None else f"{prefix}:{value}"


def is_formative() -> bool:
    """True when the calibrated score is the one being learned from."""
    return current_semantics() == SEMANTICS_FORMATIVE_V2
