# Objective: Per-provider circuit breakers that actually count failures.
"""Circuit breakers for the provider layer.

Two defects lived in the previous single-breaker design, and the second hid the
first completely.

**The breakers never counted anything.** ``pybreaker``'s decorator wraps the
function with the *synchronous* ``CircuitBreaker.call``. Applied to an ``async
def``, that call returns a coroutine object rather than raising, so the breaker
recorded a success on every request and the failure counter never left zero.
Neither the cloud breaker nor the local one could ever open, whatever the
providers did. ``guarded_by`` awaits the call properly.

**One breaker was shared by every cloud provider.** With the accounting fixed,
that would mean an OpenAI outage opening the circuit for Anthropic, Gemini and
OpenRouter too — three healthy providers taken down by a fourth. Each provider
now has its own.

The decorator order matters as much as the instance. The retry strategy must
sit *inside* the breaker, so one user request counts as one failure rather than
five; with the retry outside, a single request exhausted ``fail_max`` on its
own.
"""

from __future__ import annotations

import logging
from typing import Dict

import pybreaker

logger = logging.getLogger("providers_async")

try:
    from app.settings_dynamic import settings as dynamic_settings

    CB_FAIL_MAX = dynamic_settings.CIRCUIT_BREAKER_FAIL_MAX
    CB_RESET_TIMEOUT = dynamic_settings.CIRCUIT_BREAKER_RESET_TIMEOUT
    CB_LOCAL_FAIL_MAX = dynamic_settings.CIRCUIT_BREAKER_LOCAL_FAIL_MAX
    CB_LOCAL_RESET_TIMEOUT = dynamic_settings.CIRCUIT_BREAKER_LOCAL_RESET_TIMEOUT
except Exception:
    CB_FAIL_MAX, CB_RESET_TIMEOUT = 5, 60
    CB_LOCAL_FAIL_MAX, CB_LOCAL_RESET_TIMEOUT = 3, 30


#: Um breaker por provider de nuvem. Partilhar uma instância faria uma falha
#: da OpenAI abrir o circuito da Anthropic, do Gemini e do OpenRouter.
CLOUD_BREAKERS: Dict[str, pybreaker.CircuitBreaker] = {
    name: pybreaker.CircuitBreaker(
        fail_max=CB_FAIL_MAX, reset_timeout=CB_RESET_TIMEOUT, name=f"cloud_breaker_{name}"
    )
    for name in ("openai", "openrouter", "anthropic", "gemini")
}

local_breaker = pybreaker.CircuitBreaker(
    fail_max=CB_LOCAL_FAIL_MAX, reset_timeout=CB_LOCAL_RESET_TIMEOUT, name="local_breaker"
)

#: Compatibilidade: era o nome exportado para os quatro providers. Aponta ao
#: breaker da OpenAI para que o que dependia dele continue a resolver, mas os
#: providers passaram a usar o seu.
cloud_breaker = CLOUD_BREAKERS["openai"]


def reset_breakers() -> None:
    """Close every provider circuit and clear its failure counter.

    ``breaker._state = CircuitClosedState(breaker)`` — the idiom this replaces,
    used here and in five test modules — does **not** reset anything:
    ``current_state`` is read from the state *storage*, so it stayed ``open``
    with the counter intact. It never showed because the breakers could not
    open in the first place.
    """
    for breaker in (*CLOUD_BREAKERS.values(), local_breaker):
        breaker.close()


def breaker_states() -> Dict[str, str]:
    """Current state per provider, for /health and diagnostics."""
    states = {name: b.current_state for name, b in CLOUD_BREAKERS.items()}
    states["local"] = local_breaker.current_state
    return states
