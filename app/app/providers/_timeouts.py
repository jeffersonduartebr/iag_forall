# Objective: Resolve the per-request timeout the cloud SDKs were ignoring.
"""The router computed a provider timeout per workload class and passed it
to ``call_model``. Only the Ollama adapter ever read it; the OpenAI,
Anthropic and Gemini paths dropped it and fell back to their SDK defaults of
ten minutes.

The global deadline still cancelled the ``await``, so the client was never
left hanging — but the connection and the token generation continued on the
provider's side, and were billed.
"""

from __future__ import annotations

import os

#: Tecto para uma chamada de nuvem quando o router não impõe um deadline.
#: Os SDKs da OpenAI e da Anthropic assumem 600 s, o que na prática é "sem
#: timeout": o deadline global cancela o `await`, mas a ligação e os tokens
#: continuam do lado do provider — e são facturados.
CLOUD_DEFAULT_TIMEOUT_S = float(os.getenv("CLOUD_DEFAULT_TIMEOUT_S", "120"))


def resolve_timeout(kwargs: dict) -> float:
    """The provider timeout the router asked for, or the ceiling above.

    O router já calculava `timeout_seconds` por classe de workload e passava-o
    a `call_model`. Só o adaptador do Ollama o lia; os três providers de nuvem
    ignoravam-no por completo.
    """
    explicit = kwargs.get("timeout_seconds")
    if explicit is None:
        return CLOUD_DEFAULT_TIMEOUT_S
    return max(1.0, float(explicit))
