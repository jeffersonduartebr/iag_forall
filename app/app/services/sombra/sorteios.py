# Objective: The shadow's two per-request draws — which candidates run and in what order the judges are taken.
"""Both draws are deterministic by request id (``sha256``), so every sample is reproducible from the logged ids.

**Candidates.** Each sampled request runs a uniform subset of the other candidates, ``ceil(fraction x N)`` of the
``N`` available, without replacement. Every candidate has the same known inclusion probability ``k / N``, recorded
on each row (``p_candidata``). Per-configuration means and the gain over the best fixed configuration stay
unbiased; the per-request oracle becomes the best *of the drawn subset*, so the regret it gives is a lower bound.

**Judges.** The base panel is shuffled per request; each answer is judged by the first ``k`` judges of that order
that are not from its own company. With four judges from four companies and ``k = 3``, every answer gets exactly
three judges, and one of the four always sits out.
"""

from __future__ import annotations

import hashlib
import math
from typing import List, Sequence, Tuple


def _chave(etiqueta: str, request_id: str, item: str) -> str:
    return hashlib.sha256(f"aristo-sombra-{etiqueta}:{request_id}:{item}".encode()).hexdigest()


def ordem(etiqueta: str, request_id: str, itens: Sequence[str]) -> List[str]:
    """A deterministic permutation of ``itens`` keyed by the request id."""
    return sorted(itens, key=lambda i: _chave(etiqueta, request_id, i))


def candidatas(request_id: str, todas: Sequence[str], fracao: float) -> Tuple[List[str], List[str], float]:
    """``(drawn, left out, inclusion probability)``; fraction >= 1 (or no candidates) runs them all."""
    todas = list(todas)
    if not todas or fracao >= 1.0:
        return todas, [], 1.0
    k = max(1, math.ceil(fracao * len(todas)))
    embaralhadas = ordem("candidata", request_id, todas)
    sorteadas = set(embaralhadas[:k])
    return [c for c in todas if c in sorteadas], [c for c in todas if c not in sorteadas], k / len(todas)
