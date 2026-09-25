# Objective: Score one answer with the shadow judge panel, without persisting any text (R5, R7).
"""Same instruments as the delivered answer's judging (rubric + delivery level), none of its side effects.

``judges.judge_answer`` stores the answer text in ``judge_logs``, draws a random pair per call, re-retrieves an
unscoped RAG context and caches by text. Here the panel is fixed per request (``SHADOW_JUDGE_MODELS``), each
candidate loses only the judges of its own company (the rule of ``judge_vendors``), the context is the request's own
retrieved context, and only numbers leave this module.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

from app.services.judge_rubric import (
    TECH_DIMENSIONS,
    build_rubric_prompt,
    combine_ratings,
    parse_rubric_weights,
    rate_with_rubric,
    weighted_quality,
)
from app.services.judge_usurpation import (
    aggregate_delivery,
    build_usurpation_prompt,
    calibrated_quality,
    rate_usurpation,
)
from app.services.judge_vendors import empresa

from .admissibilidade import motivo_recusa
from .config import ConfigSombra

TEMPERATURA_JUIZ, TETO_JUIZ = 0.0, 512
ChamarModelo = Callable[..., Awaitable[Tuple[str, Any]]]


def painel_para(avaliado: str, cfg: ConfigSombra, modalidade: str = "text") -> List[str]:
    """Admissible judges of the base panel that are not from the evaluated model's company."""
    empresa_avaliado = empresa(avaliado)
    return [
        j for j in cfg.juizes
        if motivo_recusa(j, cfg, "text") is None and (empresa_avaliado is None or empresa(j) != empresa_avaliado)
    ]


def _custo(meta: Dict[str, Any]) -> float:
    return float(meta.get("call_cost_usd") or meta.get("cost_per_1k") or 0.0) if meta else 0.0


async def _um_juiz(chamar: ChamarModelo, juiz: str, rubrica: str, entrega: str) -> Tuple[str, Optional[dict], Optional[float], float]:
    (notas, meta_r), (nivel, meta_e) = await asyncio.gather(
        rate_with_rubric(chamar, juiz, rubrica, TEMPERATURA_JUIZ, TETO_JUIZ),
        rate_usurpation(chamar, juiz, entrega, TEMPERATURA_JUIZ, TETO_JUIZ),
    )
    return juiz, notas, nivel, _custo(meta_r) + _custo(meta_e)


async def julgar(
    chamar: ChamarModelo, painel: Sequence[str], *, pergunta: str, resposta: str, contexto: str, pesos_brutos: Any,
    formativa: bool,
) -> Dict[str, Any]:
    """Per-judge scores, the aggregate and the judges' cost; the answer text is only read, never returned."""
    pesos = parse_rubric_weights(pesos_brutos)
    rubrica = build_rubric_prompt(pergunta, resposta, rag_context=contexto)
    entrega = build_usurpation_prompt(pergunta, resposta)
    resultados = await asyncio.gather(*[_um_juiz(chamar, j, rubrica, entrega) for j in painel])
    por_juiz: Dict[str, Dict[str, Any]] = {}
    for juiz, notas, nivel, _ in resultados:
        por_juiz[juiz] = {
            "q": round(weighted_quality(notas, pesos), 4) if notas else None,
            "dimensoes": notas,
            "p_entrega": nivel,
        }
    agregado = combine_ratings([n for _, n, _, _ in resultados if n], pesos)
    escore = None
    if agregado:
        q_tech = weighted_quality(agregado["dimensions"], pesos, TECH_DIMENSIONS)
        p = aggregate_delivery([nv for _, _, nv, _ in resultados])
        escore = round(calibrated_quality(q_tech, p) if formativa else agregado["quality"], 4)
    return {"escores_juizes": por_juiz, "escore_agregado": escore, "custo_juizes": sum(c for *_, c in resultados)}
