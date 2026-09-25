# Objective: query_log aggregates for one tenant: exploration regime counts and reasons, cost and latency totals.
"""Resumo do ``query_log`` de um tenant: regime de exploração (``decision_json.regime``) e totais operacionais."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

from app.services.sombra.analise import _bool, _num


def _decisao(bruto: Any) -> Dict[str, Any]:
    if isinstance(bruto, dict):
        return bruto
    try:
        valor = json.loads(bruto) if bruto else {}
    except (TypeError, ValueError):
        return {}
    return valor if isinstance(valor, dict) else {}


def percentil(valores: List[float], q: float) -> Optional[float]:
    """Linear-interpolated percentile (``q`` in 0..1); ``None`` for an empty list."""
    if not valores:
        return None
    ordenados = sorted(valores)
    pos = (len(ordenados) - 1) * q
    baixo = int(pos)
    alto = min(baixo + 1, len(ordenados) - 1)
    return ordenados[baixo] + (ordenados[alto] - ordenados[baixo]) * (pos - baixo)


def resumo_regime(linhas: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Exploit/explore counts, why epsilon was zero, regenerations, mean assignment probability, per-day series."""
    contagem: Counter = Counter()
    motivos: Counter = Counter()
    por_dia: Dict[str, Counter] = defaultdict(Counter)
    probs: List[float] = []
    regenerados = 0
    for linha in linhas:
        regime = _decisao(linha.get("decision_json")).get("regime")
        if not isinstance(regime, dict):
            continue
        tipo = "exploracao" if regime.get("explorou") else "aproveitamento"
        contagem[tipo] += 1
        por_dia[str(linha.get("created_at") or "")[:10]][tipo] += 1
        if regime.get("motivo_sem_exploracao"):
            motivos[str(regime["motivo_sem_exploracao"])] += 1
        regenerados += int(bool(regime.get("regenerado")))
        p = _num(regime.get("p_atribuicao"))
        if p is not None:
            probs.append(p)
    return {
        "n": sum(contagem.values()),
        "aproveitamento": contagem["aproveitamento"],
        "exploracao": contagem["exploracao"],
        "motivos_sem_exploracao": dict(motivos),
        "regenerados": regenerados,
        "p_atribuicao_media": sum(probs) / len(probs) if probs else None,
        "por_dia": {d: dict(c) for d, c in sorted(por_dia.items())},
    }


def resumo_totais(linhas: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Cost per day by model, latency p50/p90, abstention and judge-sampling rates."""
    custo: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    latencias: List[float] = []
    abst = juiz = 0
    for linha in linhas:
        dia, modelo = str(linha.get("created_at") or "")[:10], str(linha.get("chosen_model") or "?")
        custo[dia][modelo] += _num(linha.get("estimated_cost_usd")) or 0.0
        latencia = _num(linha.get("latency_s"))
        if latencia is not None:
            latencias.append(latencia)
        abst += int(_bool(linha.get("abstained")))
        juiz += int(_bool(linha.get("judge_sampled")))
    n = len(linhas)
    return {
        "n": n,
        "custo_total_usd": round(sum(v for d in custo.values() for v in d.values()), 6),
        "custo_por_dia_modelo": {d: {m: round(v, 6) for m, v in sorted(ms.items())} for d, ms in sorted(custo.items())},
        "latencia_p50_s": percentil(latencias, 0.5),
        "latencia_p90_s": percentil(latencias, 0.9),
        "taxa_abstencao": abst / n if n else None,
        "taxa_juiz_amostrado": juiz / n if n else None,
    }
