# Objective: Shadow-evaluation aggregates for the research dashboard: counts, per-model means, regret over time.
"""Resumo da execução em sombra: contagens, médias por modelo e arrependimento/oráculo por dia e disciplina."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

from app.services.sombra.analise import _bool, _num, analisar, metricas

#: Campos de ``metricas`` que o painel usa (o acumulado depende do tamanho da janela e não é comparável).
_CAMPOS = (
    "n",
    "arrependimento_medio",
    "concordancia_oraculo",
    "melhor_config_fixa",
    "ganho_sobre_melhor_fixa",
    "media_por_config",
)


def disciplina(estrato: Any) -> str:
    """The ``disciplina`` key of a stratum (JSON text or dict); ``"?"`` when absent or unreadable."""
    if isinstance(estrato, str):
        try:
            estrato = json.loads(estrato)
        except ValueError:
            return "?"
    valor = estrato.get("disciplina") if isinstance(estrato, dict) else None
    return str(valor) if valor else "?"


def _media(valores: List[float]) -> Optional[float]:
    return sum(valores) / len(valores) if valores else None


def por_modelo(linhas: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per model: scored rows, mean score, total cost, mean latency, delivered vs shadow rows, would-abstain."""
    acum: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"escores": [], "latencias": [], "custo": 0.0, "entregue": 0, "sombra": 0, "teria_abstido": 0}
    )
    for linha in linhas:
        if not _bool(linha.get("executada")) or not linha.get("modelo"):
            continue
        a = acum[str(linha["modelo"])]
        escore, latencia = _num(linha.get("escore_agregado")), _num(linha.get("latencia_s"))
        if linha.get("status") == "ok" and escore is not None:
            a["escores"].append(escore)
        if latencia is not None:
            a["latencias"].append(latencia)
        a["custo"] += _num(linha.get("custo_usd")) or 0.0
        a["entregue" if linha.get("papel") == "entregue" else "sombra"] += 1
        a["teria_abstido"] += int(_bool(linha.get("teria_abstido")))
    return {
        m: {
            "n": len(a["escores"]),
            "escore_medio": _media(a["escores"]),
            "custo_usd": round(a["custo"], 6),
            "latencia_media_s": _media(a["latencias"]),
            "entregue": a["entregue"],
            "sombra": a["sombra"],
            "teria_abstido": a["teria_abstido"],
        }
        for m, a in sorted(acum.items())
    }


def _agrupar(
    avaliadas: List[Dict[str, Any]], pesos: Dict[str, float], chave: Callable[[Dict[str, Any]], str]
) -> Dict[str, Dict[str, Any]]:
    grupos: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in avaliadas:
        grupos[chave(r)].append(r)
    return {g: _enxuto(metricas(membros, pesos)) for g, membros in sorted(grupos.items())}


def _enxuto(m: Dict[str, Any]) -> Dict[str, Any]:
    return {c: m.get(c) for c in _CAMPOS if c in m}


def resumo_sombra(linhas: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Counts, per-model means and IPW-weighted regret/oracle agreement overall, per day, discipline and regime."""
    res = analisar(linhas)
    avaliadas, pesos = res["avaliadas"], res["pesos"]
    return {
        "contagens": res["contagens"],
        "por_modelo": por_modelo(linhas),
        "arrependimento": {
            "geral": _enxuto(metricas(avaliadas, pesos)),
            "por_dia": _agrupar(avaliadas, pesos, lambda r: r["dia"]),
            "por_disciplina": _agrupar(avaliadas, pesos, lambda r: disciplina(r["estrato"])),
            "por_regime": _agrupar(avaliadas, pesos, lambda r: r["regime"] or "?"),
        },
    }
