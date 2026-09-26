# Objective: Pure shadow-evaluation analysis (IPW, regret, oracle agreement), shared by the export script and the API.
"""Métricas da execução em sombra a partir das linhas de ``shadow_evaluations`` (sem acesso a banco).

Probabilidade efetiva de inclusão por estrato x dia: ``pi_hat = p x executadas / sorteadas``; cada requisição
executada pesa ``1 / pi_hat`` (Horvitz-Thompson). Por requisição executada com a entregue e ao menos outra candidata
pontuadas: arrependimento = max escore - escore da entregue; concordância com o oráculo = a entregue atinge o máximo.
Com a subamostra de candidatas (``p_candidata < 1``) o máximo é o das sorteadas: o arrependimento é um limite
inferior e a concordância um limite superior; as médias por configuração seguem não enviesadas.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional


def _num(v: Any) -> Optional[float]:
    try:
        return None if v in (None, "") else float(v)
    except (TypeError, ValueError):
        return None


def _bool(v: Any) -> bool:
    return str(v).strip().lower() in ("1", "true", "t")


def _requisicoes(linhas: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    reqs: Dict[str, Dict[str, Any]] = {}
    for linha in linhas:
        r = reqs.setdefault(
            str(linha["request_id"]),
            {
                "request_id": str(linha["request_id"]),
                "estrato": str(linha.get("estrato") or "{}"),
                "dia": str(linha.get("criado_em") or "")[:10],
                "criado_em": str(linha.get("criado_em") or ""),
                "regime": linha.get("regime_entrega") or "",
                "p": _num(linha.get("p_nominal")) or 0.0,
                "executada": _bool(linha.get("executada")),
                "motivo": linha.get("motivo_corte"),
                "escores": {},
                "entregue": None,
            },
        )
        escore = _num(linha.get("escore_agregado"))
        if linha.get("status") == "ok" and escore is not None:
            r["escores"][linha["modelo"]] = escore
        if linha.get("papel") == "entregue":
            r["entregue"] = linha["modelo"]
    return reqs


def pesos_ipw(reqs: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """1 / pi_hat per executed request, pi_hat = p x executed / sampled within (stratum, day)."""
    grupos: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in reqs.values():
        grupos[(r["estrato"], r["dia"])].append(r)
    pesos = {}
    for membros in grupos.values():
        executadas = [r for r in membros if r["executada"]]
        for r in executadas:
            pi = r["p"] * len(executadas) / len(membros)
            pesos[r["request_id"]] = 1.0 / pi if pi > 0 else 0.0
    return pesos


def _media(pares: List[tuple]) -> Optional[float]:
    total = sum(w for _, w in pares)
    return sum(v * w for v, w in pares) / total if total > 0 else None


def _ordem(valor: Optional[float]) -> float:
    return valor if valor is not None else float("-inf")


def metricas(avaliadas: List[Dict[str, Any]], pesos: Dict[str, float]) -> Dict[str, Any]:
    """Weighted regret, oracle agreement and gain over the best fixed configuration for one group."""
    if not avaliadas:
        return {"n": 0}
    w = {r["request_id"]: pesos.get(r["request_id"], 0.0) for r in avaliadas}
    arrep = _media([(r["arrependimento"], w[r["request_id"]]) for r in avaliadas])
    oraculo = _media([(1.0 if r["oraculo"] else 0.0, w[r["request_id"]]) for r in avaliadas])
    configs = {c for r in avaliadas for c in r["escores"]}
    medias = {
        c: _media([(r["escores"][c], w[r["request_id"]]) for r in avaliadas if c in r["escores"]]) for c in configs
    }
    melhor = max(medias, key=lambda c: _ordem(medias[c]))
    comuns = [r for r in avaliadas if melhor in r["escores"]]
    ganho = None
    if comuns:
        entregue = _media([(r["escores"][r["entregue"]], w[r["request_id"]]) for r in comuns])
        fixa = _media([(r["escores"][melhor], w[r["request_id"]]) for r in comuns])
        ganho = None if entregue is None or fixa is None else entregue - fixa
    return {
        "n": len(avaliadas),
        "arrependimento_medio": arrep,
        "arrependimento_acumulado": sum(r["arrependimento"] * w[r["request_id"]] for r in avaliadas),
        "concordancia_oraculo": oraculo,
        "melhor_config_fixa": melhor,
        "ganho_sobre_melhor_fixa": ganho,
        "media_por_config": medias,
    }


def analisar(linhas: List[Dict[str, Any]]) -> Dict[str, Any]:
    reqs = _requisicoes(linhas)
    pesos = pesos_ipw(reqs)
    avaliadas = []
    for r in sorted(reqs.values(), key=lambda r: r["criado_em"]):
        entregue = r["entregue"]
        if not r["executada"] or entregue not in r["escores"] or len(r["escores"]) < 2:
            continue
        melhor = max(r["escores"].values())
        avaliadas.append(
            {**r, "arrependimento": melhor - r["escores"][entregue], "oraculo": r["escores"][entregue] >= melhor}
        )
    por_grupo: Dict[str, Dict[str, Any]] = {}
    for regime in ("aproveitamento", "exploracao", "todas"):
        doregime = [r for r in avaliadas if regime == "todas" or r["regime"] == regime]
        por_grupo[regime] = {
            "agregado": metricas(doregime, pesos),
            "por_estrato": {
                e: metricas([r for r in doregime if r["estrato"] == e], pesos)
                for e in sorted({r["estrato"] for r in doregime})
            },
        }
    cortes: Dict[str, int] = defaultdict(int)
    for r in reqs.values():
        if not r["executada"]:
            cortes[str(r["motivo"])] += 1
    contagens = {
        "sorteadas": len(reqs),
        "executadas": sum(r["executada"] for r in reqs.values()),
        "avaliadas": len(avaliadas),
        "cortadas_por_motivo": dict(cortes),
    }
    return {"contagens": contagens, "metricas": por_grupo, "avaliadas": avaliadas, "pesos": pesos}
