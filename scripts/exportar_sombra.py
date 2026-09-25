#!/usr/bin/env python3
# Objective: Export the shadow evaluation matrix and the regret/oracle metrics, weighted by effective inclusion.
"""Exporta a matriz de desempenho da execução em sombra e calcula as métricas da tese.

Uso:
    python scripts/exportar_sombra.py --saida out/ [--entrada linhas.csv] [--desde 2026-11-16] [--tenant ifrn-caso1]

Sem ``--entrada``, lê ``shadow_evaluations`` do banco do ARISTO (variáveis DB_* do .env).

Probabilidade efetiva de inclusão (por estrato x dia local): cortes por orçamento, teto por tenant, região ou GPU
fazem a inclusão depender do horário. Como cada requisição elegível é sorteada com probabilidade nominal ``p``, o
número de sorteadas estima ``elegíveis x p``; logo ``pi_hat = p x executadas / sorteadas``. As requisições
executadas recebem peso ``1 / pi_hat`` (Horvitz-Thompson). Nenhum dado do query_log é necessário.

Por requisição executada com a resposta entregue e ao menos uma candidata pontuadas:
- arrependimento = max_c escore(c) - escore(entregue);
- concordância com o oráculo = a entregue atinge o máximo;
- ganho sobre a melhor configuração fixa a posteriori = média ponderada(entregue) - média ponderada(melhor fixa),
  sobre as requisições em que a melhor fixa foi avaliada.
Tudo por estrato e no agregado, separando entregas em aproveitamento e em exploração.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

SEMENTE_SORTEIO = "aristo-sombra"  # sha256(SEMENTE:request_id) < p (services/sombra/captura.py)


def _num(v: Any) -> Optional[float]:
    try:
        return None if v in (None, "") else float(v)
    except (TypeError, ValueError):
        return None


def _bool(v: Any) -> bool:
    return str(v).strip().lower() in ("1", "true", "t")


def ler_csv(caminho: Path) -> List[Dict[str, Any]]:
    with caminho.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def ler_banco(desde: Optional[str], tenant: Optional[str]) -> List[Dict[str, Any]]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
    from app.db import get_engine
    from sqlalchemy import text

    sql, params = "SELECT * FROM shadow_evaluations WHERE 1=1", {}
    if desde:
        sql, params["desde"] = sql + " AND criado_em >= :desde", desde
    if tenant:
        sql, params["tenant"] = sql + " AND tenant = :tenant", tenant
    with get_engine().connect() as conn:
        return [dict(r._mapping) for r in conn.execute(text(sql + " ORDER BY criado_em, id"), params)]


def _requisicoes(linhas: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    reqs: Dict[str, Dict[str, Any]] = {}
    for linha in linhas:
        r = reqs.setdefault(str(linha["request_id"]), {
            "request_id": str(linha["request_id"]), "estrato": str(linha.get("estrato") or "{}"),
            "dia": str(linha.get("criado_em") or "")[:10], "criado_em": str(linha.get("criado_em") or ""),
            "regime": linha.get("regime_entrega") or "", "p": _num(linha.get("p_nominal")) or 0.0,
            "executada": _bool(linha.get("executada")), "motivo": linha.get("motivo_corte"),
            "escores": {}, "entregue": None,
        })
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


def metricas(avaliadas: List[Dict[str, Any]], pesos: Dict[str, float]) -> Dict[str, Any]:
    """Weighted regret, oracle agreement and gain over the best fixed configuration for one group."""
    if not avaliadas:
        return {"n": 0}
    w = {r["request_id"]: pesos.get(r["request_id"], 0.0) for r in avaliadas}
    arrep = _media([(r["arrependimento"], w[r["request_id"]]) for r in avaliadas])
    oraculo = _media([(1.0 if r["oraculo"] else 0.0, w[r["request_id"]]) for r in avaliadas])
    configs = {c for r in avaliadas for c in r["escores"]}
    medias = {c: _media([(r["escores"][c], w[r["request_id"]]) for r in avaliadas if c in r["escores"]]) for c in configs}
    melhor = max(medias, key=lambda c: medias[c] if medias[c] is not None else float("-inf"))
    comuns = [r for r in avaliadas if melhor in r["escores"]]
    ganho = None
    if comuns:
        entregue = _media([(r["escores"][r["entregue"]], w[r["request_id"]]) for r in comuns])
        fixa = _media([(r["escores"][melhor], w[r["request_id"]]) for r in comuns])
        ganho = None if entregue is None or fixa is None else entregue - fixa
    return {"n": len(avaliadas), "arrependimento_medio": arrep, "arrependimento_acumulado":
            sum(r["arrependimento"] * w[r["request_id"]] for r in avaliadas), "concordancia_oraculo": oraculo,
            "melhor_config_fixa": melhor, "ganho_sobre_melhor_fixa": ganho, "media_por_config": medias}


def analisar(linhas: List[Dict[str, Any]]) -> Dict[str, Any]:
    reqs = _requisicoes(linhas)
    pesos = pesos_ipw(reqs)
    avaliadas = []
    for r in sorted(reqs.values(), key=lambda r: r["criado_em"]):
        entregue = r["entregue"]
        if not r["executada"] or entregue not in r["escores"] or len(r["escores"]) < 2:
            continue
        melhor = max(r["escores"].values())
        avaliadas.append({**r, "arrependimento": melhor - r["escores"][entregue],
                          "oraculo": r["escores"][entregue] >= melhor})
    por_grupo: Dict[str, Dict[str, Any]] = {}
    for regime in ("aproveitamento", "exploracao", "todas"):
        doregime = [r for r in avaliadas if regime == "todas" or r["regime"] == regime]
        por_grupo[regime] = {"agregado": metricas(doregime, pesos), "por_estrato": {
            e: metricas([r for r in doregime if r["estrato"] == e], pesos) for e in sorted({r["estrato"] for r in doregime})}}
    cortes: Dict[str, int] = defaultdict(int)
    for r in reqs.values():
        if not r["executada"]:
            cortes[str(r["motivo"])] += 1
    contagens = {"sorteadas": len(reqs), "executadas": sum(r["executada"] for r in reqs.values()),
                 "avaliadas": len(avaliadas), "cortadas_por_motivo": dict(cortes)}
    return {"contagens": contagens, "metricas": por_grupo, "avaliadas": avaliadas, "pesos": pesos}


def _manifesto() -> Optional[Dict[str, Any]]:
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
        from app.services.experiment_manifest import _config_snapshot

        return _config_snapshot()
    except Exception:
        return None


def escrever(resultado: Dict[str, Any], saida: Path) -> None:
    saida.mkdir(parents=True, exist_ok=True)
    configs = sorted({c for r in resultado["avaliadas"] for c in r["escores"]})
    with (saida / "sombra_matriz.csv").open("w", newline="", encoding="utf-8") as fh:
        escritor = csv.writer(fh)
        escritor.writerow(["request_id", "criado_em", "estrato", "regime", "entregue", "peso_ipw", "arrependimento",
                           "concordancia_oraculo", *configs])
        for r in resultado["avaliadas"]:
            escritor.writerow([r["request_id"], r["criado_em"], r["estrato"], r["regime"], r["entregue"],
                               resultado["pesos"].get(r["request_id"]), r["arrependimento"], int(r["oraculo"]),
                               *[r["escores"].get(c, "") for c in configs]])
    resumo = {"semente_sorteio": SEMENTE_SORTEIO, "manifesto": _manifesto(), "contagens": resultado["contagens"],
              "metricas": resultado["metricas"]}
    (saida / "sombra_resumo.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2, default=str))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--saida", type=Path, required=True)
    ap.add_argument("--entrada", type=Path, help="CSV com as colunas de shadow_evaluations (padrão: banco)")
    ap.add_argument("--desde")
    ap.add_argument("--tenant")
    args = ap.parse_args(argv)
    linhas = ler_csv(args.entrada) if args.entrada else ler_banco(args.desde, args.tenant)
    resultado = analisar(linhas)
    escrever(resultado, args.saida)
    print(json.dumps(resultado["contagens"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
