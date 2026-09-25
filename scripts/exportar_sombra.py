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
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from app.services.sombra.analise import _requisicoes, analisar, metricas, pesos_ipw  # noqa: E402

__all__ = ["_requisicoes", "analisar", "metricas", "pesos_ipw"]

SEMENTE_SORTEIO = "aristo-sombra"  # sha256(SEMENTE:request_id) < p (services/sombra/captura.py)


def ler_csv(caminho: Path) -> List[Dict[str, Any]]:
    with caminho.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def ler_banco(desde: Optional[str], tenant: Optional[str]) -> List[Dict[str, Any]]:
    from app.db import get_engine
    from sqlalchemy import text

    sql, params = "SELECT * FROM shadow_evaluations WHERE 1=1", {}
    if desde:
        sql, params["desde"] = sql + " AND criado_em >= :desde", desde
    if tenant:
        sql, params["tenant"] = sql + " AND tenant = :tenant", tenant
    with get_engine().connect() as conn:
        return [dict(r._mapping) for r in conn.execute(text(sql + " ORDER BY criado_em, id"), params)]


def _manifesto() -> Optional[Dict[str, Any]]:
    try:
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
