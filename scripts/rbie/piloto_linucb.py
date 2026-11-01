# -*- coding: utf-8 -*-
"""Piloto de integracao do LinUCB no ARISTO (chamadas pagas, alguns centavos).

Cria um estado novo com a politica ``linucb``, dispara 30 requisicoes reais com
redacoes diferentes (textos sinteticos distintos; nenhuma nota humana real e
lida) e confere a distribuicao de bracos, a persistencia de A e b e a validade
dos instantaneos do estado.

Uso, a partir da raiz do ARISTO, com o roteador no ar em modo real:

    .venv/bin/python scripts/rbie/piloto_linucb.py \
        --saida state/rbie/piloto_linucb.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from collections import Counter
from pathlib import Path

import httpx

URL = "http://127.0.0.1:8000"

PROMPT = (
    "Responda apenas com um objeto JSON, sem nenhum texto fora dele, no formato "
    '{"competencias": [{"competencia": 1, "nivel": 120, "evidencia": "trecho", '
    '"proximo_passo": "acao"}, ... ate a competencia 5], "comentario_geral": "texto"}. '
    "Use nivel em {0, 40, 80, 120, 160, 200}; evidencia e proximo_passo curtos."
)

TEMAS = [
    "desafios da mobilidade urbana nas grandes cidades brasileiras",
    "impactos das redes sociais na saude mental dos jovens",
    "importancia da educacao financeira no ensino medio",
    "caminhos para reduzir o desperdicio de alimentos",
    "relacao entre tecnologia e mercado de trabalho",
    "preservacao de recursos hidricos no semiarido",
]


def _textos() -> list[str]:
    """Trinta textos distintos, sem relacao com as redacoes do experimento."""
    textos = []
    for i in range(30):
        tema = TEMAS[i % len(TEMAS)]
        textos.append(
            f"A redacao numero {i + 1} discute {tema}. "
            f"{'Primeiro' if i % 2 == 0 else 'Inicialmente'}, o texto apresenta uma "
            f"visao geral do problema e cita o exemplo {chr(65 + i % 26)}. "
            f"Em seguida, defende que a escola e a familia devem atuar juntas, "
            f"com {i % 7 + 2} argumentos complementares. "
            f"Por fim, propoe uma acao concreta do poder publico para o tema."
        )
    return textos


def _notas(indice: int) -> list[int]:
    """Notas humanas sinteticas, so para exercitar o sinal de concordancia."""
    base = (80, 120, 160)[indice % 3]
    return [base] * 5


async def executar(saida: Path) -> dict:
    sufixo = uuid.uuid4().hex[:8]
    id_estado = f"teste-linucb-{sufixo}"
    async with httpx.AsyncClient(timeout=300.0) as cli:
        await cli.post(URL + "/estados", json={
            "id_estado": id_estado, "politica": "linucb",
            "alpha": 0.25, "n0": 20.0, "aprender": True,
        })
        await cli.post(URL + f"/estados/{id_estado}/retomar")
        textos = _textos()
        pedidos = []
        for i, texto in enumerate(textos):
            r = await cli.post(URL + "/rotear", json={
                "id_estado": id_estado, "id_requisicao": f"piloto-linucb|{sufixo}|{i}",
                "prompt": PROMPT, "semente": 20261101 + i,
                "texto_redacao": texto, "notas_humanas": _notas(i),
                "usar_recuperacao": False, "usar_cache_semantico": False,
                "metadados": {"etapa": "piloto_linucb", "id_redacao": 900000 + i},
            })
            r.raise_for_status()
            d = r.json()
            custo = sum(float(c.get("custo_cobrado_usd") or 0.0) for c in d.get("chamadas") or [])
            pedidos.append({
                "i": i, "modelo": d.get("modelo"), "erro": d.get("erro"),
                "politica": d.get("politica"), "braco": (d.get("decisao") or {}).get("braco"),
                "recompensa": d.get("recompensa"),
                "recompensa_disponivel": d.get("recompensa_disponivel"),
                "qualidade_concordancia": d.get("qualidade_concordancia"),
                "status_parser": d.get("status_parser"),
                "custo_cobrado_usd": round(custo, 8),
                "n_atualizacoes": None,
            })
        resumo_antes = (await cli.get(URL + f"/estados/{id_estado}/resumo")).json()
        await cli.post(URL + f"/estados/{id_estado}/instantaneo",
                       json={"id_instantaneo": f"piloto-linucb-snap-{sufixo}"})
        await cli.post(URL + f"/estados/{id_estado}/restaurar",
                       json={"id_instantaneo": f"piloto-linucb-snap-{sufixo}"})
        resumo_depois = (await cli.get(URL + f"/estados/{id_estado}/resumo")).json()

    distribuicao = Counter(p["braco"] for p in pedidos if p["braco"])
    custo_total = round(sum(p["custo_cobrado_usd"] for p in pedidos), 8)
    resultado = {
        "id_estado": id_estado,
        "n_requisicoes": len(pedidos),
        "distribuicao_bracos": dict(distribuicao),
        "braco_por_requisicao": [p["braco"] for p in pedidos],
        "n_erros": sum(1 for p in pedidos if p["erro"]),
        "recompensas": [p["recompensa"] for p in pedidos],
        "custo_total_usd": custo_total,
        "resumo_antes": resumo_antes,
        "resumo_depois": resumo_depois,
        "hash_instantaneo_igual": resumo_antes.get("hash_estado") == resumo_depois.get("hash_estado"),
        "pedidos": pedidos,
    }
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf8")
    return resultado


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Piloto de integracao do LinUCB (pago).")
    ap.add_argument("--saida", default="state/rbie/piloto_linucb.json")
    args = ap.parse_args(argv)
    resultado = asyncio.run(executar(Path(args.saida)))
    resumo = resultado["resumo_depois"]
    linucb = resumo.get("linucb") or {}
    print(json.dumps({
        "id_estado": resultado["id_estado"],
        "n_requisicoes": resultado["n_requisicoes"],
        "distribuicao_bracos": resultado["distribuicao_bracos"],
        "braco_por_requisicao": resultado["braco_por_requisicao"],
        "n_erros": resultado["n_erros"],
        "n_atualizacoes_linucb": linucb.get("n_atualizacoes"),
        "n_por_braco": linucb.get("n_por_braco"),
        "desvio_a": linucb.get("desvio_a"),
        "norma_b": linucb.get("norma_b"),
        "alpha": linucb.get("alpha"),
        "n0": linucb.get("n0"),
        "hash_instantaneo_igual": resultado["hash_instantaneo_igual"],
        "custo_total_usd": resultado["custo_total_usd"],
        "arquivo": str(args.saida),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
