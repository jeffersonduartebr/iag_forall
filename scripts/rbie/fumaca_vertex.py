# -*- coding: utf-8 -*-
"""Fumaca real da via Vertex do roteador (chamada paga de centavos).

Uso, a partir da raiz do ARISTO, em modo real (sem OPENROUTER_BASE_URL):

    PYTHONPATH=app ARISTO_CONFIG_EFETIVA=state/rbie/config_efetiva.json \
    .venv/bin/python scripts/rbie/fumaca_vertex.py \
        --modelo google/gemini-3.8-flash --saida state/rbie/fumaca_vertex_gemini.json

Grava o corpo enviado, o hash e a chamada. A comparacao com
``corpo_requisicao.corpo_gerador`` do harness e feita do lado do harness, no
mesmo comando que valida o custo de tabela.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ / "app") not in sys.path:
    sys.path.insert(0, str(RAIZ / "app"))

from app.rbie import cliente, configuracao  # noqa: E402


async def executar(modelo: str, prompt: str, semente: int) -> dict:
    corpo = cliente.corpo_gerador(modelo, prompt, semente)
    chamada, texto = await cliente.chamar(modelo, corpo, papel="gerador")
    return {
        "modelo": modelo,
        "prompt": prompt,
        "semente": semente,
        "via": configuracao.via_modelo(modelo),
        "corpo": corpo,
        "hash_corpo": cliente.hash_corpo(corpo),
        "texto": texto,
        "chamada": chamada,
        "custo_tabela_calculado_usd": cliente.custo_tabela_chamada(chamada),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Fumaca real da via Vertex do roteador.")
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--prompt",
                    default=("Responda apenas com a palavra OK, sem pontuacao e sem "
                             "qualquer outro texto."))
    ap.add_argument("--semente", type=int, default=20261030)
    ap.add_argument("--saida", required=True)
    args = ap.parse_args(argv)
    resultado = asyncio.run(executar(args.modelo, args.prompt, args.semente))
    destino = Path(args.saida)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf8")
    chamada = resultado["chamada"]
    print(json.dumps({
        "arquivo": str(destino),
        "modelo": args.modelo,
        "via": resultado["via"],
        "erro": chamada.get("erro"),
        "http_status": chamada.get("http_status"),
        "finish_reason": chamada.get("finish_reason"),
        "modelo_api": chamada.get("modelo_api"),
        "provedor_solicitado": chamada.get("provedor_solicitado"),
        "tokens_entrada": chamada.get("tokens_entrada"),
        "tokens_saida": chamada.get("tokens_saida"),
        "tokens_raciocinio": chamada.get("tokens_raciocinio"),
        "custo_tabela_calculado_usd": resultado["custo_tabela_calculado_usd"],
        "custo_cobrado_usd": chamada.get("custo_cobrado_usd"),
        "origem_custo": chamada.get("origem_custo"),
        "is_byok": chamada.get("is_byok"),
        "texto": resultado["texto"][:80],
        "hash_corpo": resultado["hash_corpo"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
