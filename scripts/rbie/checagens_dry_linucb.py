# -*- coding: utf-8 -*-
"""Checagens em modo seco do contrato: caminho com centroides e aprender=False.

Roda contra o ARISTO em modo seco (OPENROUTER_BASE_URL apontando para o servidor
falso). Nao gasta chamada paga. Verifica:

- um estado novo com politica ``centroide`` (padrao) continua decidindo por
  centroides, votos das tres politicas e fatores de risco;
- um estado ``linucb`` com ``aprender=False`` escolhe pelo bandit congelado e o
  hash do estado nao muda depois da requisicao, com ``recompensa`` nula.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

import httpx

URL = os.environ.get("ROTEADOR_URL", "http://127.0.0.1:8000")


def _post(caminho: str, corpo: dict | None = None) -> dict:
    r = httpx.post(URL + caminho, json=corpo or {}, timeout=300.0)
    r.raise_for_status()
    return r.json()


def _get(caminho: str) -> dict:
    r = httpx.get(URL + caminho, timeout=60.0)
    r.raise_for_status()
    return r.json()


def main() -> int:
    sufixo = uuid.uuid4().hex[:8]
    saida: dict = {}

    # 1. caminho antigo: centroides
    antigo = f"teste-centroide-{sufixo}"
    _post("/estados", {"id_estado": antigo, "politica": "centroide"})
    _post(f"/estados/{antigo}/retomar")
    r_antigo = _post("/rotear", {
        "id_estado": antigo, "id_requisicao": f"centroide|{sufixo}",
        "prompt": "PROMPT DE TESTE", "semente": 20261030,
        "metadados": {"etapa": "teste", "id_redacao": 1},
    })
    saida["centroide"] = {
        "politica": r_antigo.get("politica"),
        "tem_votos": set((r_antigo.get("decisao") or {}).get("votos") or {})
        == {"epsilon_greedy", "ucb1", "thompson"},
        "id_centroide": (r_antigo.get("decisao") or {}).get("id_centroide"),
        "n_centroides": r_antigo.get("n_centroides"),
        "ok": (r_antigo.get("politica") == "centroide"
               and set((r_antigo.get("decisao") or {}).get("votos") or {})
               == {"epsilon_greedy", "ucb1", "thompson"}
               and int(r_antigo.get("n_centroides") or 0) >= 1),
    }

    # 2. LinUCB com aprender falso: hash nao muda e recompensa e nula
    congelado = f"teste-congelado-{sufixo}"
    _post("/estados", {"id_estado": congelado, "politica": "linucb",
                       "alpha": 0.25, "n0": 20, "aprender": False})
    _post(f"/estados/{congelado}/retomar")
    antes = _get(f"/estados/{congelado}/resumo")
    r_congelado = _post("/rotear", {
        "id_estado": congelado, "id_requisicao": f"congelado|{sufixo}",
        "prompt": "PROMPT DE TESTE", "texto_redacao": "Uma redacao curta de teste.",
        "notas_humanas": [120, 120, 120, 120, 120], "semente": 20261030,
        "metadados": {"etapa": "teste", "id_redacao": 2},
    })
    depois = _get(f"/estados/{congelado}/resumo")
    saida["congelado"] = {
        "politica": r_congelado.get("politica"),
        "aprender": r_congelado.get("aprender"),
        "recompensa": r_congelado.get("recompensa"),
        "alpha": r_congelado.get("alpha"),
        "n0": r_congelado.get("n0"),
        "hash_antes": antes["hash_estado"],
        "hash_depois": depois["hash_estado"],
        "n_atualizacoes_linucb": (depois.get("linucb") or {}).get("n_atualizacoes"),
        "ok": (r_congelado.get("politica") == "linucb"
               and r_congelado.get("aprender") is False
               and r_congelado.get("recompensa") is None
               and antes["hash_estado"] == depois["hash_estado"]
               and (depois.get("linucb") or {}).get("n_atualizacoes") == 0),
    }

    print(json.dumps(saida, ensure_ascii=False, indent=2))
    destino = Path(os.environ.get(
        "ARISTO_CHECAGENS_DRY",
        "/data/desenvolvimento/iag_forall/state/rbie/checagens_dry_linucb.json"))
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf8")
    return 0 if all(item["ok"] for item in saida.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
