# -*- coding: utf-8 -*-
"""Gera a configuracao efetiva do contrato RBIE a partir do config.yaml do experimento.

Uso, a partir da raiz do ARISTO:
    .venv/bin/python scripts/rbie/gerar_config_efetiva.py \
        --config /caminho/para/experimentos/codigo/config.yaml \
        --saida state/rbie/config_efetiva.json

O arquivo gerado e lido por ``app.rbie.configuracao``; ``GET /configuracao``
devolve a secao ``configuracao`` (a mesma forma que o adaptador do harness monta)
e o motor de decisao usa ``modelos``, ``juizes`` e ``openrouter``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parents[2]


def _modelo(cfg: dict, id_modelo: str) -> dict:
    modelos = cfg.get("modelos") or {}
    if id_modelo in modelos:
        return dict(modelos[id_modelo] or {})
    curto = id_modelo.split("/", 1)[-1]
    for chave, valor in modelos.items():
        if chave.split("/", 1)[-1] == curto:
            return dict(valor or {})
    return {}


def gerar(cfg: dict) -> dict:
    roteador = cfg["roteador"]
    experimento = cfg["experimento"]
    classes = roteador.get("classes_risco") or {}
    candidatos = []
    geracao = {}
    for id_modelo, dados in (cfg.get("modelos") or {}).items():
        dados = dict(dados or {})
        if dados.get("papel") != "candidato":
            continue
        candidatos.append({
            "id": id_modelo,
            "provedor_slug": dados.get("provedor_slug"),
            "provedor_slug_lote": dados.get("provedor_slug_lote"),
            "preco_tabela": list(dados.get("preco") or [0.0, 0.0]),
            "classe_risco": classes.get(id_modelo),
            "empresa": dados.get("empresa"),
            "via": dados.get("via", "openrouter"),
            "aceita_temperatura": bool(dados.get("aceita_temperatura", True)),
            "aceita_seed": bool(dados.get("aceita_seed", True)),
            "raciocinio": dados.get("raciocinio"),
        })
        limite = int(experimento["max_tokens_saida"])
        if dados.get("raciocinio"):
            limite += int(experimento["orcamento_raciocinio_tokens"])
        geracao[id_modelo] = {
            "temperatura": (float(experimento["temperatura_geradores"])
                            if dados.get("aceita_temperatura", True) else None),
            "max_tokens": limite,
            "reasoning": dados.get("raciocinio"),
            "seed": bool(dados.get("aceita_seed", True)) or None,
        }

    painel = list((cfg.get("juizes") or {}).get("painel") or [])
    detalhes_juizes = {}
    for id_modelo in painel:
        dados = _modelo(cfg, id_modelo)
        detalhes_juizes[id_modelo] = {
            "empresa": dados.get("empresa"),
            "provedor_slug": dados.get("provedor_slug"),
            "provedor_slug_lote": dados.get("provedor_slug_lote"),
            "preco_tabela": list(dados.get("preco") or [0.0, 0.0]),
            "aceita_temperatura": bool(dados.get("aceita_temperatura", True)),
            "aceita_seed": bool(dados.get("aceita_seed", True)),
            "via": dados.get("via", "openrouter"),
        }

    configuracao = {
        "commit": roteador.get("commit"),
        "candidatos": candidatos,
        "juizes_internos": painel,
        "parametros": {
            "epsilon_base": roteador["epsilon_base"],
            "ucb1_c": roteador["ucb1_c"],
            "beta_priors": list(roteador["beta_priors"]),
            "max_centroides": roteador["max_centroides"],
            "limiar_novo_centroide": roteador["limiar_novo_centroide"],
            "limiar_novidade": roteador["limiar_novidade"],
            "fatores_risco": dict(roteador["fatores_risco"]),
            "latencia_k": roteador["latencia_k"],
            "latencia_x0": roteador["latencia_x0"],
            "pesos_padrao": dict(roteador["pesos_padrao"]),
            "ciclo_evolutivo_s": roteador["ciclo_evolutivo_s"],
        },
        "fatores_antes_da_votacao": roteador.get("fatores_antes_da_votacao"),
        "c_base_usd_por_mil": roteador["c_base_usd_por_mil"],
        "geracao": geracao,
        "allow_fallbacks": bool((cfg.get("openrouter") or {}).get("allow_fallbacks", False)),
        "cache_semantico": bool(roteador.get("cache_semantico", False)),
        "recuperacao_interna": bool(roteador.get("recuperacao_interna", False)),
        "classes_risco": dict(classes) or None,
        "openrouter_base_url": (cfg.get("openrouter") or {}).get("url_chat"),
    }
    return {
        "origem": str(cfg.get("_origem", "")),
        "gerado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "configuracao": configuracao,
        "modelos": {k: dict(v or {}) for k, v in (cfg.get("modelos") or {}).items()},
        "juizes": {
            "painel": painel,
            "temperatura": (cfg.get("juizes") or {}).get("temperatura", 0.0),
            "max_tokens": (cfg.get("juizes") or {}).get("max_tokens", 300),
            "pesos_q": (cfg.get("juizes") or {}).get("pesos_q"),
            "detalhes": detalhes_juizes,
        },
        "experimento": {
            "max_tokens_saida": experimento.get("max_tokens_saida"),
            "orcamento_raciocinio_tokens": experimento.get("orcamento_raciocinio_tokens"),
            "temperatura_geradores": experimento.get("temperatura_geradores"),
        },
        "openrouter": dict(cfg.get("openrouter") or {}),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Gera a configuracao efetiva do contrato RBIE.")
    ap.add_argument("--config", default=str(RAIZ / "state" / "rbie" / "config_experimento.yaml"))
    ap.add_argument("--saida", default=str(RAIZ / "state" / "rbie" / "config_efetiva.json"))
    args = ap.parse_args(argv)
    caminho = Path(args.config)
    cfg = yaml.safe_load(caminho.read_text(encoding="utf8")) or {}
    cfg["_origem"] = str(caminho)
    dados = gerar(cfg)
    saida = Path(args.saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps({"saida": str(saida), "candidatos": len(dados["configuracao"]["candidatos"]),
                      "c_base": dados["configuracao"]["c_base_usd_por_mil"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
