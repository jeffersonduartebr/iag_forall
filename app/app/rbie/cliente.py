# -*- coding: utf-8 -*-
# Objective: Provider client for the RBIE contract surface (OpenRouter bodies).
"""Cliente HTTP do roteador para os candidatos e juizes internos.

O corpo de geracao replica ``codigo/src/corpo_requisicao.py`` do harness: uma
unica mensagem de usuario com a instrucao recebida sem alteracao, sem mensagem
de sistema, ``provider.order`` com o provedor fixado, ``allow_fallbacks: false``,
temperatura 0,4, ``max_tokens`` de 640 (1.152 com raciocinio), ``reasoning``
conforme a configuracao, ``seed`` igual a semente recebida quando o modelo aceita,
``usage: {"include": true}`` e nenhum campo ``models``.

Retentativas: ate 4 tentativas, com espera de 5, 10 e 20 s, apenas em erro de
comunicacao (tempo esgotado, conexao, HTTP 408, 429 e 5xx). HTTP 400 a 404 nao
sao repetidos.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from typing import Any, Dict, Optional, Tuple

import httpx

from . import configuracao

TENTATIVAS = 4
ESPERAS_S = (5.0, 10.0, 20.0)


def hash_corpo(corpo: Dict[str, Any]) -> str:
    """SHA-256 do JSON canonico do corpo (chaves ordenadas, sem espacos)."""
    canonico = json.dumps(corpo, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonico.encode("utf8")).hexdigest()


def _limite(id_modelo: str) -> int:
    parametros = configuracao.geracao(id_modelo)
    if parametros.get("max_tokens"):
        return int(parametros["max_tokens"])
    m = configuracao.modelo(id_modelo)
    visivel = int((configuracao.configuracao_bruta().get("experimento") or {}).get("max_tokens_saida", 640))
    if m.get("raciocinio"):
        visivel += int((configuracao.configuracao_bruta().get("experimento") or {})
                       .get("orcamento_raciocinio_tokens", 512))
    return visivel


def corpo_gerador(id_modelo: str, prompt: str, semente: Optional[int]) -> Dict[str, Any]:
    """Corpo de uma chamada a candidato, identico ao de ``corpo_requisicao.corpo_gerador``."""
    m = configuracao.modelo(id_modelo)
    parametros = configuracao.geracao(id_modelo)
    slug = m.get("provedor_slug")
    corpo: Dict[str, Any] = {
        "model": id_modelo,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": _limite(id_modelo),
        "provider": {"order": [slug], "allow_fallbacks": False},
        "usage": {"include": True},
    }
    temperatura = parametros.get("temperatura", 0.4)
    if temperatura is not None and m.get("aceita_temperatura", True):
        corpo["temperature"] = float(temperatura)
    raciocinio = parametros.get("reasoning")
    if raciocinio:
        corpo["reasoning"] = raciocinio
    if m.get("aceita_seed", True) and semente is not None and parametros.get("seed", True):
        corpo["seed"] = int(semente)
    return corpo


def corpo_juiz(id_modelo: str, prompt: str, max_tokens: int | None = None) -> Dict[str, Any]:
    """Corpo de uma chamada a juiz interno: temperatura 0 e limite dos juizes (ou o informado)."""
    m = configuracao.modelo(id_modelo)
    juiz = configuracao.parametros_juiz()
    if max_tokens is not None:
        juiz["max_tokens"] = int(max_tokens)
    corpo: Dict[str, Any] = {
        "model": id_modelo,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(juiz["max_tokens"]),
        "provider": {"order": [m.get("provedor_slug")], "allow_fallbacks": False},
        "usage": {"include": True},
    }
    if m.get("aceita_temperatura", True) and juiz["temperatura"] is not None:
        corpo["temperature"] = float(juiz["temperatura"])
    return corpo


def _url_completions() -> str:
    base = configuracao.openrouter_base_url().rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def _timeout_s() -> float:
    try:
        return float(os.environ.get("RBVIE_TIMEOUT_S", "300"))
    except ValueError:
        return 300.0


def _deve_repetir(status: Optional[int], erro: Optional[str]) -> bool:
    if erro is not None:
        return True
    if status is None:
        return True
    return status in (408, 429) or status >= 500


async def chamar(
    id_modelo: str,
    corpo: Dict[str, Any],
    *,
    papel: str,
) -> Tuple[Dict[str, Any], str]:
    """Envia o corpo ao provedor e devolve ``(chamada, texto)``.

    A ``chamada`` segue o esquema do CONTRATO (papel, modelo, provedor, tokens,
    custo cobrado, latencia, hash do corpo, tentativas). O texto e vazio quando
    todas as tentativas falham.
    """
    cabecalhos = {"Content-Type": "application/json"}
    chave = os.environ.get("OPENROUTER_API_KEY")
    if chave:
        cabecalhos["Authorization"] = f"Bearer {chave}"
    url = _url_completions()
    inicio = time.perf_counter()
    status: Optional[int] = None
    erro: Optional[str] = None
    dados: Dict[str, Any] = {}
    tentativas = 0
    for tentativa in range(1, TENTATIVAS + 1):
        tentativas = tentativa
        status, erro, dados = None, None, {}
        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=_timeout_s()) as cliente:
                resposta = await cliente.post(url, json=corpo, headers=cabecalhos)
            status = resposta.status_code
            try:
                dados = resposta.json()
            except ValueError:
                dados = {"_bruto": resposta.text[:2000]}
            if status >= 400:
                erro = f"http_{status}"
        except httpx.HTTPError as exc:
            erro = f"comunicacao:{type(exc).__name__}"
            status = None
        if erro is None:
            break
        if not _deve_repetir(status, erro) or tentativa == TENTATIVAS:
            break
        await asyncio.sleep(ESPERAS_S[min(tentativa - 1, len(ESPERAS_S) - 1)])
    latencia = time.perf_counter() - inicio

    escolha = ((dados.get("choices") or [{}])[0]) if isinstance(dados, dict) else {}
    mensagem = escolha.get("message") or {}
    texto = str(mensagem.get("content") or "")
    uso = dados.get("usage") or {} if isinstance(dados, dict) else {}
    detalhes = uso.get("completion_tokens_details") or {}
    tokens_entrada = int(uso.get("prompt_tokens") or 0)
    tokens_saida = int(uso.get("completion_tokens") or 0)
    tokens_raciocinio = int(detalhes.get("reasoning_tokens") or 0)
    if erro is not None:
        # CONTRATO, regra 3: esgotadas as retentativas, o erro exposto e
        # "erro_chamada"; o detalhe do transporte fica em erro_detalhe.
        erro_detalhe = erro
        erro = "erro_chamada"
    else:
        erro_detalhe = None
    tabela = configuracao.custo_tabela_usd(id_modelo, tokens_entrada, tokens_saida)
    cobrado = uso.get("cost")
    if cobrado is None:
        cobrado = tabela
    m = configuracao.modelo(id_modelo)
    chamada = {
        "papel": papel,
        "modelo": id_modelo,
        "provedor_solicitado": m.get("provedor_slug"),
        "provedor_servico": dados.get("provider") if isinstance(dados, dict) else None,
        "modelo_api": dados.get("model") if isinstance(dados, dict) else None,
        "generation_id": dados.get("id") if isinstance(dados, dict) else None,
        "system_fingerprint": dados.get("system_fingerprint") if isinstance(dados, dict) else None,
        "tentativa_final": tentativas,
        "tentativas": tentativas,
        "http_status": status,
        "erro": erro,
        "erro_detalhe": erro_detalhe,
        "finish_reason": escolha.get("finish_reason"),
        "tokens_entrada": tokens_entrada,
        "tokens_saida": tokens_saida,
        "tokens_raciocinio": tokens_raciocinio,
        "is_byok": bool(uso.get("is_byok")) if isinstance(uso, dict) else False,
        "custo_cobrado_usd": float(cobrado or 0.0),
        "latencia_s": round(latencia, 6),
        "hash_corpo": hash_corpo(corpo),
    }
    return chamada, texto


def custo_tabela_chamada(chamada: Dict[str, Any]) -> float:
    """Custo de tabela de uma chamada, recalculado dos tokens com os precos da configuracao."""
    return configuracao.custo_tabela_usd(
        str(chamada.get("modelo") or ""),
        int(chamada.get("tokens_entrada") or 0),
        int(chamada.get("tokens_saida") or 0),
    )
