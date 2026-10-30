# -*- coding: utf-8 -*-
# Objective: Provider client for the RBIE contract surface (OpenRouter and Vertex).
"""Cliente do roteador para os candidatos e juizes internos.

Duas vias de acesso, como no harness:

- ``openrouter``: corpo de chat completions com ``messages``, ``provider.order``
  fixado, ``allow_fallbacks: false``, temperatura 0,4, ``max_tokens`` e ``seed``;
- ``vertex``: familia Gemini pela Vertex AI, autenticada por ADC. O corpo e o de
  ``corpo_requisicao.corpo_gerador`` para a via Vertex
  (``{"model", "contents", "config"}``, com ``max_output_tokens``,
  ``temperature``, ``seed`` e ``thinking_config``); a chamada usa o SDK
  ``google-genai``. Como a Vertex nao devolve custo cobrado, os tokens vem de
  ``usage_metadata`` (raciocinio contado como saida) e o custo e estimado pela
  tabela de precos da configuracao, com ``origem_custo`` explicito.

Retentativas, em ambas as vias: ate 4 tentativas, com espera de 5, 10 e 20 s,
apenas em erro de comunicacao (tempo esgotado, conexao, HTTP 408, 429 e 5xx).
Esgotadas, ``erro = "erro_chamada"`` e texto vazio.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

import httpx

from . import configuracao

TENTATIVAS = 4
ESPERAS_S = (5.0, 10.0, 20.0)

_CLIENTE_VERTEX = None
_TRAVA_VERTEX = threading.Lock()

MAPA_FINISH_VERTEX = {
    "STOP": "stop", "MAX_TOKENS": "length", "SAFETY": "safety",
    "RECITATION": "recitation", "OTHER": "other",
    "FINISH_REASON_UNSPECIFIED": None,
}


def hash_corpo(corpo: Dict[str, Any]) -> str:
    """SHA-256 do JSON canonico do corpo (chaves ordenadas, sem espacos)."""
    canonico = json.dumps(corpo, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonico.encode("utf8")).hexdigest()


# --------------------------------------------------------------------------
# Corpos
# --------------------------------------------------------------------------

def corpo_gerador(id_modelo: str, prompt: str, semente: Optional[int]) -> Dict[str, Any]:
    """Corpo de uma chamada a candidato, identico ao de ``corpo_requisicao.corpo_gerador``."""
    m = configuracao.modelo(id_modelo)
    via = configuracao.via_modelo(id_modelo)
    limite = configuracao.limite_geracao(id_modelo)
    aceita_temperatura = bool(m.get("aceita_temperatura", True))
    temperatura = configuracao.geracao(id_modelo).get("temperatura")
    if temperatura is None:
        temperatura = float((configuracao.configuracao_bruta().get("experimento") or {})
                            .get("temperatura_geradores", 0.4))
    aceita_seed = bool(m.get("aceita_seed", True))
    raciocinio = m.get("raciocinio")

    if via == "vertex":
        conf: Dict[str, Any] = {"max_output_tokens": limite}
        if aceita_temperatura:
            conf["temperature"] = float(temperatura)
        if aceita_seed and semente is not None:
            conf["seed"] = int(semente)
        if raciocinio:
            conf["thinking_config"] = dict(raciocinio)
        return {"model": id_modelo, "contents": prompt, "config": conf}

    # Via OpenRouter (padrao tambem para o modo seco do servidor falso).
    corpo: Dict[str, Any] = {
        "model": id_modelo,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": limite,
        "provider": {"order": [m.get("provedor_slug")], "allow_fallbacks": False},
        "usage": {"include": True},
    }
    if aceita_temperatura:
        corpo["temperature"] = float(temperatura)
    if raciocinio:
        corpo["reasoning"] = raciocinio
    if aceita_seed and semente is not None:
        corpo["seed"] = int(semente)
    return corpo


def corpo_juiz(id_modelo: str, prompt: str, max_tokens: Optional[int] = None) -> Dict[str, Any]:
    """Corpo de uma chamada a juiz interno: temperatura 0 e limite dos juizes (ou o informado)."""
    m = configuracao.modelo(id_modelo)
    juiz = configuracao.parametros_juiz()
    if max_tokens is not None:
        juiz["max_tokens"] = int(max_tokens)
    via = configuracao.via_modelo(id_modelo)
    if via == "vertex":
        conf: Dict[str, Any] = {"max_output_tokens": int(juiz["max_tokens"])}
        if m.get("aceita_temperatura", True):
            conf["temperature"] = float(juiz["temperatura"])
        if m.get("raciocinio"):
            conf["thinking_config"] = dict(m["raciocinio"])
        return {"model": id_modelo, "contents": prompt, "config": conf}
    corpo: Dict[str, Any] = {
        "model": id_modelo,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(juiz["max_tokens"]),
        "provider": {"order": [m.get("provedor_slug")], "allow_fallbacks": False},
        "usage": {"include": True},
    }
    if m.get("aceita_temperatura", True) and juiz["temperatura"] is not None:
        corpo["temperature"] = float(juiz["temperatura"])
    if m.get("raciocinio"):
        corpo["reasoning"] = m["raciocinio"]
    return corpo


# --------------------------------------------------------------------------
# HTTP (OpenRouter e servidor falso)
# --------------------------------------------------------------------------

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


async def _chamar_http(id_modelo: str, corpo: Dict[str, Any], papel: str,
                       via: str) -> Tuple[Dict[str, Any], str]:
    """Chamada HTTP: OpenRouter real ou servidor falso no modo seco."""
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
    uso = (dados.get("usage") or {}) if isinstance(dados, dict) else {}
    detalhes = uso.get("completion_tokens_details") or {}
    tokens_entrada = int(uso.get("prompt_tokens") or 0)
    tokens_saida = int(uso.get("completion_tokens") or 0)
    tokens_raciocinio = int(detalhes.get("reasoning_tokens") or 0)
    erro_detalhe = erro
    if erro is not None:
        erro = "erro_chamada"
    tabela = configuracao.custo_tabela_usd(id_modelo, tokens_entrada, tokens_saida)
    if via == "vertex":
        # A Vertex nao devolve custo cobrado: o valor e a estimativa de tabela.
        cobrado = tabela
        origem_custo = "estimado_tabela"
        provedor_solicitado = configuracao.provedor_modelo(id_modelo)
        is_byok = None
    else:
        cobrado = uso.get("cost")
        if cobrado is None:
            cobrado = tabela
            origem_custo = "estimado_tabela"
        else:
            origem_custo = "cobrado_agregador"
        provedor_solicitado = configuracao.modelo(id_modelo).get("provedor_slug")
        is_byok = bool(uso.get("is_byok")) if isinstance(uso, dict) else None
    chamada = {
        "papel": papel,
        "modelo": id_modelo,
        "provedor_solicitado": provedor_solicitado,
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
        "is_byok": is_byok,
        "custo_cobrado_usd": float(cobrado or 0.0),
        "origem_custo": origem_custo,
        "latencia_s": round(latencia, 6),
        "hash_corpo": hash_corpo(corpo),
    }
    return chamada, texto


# --------------------------------------------------------------------------
# Vertex AI (google-genai + ADC)
# --------------------------------------------------------------------------

def _cliente_vertex():
    """Cliente google-genai na Vertex, criado uma vez por processo."""
    global _CLIENTE_VERTEX
    if _CLIENTE_VERTEX is not None:
        return _CLIENTE_VERTEX
    with _TRAVA_VERTEX:
        if _CLIENTE_VERTEX is not None:
            return _CLIENTE_VERTEX
        from google import genai

        vertex = configuracao.vertex_parametros()
        env_projeto = vertex.get("projeto_env")
        env_local = vertex.get("localizacao_env")
        projeto = vertex.get("projeto") or (os.environ.get(env_projeto) if env_projeto else None)
        localizacao = (vertex.get("localizacao")
                       or (os.environ.get(env_local) if env_local else None) or "global")
        if not projeto:
            raise RuntimeError("defina vertex.projeto na configuracao ou GOOGLE_CLOUD_PROJECT")
        _CLIENTE_VERTEX = genai.Client(vertexai=True, project=projeto, location=localizacao)
    return _CLIENTE_VERTEX


def _normalizar_finish_vertex(finish: Any) -> Optional[str]:
    """Normaliza o enum de finish_reason da Vertex para ``stop``/``length``/etc."""
    if finish is None:
        return None
    nome = getattr(finish, "name", None)
    if nome is None:
        nome = getattr(finish, "value", None)
    if nome is None:
        nome = str(finish)
    nome = str(nome).split(".")[-1].upper()
    if nome in MAPA_FINISH_VERTEX:
        return MAPA_FINISH_VERTEX[nome]
    return str(nome).lower()


def _codigo_erro(exc: BaseException) -> Optional[int]:
    for atributo in ("code", "status_code", "http_status"):
        valor = getattr(exc, atributo, None)
        if callable(valor):
            try:
                valor = valor()
            except Exception:
                continue
        if isinstance(valor, int):
            return valor
    return None


def _retentavel_vertex(exc: BaseException) -> bool:
    """Erro de comunicacao, 408, 429 ou 5xx: repete; 400 a 404 nao."""
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    try:
        from google.api_core import exceptions as gexc

        if isinstance(exc, (gexc.ServiceUnavailable, gexc.DeadlineExceeded,
                            gexc.ResourceExhausted, gexc.InternalServerError,
                            gexc.BadGateway, gexc.GatewayTimeout, gexc.TooManyRequests)):
            return True
    except Exception:
        pass
    codigo = _codigo_erro(exc)
    return codigo is not None and (codigo in (408, 429) or codigo >= 500)


async def _chamar_vertex(id_modelo: str, corpo: Dict[str, Any],
                         papel: str) -> Tuple[Dict[str, Any], str]:
    """Chamada sincrona a Vertex, com a mesma politica de retentativas do contrato."""
    inicio = time.perf_counter()
    provedor = configuracao.provedor_modelo(id_modelo)
    tentativas = 0
    erro: Optional[str] = None
    erro_detalhe: Optional[str] = None
    resposta = None
    while tentativas < TENTATIVAS:
        tentativas += 1
        erro = None
        try:
            cliente = _cliente_vertex()
            resposta = await asyncio.to_thread(
                cliente.models.generate_content,
                model=id_modelo,
                contents=corpo.get("contents"),
                config=corpo.get("config"),
            )
            break
        except BaseException as exc:  # noqa: BLE001 - erro de chamada vira Resposta
            erro_detalhe = repr(exc)
            erro = "erro_chamada"
            if not _retentavel_vertex(exc) or tentativas == TENTATIVAS:
                break
            await asyncio.sleep(ESPERAS_S[min(tentativas - 1, len(ESPERAS_S) - 1)])
    latencia = time.perf_counter() - inicio

    texto = ""
    tokens_entrada = 0
    tokens_saida = 0
    tokens_raciocinio = 0
    finish_reason = None
    modelo_api = None
    generation_id = None
    if resposta is not None and erro is None:
        texto = str(getattr(resposta, "text", "") or "")
        uso = getattr(resposta, "usage_metadata", None)
        tokens_entrada = int(getattr(uso, "prompt_token_count", 0) or 0)
        tokens_raciocinio = int(getattr(uso, "thoughts_token_count", 0) or 0)
        # tokens de saida incluem os de raciocinio (ESPECIFICACAO, Secao 10)
        tokens_saida = int(getattr(uso, "candidates_token_count", 0) or 0) + tokens_raciocinio
        candidato = (getattr(resposta, "candidates", None) or [None])[0]
        finish_reason = _normalizar_finish_vertex(getattr(candidato, "finish_reason", None))
        modelo_api = getattr(resposta, "model_version", None)
        generation_id = getattr(resposta, "response_id", None)
    tabela = configuracao.custo_tabela_usd(id_modelo, tokens_entrada, tokens_saida)
    chamada = {
        "papel": papel,
        "modelo": id_modelo,
        "provedor_solicitado": provedor,
        "provedor_servico": provedor,
        "modelo_api": modelo_api,
        "generation_id": generation_id,
        "system_fingerprint": None,
        "tentativa_final": tentativas,
        "tentativas": tentativas,
        "http_status": 200 if erro is None else None,
        "erro": erro,
        "erro_detalhe": erro_detalhe,
        "finish_reason": finish_reason,
        "tokens_entrada": tokens_entrada,
        "tokens_saida": tokens_saida,
        "tokens_raciocinio": tokens_raciocinio,
        "is_byok": None,
        "custo_cobrado_usd": float(tabela),
        "origem_custo": "estimado_tabela",
        "latencia_s": round(latencia, 6),
        "hash_corpo": hash_corpo(corpo),
    }
    return chamada, texto


# --------------------------------------------------------------------------
# Ponto de entrada
# --------------------------------------------------------------------------

async def chamar(id_modelo: str, corpo: Dict[str, Any],
                 *, papel: str) -> Tuple[Dict[str, Any], str]:
    """Envia o corpo pela via do modelo e devolve ``(chamada, texto)``."""
    via = configuracao.via_modelo(id_modelo)
    if via == "vertex" and not configuracao.modo_seco():
        return await _chamar_vertex(id_modelo, corpo, papel)
    return await _chamar_http(id_modelo, corpo, papel, via)


def custo_tabela_chamada(chamada: Dict[str, Any]) -> float:
    """Custo de tabela de uma chamada, recalculado dos tokens com os precos da configuracao."""
    return configuracao.custo_tabela_usd(
        str(chamada.get("modelo") or ""),
        int(chamada.get("tokens_entrada") or 0),
        int(chamada.get("tokens_saida") or 0),
    )
