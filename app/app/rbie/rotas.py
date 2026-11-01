# -*- coding: utf-8 -*-
# Objective: Additive HTTP routes for the RBIE router contract.
"""Rotas aditivas do contrato: ``/saude``, ``/configuracao``, ``/estados`` e ``/rotear``.

Nenhuma rota existente e alterada. Erros de validacao devolvem HTTP 400 com
``{"erro": "<mensagem>"}``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse

from . import configuracao, estado as estado_modulo, roteador

router = APIRouter(tags=["Contrato RBIE"])


def _erro(mensagem: str) -> JSONResponse:
    return JSONResponse(status_code=400, content={"erro": mensagem})


def _corpo(dados: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return dados if isinstance(dados, dict) else {}


@router.get("/saude")
def saude() -> Dict[str, Any]:
    """Commit avaliado, arvore do ARISTO e hash da configuracao efetiva."""
    return {
        "status": "ok",
        "commit": configuracao.commit_avaliado(),
        "arvore_limpa": configuracao.arvore_limpa(),
        "hash_configuracao": configuracao.hash_configuracao(),
    }


@router.get("/configuracao")
def configuracao_efetiva() -> Dict[str, Any]:
    """Configuracao efetiva comparada pelo adaptador com o ``config.yaml``."""
    return configuracao.configuracao_efetiva()


# --------------------------------------------------------------------------
# Estados
# --------------------------------------------------------------------------

@router.post("/estados")
def criar_estado(dados: Dict[str, Any] = Body(default={})) -> Any:
    corpo = _corpo(dados)
    id_estado = str(corpo.get("id_estado") or "").strip()
    if not id_estado:
        return _erro("id_estado obrigatorio")
    try:
        estado = estado_modulo.criar(
            id_estado, corpo.get("de_instantaneo"),
            politica=corpo.get("politica"),
            alpha=corpo.get("alpha"),
            n0=corpo.get("n0"),
            aprender=corpo.get("aprender"),
        )
    except KeyError as exc:
        return _erro(str(exc))
    return {"id_estado": id_estado, "hash_estado": estado_modulo.hash_estado(estado)}


@router.post("/estados/{id_estado}/copiar")
def copiar_estado(id_estado: str, dados: Dict[str, Any] = Body(default={})) -> Any:
    destino = str(_corpo(dados).get("destino") or "").strip()
    if not destino:
        return _erro("destino obrigatorio")
    try:
        origem_hash, destino_hash = estado_modulo.copiar(id_estado, destino)
    except KeyError as exc:
        return _erro(str(exc))
    return {"hash_origem": origem_hash, "hash_destino": destino_hash}


@router.post("/estados/{id_estado}/instantaneo")
def instantaneo(id_estado: str, dados: Dict[str, Any] = Body(default={})) -> Any:
    id_instantaneo = str(_corpo(dados).get("id_instantaneo") or "").strip()
    if not id_instantaneo:
        return _erro("id_instantaneo obrigatorio")
    try:
        hash_estado, caminho = estado_modulo.instantaneo(id_estado, id_instantaneo)
    except KeyError as exc:
        return _erro(str(exc))
    return {"id_instantaneo": id_instantaneo, "hash_estado": hash_estado, "caminho": caminho}


@router.post("/estados/{id_estado}/restaurar")
def restaurar(id_estado: str, dados: Dict[str, Any] = Body(default={})) -> Any:
    id_instantaneo = str(_corpo(dados).get("id_instantaneo") or "").strip()
    if not id_instantaneo:
        return _erro("id_instantaneo obrigatorio")
    try:
        hash_estado = estado_modulo.restaurar(id_estado, id_instantaneo)
    except KeyError as exc:
        return _erro(str(exc))
    return {"hash_estado": hash_estado}


@router.post("/estados/{id_estado}/pausar")
def pausar(id_estado: str) -> Any:
    try:
        estado_modulo.pausar(id_estado)
    except KeyError as exc:
        return _erro(str(exc))
    return {"id_estado": id_estado, "pausado": True}


@router.post("/estados/{id_estado}/retomar")
def retomar(id_estado: str) -> Any:
    try:
        estado_modulo.retomar(id_estado)
    except KeyError as exc:
        return _erro(str(exc))
    return {"id_estado": id_estado, "pausado": False}


@router.get("/estados/{id_estado}/resumo")
def resumo(id_estado: str) -> Any:
    try:
        return estado_modulo.resumo(id_estado)
    except KeyError as exc:
        return _erro(str(exc))


# --------------------------------------------------------------------------
# Roteamento
# --------------------------------------------------------------------------

@router.post("/rotear")
async def rotear_requisicao(dados: Dict[str, Any] = Body(default={})) -> Any:
    corpo = _corpo(dados)
    id_estado = str(corpo.get("id_estado") or "").strip()
    id_requisicao = str(corpo.get("id_requisicao") or "").strip()
    prompt = corpo.get("prompt")
    if not id_estado:
        return _erro("id_estado obrigatorio")
    if not id_requisicao:
        return _erro("id_requisicao obrigatorio")
    if not isinstance(prompt, str) or not prompt:
        return _erro("prompt obrigatorio")
    if bool(corpo.get("usar_recuperacao")):
        return _erro("recuperacao interna desligada neste experimento")
    if bool(corpo.get("usar_cache_semantico")):
        return _erro("cache semantico desligado neste experimento")
    semente = corpo.get("semente")
    texto_redacao = corpo.get("texto_redacao")
    notas_humanas = corpo.get("notas_humanas")
    if notas_humanas is not None and not isinstance(notas_humanas, (list, tuple)):
        return _erro("notas_humanas deve ser lista de cinco inteiros")
    aprender = corpo.get("aprender")
    try:
        resposta = await roteador.rotear(
            id_estado, id_requisicao, prompt,
            int(semente) if semente is not None else None,
            _corpo(corpo.get("metadados")),
            texto_redacao if isinstance(texto_redacao, str) else None,
            list(notas_humanas) if notas_humanas is not None else None,
            bool(aprender) if aprender is not None else None,
        )
    except KeyError as exc:
        return _erro(str(exc))
    return resposta
