# -*- coding: utf-8 -*-
# Objective: Per-state independent router state for the RBIE contract surface.
"""Estados independents do contrato: criacao, copia, instantaneo, restauracao e pausa.

Cada estado guarda os componentes que o artigo atribui ao roteador: populacao do
NSGA-II, centroides, estatisticas por centroide e modelo, meta-bandit, controlador
proporcional, historico usado pelos objetivos e o estado dos geradores aleatorios
(``random`` e ``numpy.random``). A copia e integral, inclusive desses geradores,
para que duas repeticoes do mesmo estado aquecido partam do mesmo ponto.

O hash cobre o estado aprendido e exclui apenas ``pausado``: a pausa e controle do
relogio do ciclo, e um estado restaurado comeca pausado (CONTRATO, item 6), sem que
o hash precise divergir do instantaneo.

A persistencia e melhor esforco: os estados vivem em memoria e sao espelhados no
Redis (``rbie:estado:<id>``) para sobreviver a reinicio do processo.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from . import configuracao

RAIZ_ARISTO = configuracao.RAIZ_ARISTO
DIR_ESTADO = Path(os.environ.get("ARISTO_ESTADO_DIR", str(RAIZ_ARISTO / "state" / "rbie")))
DIR_INSTANTANEOS = DIR_ESTADO / "instantaneos"

#: Serializa operacoes de estado no processo (o harness nao envia concorrencia).
TRAVA = threading.Lock()

_ESTADOS: Dict[str, Dict[str, Any]] = {}
_INSTANTANEOS: Dict[str, Dict[str, Any]] = {}
_CARREGADO = False

PREFIXO_REDIS = "rbie:estado:"
PREFIXO_INSTANTANEO_REDIS = "rbie:instantaneo:"


# --------------------------------------------------------------------------
# Redis (persistencia melhor esforco)
# --------------------------------------------------------------------------

def _rds():
    """Cliente Redis sincrono; ``None`` quando indisponivel."""
    try:
        import redis

        from app.settings_dynamic import settings

        cliente = redis.Redis(
            host=str(settings.get("REDIS_HOST", "127.0.0.1")),
            port=int(settings.get("REDIS_PORT", 6379) or 6379),
            password=(settings.get("REDIS_PASSWORD") or None),
            db=int(settings.get("REDIS_DB", 0) or 0),
            socket_timeout=2,
        )
        cliente.ping()
        return cliente
    except Exception:
        return None


def _gravar_redis(chave: str, dados: Dict[str, Any]) -> None:
    rds = _rds()
    if rds is None:
        return
    try:
        rds.set(chave, json.dumps(dados, ensure_ascii=False, default=str))
    except Exception:
        pass


def _ler_redis(chave: str) -> Optional[Dict[str, Any]]:
    rds = _rds()
    if rds is None:
        return None
    try:
        bruto = rds.get(chave)
        if not bruto:
            return None
        dados = json.loads(bruto.decode() if isinstance(bruto, bytes) else bruto)
        return dados if isinstance(dados, dict) else None
    except Exception:
        return None


def _carregar_uma_vez() -> None:
    """Reconstroi os estados e instantaneos a partir do Redis, na primeira operacao."""
    global _CARREGADO
    if _CARREGADO:
        return
    _CARREGADO = True
    rds = _rds()
    if rds is None:
        return
    try:
        for chave in rds.scan_iter(match=PREFIXO_REDIS + "*"):
            nome = chave.decode() if isinstance(chave, bytes) else str(chave)
            dados = _ler_redis(nome)
            if isinstance(dados, dict) and "id_estado" in dados:
                _ESTADOS[str(dados["id_estado"])] = dados
        for chave in rds.scan_iter(match=PREFIXO_INSTANTANEO_REDIS + "*"):
            nome = chave.decode() if isinstance(chave, bytes) else str(chave)
            id_instantaneo = nome[len(PREFIXO_INSTANTANEO_REDIS):]
            dados = _ler_redis(nome)
            if isinstance(dados, dict):
                _INSTANTANEOS[id_instantaneo] = dados
    except Exception:
        pass


# --------------------------------------------------------------------------
# Utilitarios de hash, tempo e RNG
# --------------------------------------------------------------------------

def _canonico(dado: Any) -> str:
    return json.dumps(dado, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def carimbo_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def hash_estado(estado: Dict[str, Any]) -> str:
    """SHA-256 do estado aprendido, sem os campos de controle do relogio e da idempotencia.

    Ficam fora do hash: ``pausado`` (um estado restaurado comeca pausado, item 6
    do CONTRATO), ``idempotencia`` (mapa de respostas ja devolvidas, bookkeeping
    do protocolo), ``id_estado`` (o nome do estado nao e estado aprendido: copia
    e origem tem de ter o mesmo hash) e os campos internos ``_*`` do laco.
    """
    limpo = {k: v for k, v in estado.items()
             if k not in ("pausado", "idempotencia", "id_estado") and not k.startswith("_")}
    return hashlib.sha256(_canonico(limpo).encode("utf8")).hexdigest()


def _rng_para_json() -> Dict[str, Any]:
    """Serializa o estado de ``random`` e ``numpy.random`` (incluido na copia)."""
    versao, interno, gauss = random.getstate()
    estado_np = np.random.get_state()
    return {
        "random": [int(versao), [int(x) for x in interno], gauss],
        "numpy": [str(estado_np[0]), [int(x) for x in estado_np[1]], int(estado_np[2]),
                  int(estado_np[3]), float(estado_np[4])],
    }


def _rng_de_json(dados: Optional[Dict[str, Any]]) -> None:
    """Restaura o estado dos geradores aleatorios a partir da serializacao."""
    if not dados:
        return
    try:
        r = dados.get("random")
        if r:
            random.setstate((int(r[0]), tuple(int(x) for x in r[1]), r[2]))
        n = dados.get("numpy")
        if n:
            np.random.set_state((str(n[0]), np.array([int(x) for x in n[1]], dtype=np.uint32),
                                 int(n[2]), int(n[3]), float(n[4])))
    except Exception:
        pass


def _pesos_padrao() -> Dict[str, float]:
    pesos = (configuracao.parametros().get("pesos_padrao") or {})
    return {"w_q": float(pesos.get("w_q", 0.55)), "w_l": float(pesos.get("w_l", 0.30)),
            "w_c": float(pesos.get("w_c", 0.15))}


def novo_estado(id_estado: str) -> Dict[str, Any]:
    """Estado vazio: pausado, com os pesos padrao e as estatisticas dos seis modelos no prior."""
    candidatos = configuracao.ids_candidatos()
    global_ctx = {
        modelo: {"mean": 0.0, "count": 0, "var": 0.0, "M2": 0.0, "alpha": 1.0, "beta": 1.0}
        for modelo in candidatos
    }
    return {
        "id_estado": id_estado,
        "criado_utc": carimbo_utc(),
        "pausado": True,
        "n_requisicoes": 0,
        "ciclos_evolutivos": 0,
        "ultimo_ciclo_utc": None,
        "politica": "centroide",
        "aprender": True,
        "pesos": _pesos_padrao(),
        "origem_pesos": "padrao",
        "tamanho_populacao": int(os.environ.get("ARISTO_NSGA_POP", "40") or 40),
        "populacao": [],
        "centroides": [],
        "estatisticas": {"global": global_ctx},
        "meta_bandit": {"estrategia_preferida": "ucb1", "desempenho": {}},
        "controlador": {"erro": 0.0, "integral": 0.0},
        "historico": [],
        "ema": {},
        "rng": _rng_para_json(),
        "idempotencia": {},
    }


# --------------------------------------------------------------------------
# Operacoes do contrato
# --------------------------------------------------------------------------

def listar_ids() -> list[str]:
    """Ids de todos os estados conhecidos (memoria + Redis)."""
    _carregar_uma_vez()
    return list(_ESTADOS.keys())


def existe(id_estado: str) -> bool:
    _carregar_uma_vez()
    return id_estado in _ESTADOS


def obter(id_estado: str) -> Dict[str, Any]:
    _carregar_uma_vez()
    if id_estado not in _ESTADOS:
        raise KeyError(f"estado inexistente: {id_estado}")
    return _ESTADOS[id_estado]


def salvar(estado: Dict[str, Any]) -> None:
    _ESTADOS[str(estado["id_estado"])] = estado
    _gravar_redis(PREFIXO_REDIS + str(estado["id_estado"]), estado)


def criar(id_estado: str, de_instantaneo: Optional[str] = None,
          politica: Optional[str] = None, alpha: Optional[float] = None,
          n0: Optional[float] = None, aprender: Optional[bool] = None) -> Dict[str, Any]:
    """Cria um estado vazio ou a partir de um instantaneo; sempre comeca pausado.

    ``politica`` escolhe a decisao por requisicao: ``centroide`` (padrao) ou
    ``linucb``. ``alpha`` e ``n0`` so valem para o LinUCB e ficam registrados no
    estado. ``aprender=False`` congela o bandit (escolhe sem atualizar A e b),
    como na fase de teste do experimento.
    """
    _carregar_uma_vez()
    if de_instantaneo is not None:
        payload = _INSTANTANEOS.get(de_instantaneo)
        if payload is None:
            payload = _ler_redis(PREFIXO_INSTANTANEO_REDIS + de_instantaneo)
        if payload is None:
            raise KeyError(f"instantaneo inexistente: {de_instantaneo}")
        estado = copy.deepcopy(payload)
        estado["id_estado"] = id_estado
        estado["pausado"] = True
        estado["idempotencia"] = {}
    else:
        estado = novo_estado(id_estado)
    if politica:
        estado["politica"] = str(politica)
    if aprender is not None:
        estado["aprender"] = bool(aprender)
    if estado.get("politica") == "linucb":
        from . import linucb

        linucb.inicializar(estado, configuracao.ids_candidatos(), alpha, n0)
    elif alpha is not None:
        estado.setdefault("linucb", {})["alpha"] = float(alpha)
    salvar(estado)
    return estado


def copiar(origem: str, destino: str) -> tuple[str, str]:
    """Copia integral do estado, inclusive o estado dos geradores aleatorios."""
    estado = obter(origem)
    copia = copy.deepcopy(estado)
    copia["id_estado"] = destino
    copia["idempotencia"] = {}
    salvar(copia)
    return hash_estado(estado), hash_estado(copia)


def instantaneo(id_estado: str, id_instantaneo: str) -> tuple[str, str]:
    """Grava uma serializacao imutavel; devolve (hash, caminho)."""
    estado = obter(id_estado)
    payload = copy.deepcopy(estado)
    _INSTANTANEOS[id_instantaneo] = payload
    DIR_INSTANTANEOS.mkdir(parents=True, exist_ok=True)
    caminho = DIR_INSTANTANEOS / f"{id_instantaneo}.json"
    caminho.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf8")
    _gravar_redis(PREFIXO_INSTANTANEO_REDIS + id_instantaneo, payload)
    return hash_estado(payload), str(caminho)


def restaurar(id_estado: str, id_instantaneo: str) -> str:
    """Substitui o estado pelo instantaneo; comeca pausado e devolve o hash do instantaneo."""
    payload = _INSTANTANEOS.get(id_instantaneo)
    if payload is None:
        payload = _ler_redis(PREFIXO_INSTANTANEO_REDIS + id_instantaneo)
    if payload is None:
        raise KeyError(f"instantaneo inexistente: {id_instantaneo}")
    estado = copy.deepcopy(payload)
    estado["id_estado"] = id_estado
    estado["pausado"] = True
    salvar(estado)
    return hash_estado(payload)


def pausar(id_estado: str) -> None:
    obter(id_estado)["pausado"] = True
    salvar(_ESTADOS[id_estado])


def retomar(id_estado: str) -> None:
    obter(id_estado)["pausado"] = False
    salvar(_ESTADOS[id_estado])


def _resumo_linucb(estado: Dict[str, Any]) -> Dict[str, Any]:
    from . import linucb

    return linucb.resumo(estado)


def resumo(id_estado: str) -> Dict[str, Any]:
    estado = obter(id_estado)
    return {
        "id_estado": id_estado,
        "hash_estado": hash_estado(estado),
        "n_requisicoes": int(estado.get("n_requisicoes", 0)),
        "n_centroides": len(estado.get("centroides") or []),
        "ciclos_evolutivos": int(estado.get("ciclos_evolutivos", 0)),
        "ultimo_ciclo_utc": estado.get("ultimo_ciclo_utc"),
        "pausado": bool(estado.get("pausado", True)),
        "tamanho_populacao": int(estado.get("tamanho_populacao", 40) or 40),
        "pesos_atuais": dict(estado.get("pesos") or {}),
        "origem_pesos_atual": estado.get("origem_pesos", "padrao"),
        "politica": estado.get("politica", "centroide"),
        "alpha": (float((estado.get("linucb") or {}).get("alpha"))
                  if estado.get("politica") == "linucb" else None),
        "n0": (float((estado.get("linucb") or {}).get("n0"))
               if estado.get("politica") == "linucb" else None),
        "aprender": bool(estado.get("aprender", True)),
        "linucb": (_resumo_linucb(estado) if estado.get("politica") == "linucb" else None),
    }


# --------------------------------------------------------------------------
# RNG por estado e idempotencia
# --------------------------------------------------------------------------

def entrar(estado: Dict[str, Any]) -> None:
    """Carrega o estado dos geradores aleatorios daquele estado."""
    _rng_de_json(estado.get("rng"))


def sair(estado: Dict[str, Any]) -> None:
    """Salva o estado dos geradores aleatorios de volta no estado."""
    estado["rng"] = _rng_para_json()


def resposta_guardada(estado: Dict[str, Any], id_requisicao: str) -> Optional[Dict[str, Any]]:
    return (estado.get("idempotencia") or {}).get(id_requisicao)


def guardar_resposta(estado: Dict[str, Any], id_requisicao: str, resposta: Dict[str, Any]) -> None:
    estado.setdefault("idempotencia", {})[id_requisicao] = copy.deepcopy(resposta)


def agora_epoch() -> float:
    return time.time()
