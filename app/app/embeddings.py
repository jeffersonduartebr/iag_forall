# -*- coding: utf-8 -*-
"""
embeddings.py
------------------------------------------------------------
Embeddings híbridos:
1) Tenta via Ollama (100% local)
2) Fallback para OpenAI (se falhar ou não disponível)

Recursos:
- Cache em Redis (opcional) com TTL
- Normalização L2 (opcional)
- Funções single e batch
- Retries com backoff
- Métricas leves via logging

Variáveis esperadas em settings_dynamic.py (todas com defaults seguros):
- EMBED_PROVIDER            : "hybrid" | "ollama" | "openai" (usamos "hybrid" aqui)
- EMBED_MODEL               : ex. "all-minilm:latest" (Ollama) ou "text-embedding-3-large" (OpenAI)
- OLLAMA_HOST               : ex. "http://ollama:11434"
- OPENAI_API_KEY            : sua chave (se for usar fallback OpenAI)
- EMBED_CACHE_ENABLED       : "true"/"false" (default: "true")
- EMBED_CACHE_TTL_S         : tempo de vida do cache em segundos (default: 86400 = 24h)
- EMBED_NORMALIZE           : "true"/"false" (default: "true")
- EMBED_MAX_RETRIES         : int (default: 2)
- EMBED_TIMEOUT_S           : int (default: 30)
"""

from __future__ import annotations

import os
import json
import time
import math
import hashlib
import logging
from typing import List, Tuple, Optional, Any

import numpy as np
import requests

# OpenAI SDK (usamos defensivo)
try:
    from openai import OpenAI as OpenAIClient  # type: ignore
except Exception:
    OpenAIClient = None  # type: ignore

from app.settings_dynamic import settings
from app.utils.redis_client import get_redis

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] embeddings: %(message)s")

# ============================================================
# Configurações (com defaults)
# ============================================================
EMBED_PROVIDER: str = str(settings.get("EMBED_PROVIDER", "hybrid")).lower()
EMBED_MODEL: str = str(settings.get("EMBED_MODEL", "all-minilm:latest"))  # modelo "local" por padrão
OLLAMA_HOST: str = str(settings.get("OLLAMA_HOST", os.getenv("OLLAMA_HOST", "http://ollama:11434")))
OPENAI_API_KEY: str = str(settings.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "")))

EMBED_CACHE_ENABLED: bool = str(settings.get("EMBED_CACHE_ENABLED", "true")).lower() in ("1", "true", "yes", "on")
EMBED_CACHE_TTL_S: int = int(settings.get("EMBED_CACHE_TTL_S", os.getenv("EMBED_CACHE_TTL_S", 86400)))
EMBED_NORMALIZE: bool = str(settings.get("EMBED_NORMALIZE", "true")).lower() in ("1", "true", "yes", "on")

EMBED_MAX_RETRIES: int = int(settings.get("EMBED_MAX_RETRIES", os.getenv("EMBED_MAX_RETRIES", 2)))
EMBED_TIMEOUT_S: int = int(settings.get("EMBED_TIMEOUT_S", os.getenv("EMBED_TIMEOUT_S", 30)))

# Redis opcional
_rds = get_redis() if EMBED_CACHE_ENABLED else None
REDIS_KEY_PREFIX = "emb:v1"  # mude caso altere a estratégia de serialização


# ============================================================
# Utilidades
# ============================================================
def _hash_text(text: str) -> str:
    h = hashlib.sha256()
    # incluir o nome do modelo no hash para não misturar diferentes espaços vetoriais
    payload = f"{EMBED_MODEL}||{text}".encode("utf-8", errors="ignore")
    h.update(payload)
    return h.hexdigest()


def _norm(vec: np.ndarray) -> np.ndarray:
    if not EMBED_NORMALIZE:
        return vec
    denom = np.linalg.norm(vec)
    if denom == 0.0:
        return vec
    return vec / denom


def _save_cache(key: str, vec: List[float]) -> None:
    if not _rds:
        return
    try:
        payload = json.dumps({"v": vec, "ts": int(time.time())})
        _rds.setex(key, EMBED_CACHE_TTL_S, payload)
    except Exception as e:
        logger.debug(f"[cache] Falha ao gravar cache Redis: {e}")


def _load_cache(key: str) -> Optional[List[float]]:
    if not _rds:
        return None
    try:
        raw = _rds.get(key)
        if not raw:
            return None
        obj = json.loads(raw)
        if isinstance(obj, dict) and "v" in obj and isinstance(obj["v"], list):
            return [float(x) for x in obj["v"]]
    except Exception as e:
        logger.debug(f"[cache] Falha ao ler cache Redis: {e}")
    return None


def _redis_key_single(text: str) -> str:
    return f"{REDIS_KEY_PREFIX}:m={EMBED_MODEL}:h={_hash_text(text)}"


def _redis_key_batch(texts: List[str]) -> List[Tuple[str, str]]:
    # retorna lista [(key, text)]
    return [(f"{REDIS_KEY_PREFIX}:m={EMBED_MODEL}:h={_hash_text(t)}", t) for t in texts]


def _backoff_sleep(attempt: int) -> None:
    # 0,1,2,...  =>  0.5, 1.0, 2.0 (cap 4s)
    delay = min(4.0, 0.5 * (2 ** attempt))
    time.sleep(delay)


# ============================================================
# Clientes
# ============================================================
def _ollama_embed_single(text: str) -> List[float]:
    """Chama Ollama /api/embeddings para um texto."""
    url = f"{OLLAMA_HOST.rstrip('/')}/api/embeddings"
    payload = {"model": EMBED_MODEL, "prompt": text}
    r = requests.post(url, json=payload, timeout=EMBED_TIMEOUT_S)
    r.raise_for_status()
    data = r.json() or {}
    vec = data.get("embedding")
    if not isinstance(vec, list):
        raise RuntimeError("Resposta de embeddings do Ollama inválida.")
    return [float(x) for x in vec]


def _ollama_embed_batch(texts: List[str]) -> List[List[float]]:
    """Algumas builds do Ollama suportam embeddings batelados; se não, faz loop."""
    # Implementação segura: faz loop 1 a 1 (mantendo compatibilidade ampla).
    out: List[List[float]] = []
    for t in texts:
        out.append(_ollama_embed_single(t))
    return out


def _openai_embed_single(client: Any, text: str) -> List[float]:
    """Chama OpenAI embeddings.create."""
    # Modelo pode vir como "text-embedding-3-large" etc.
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    if not resp or not getattr(resp, "data", None):
        raise RuntimeError("Resposta de embeddings da OpenAI inválida.")
    vec = resp.data[0].embedding
    return [float(x) for x in vec]


def _openai_client() -> Any:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY ausente para fallback.")
    if not OpenAIClient:
        raise RuntimeError("SDK OpenAI não disponível no ambiente.")
    return OpenAIClient(api_key=OPENAI_API_KEY)


# ============================================================
# API pública
# ============================================================
def get_embedding(text: str) -> np.ndarray:
    """
    Retorna um vetor np.ndarray (normalizado se EMBED_NORMALIZE=True).
    Fluxo:
      1) Cache Redis
      2) Ollama
      3) Fallback OpenAI (se falhar)
    """
    text = (text or "").strip()
    if not text:
        return np.zeros(1, dtype=np.float32)

    # 1) Cache
    key = _redis_key_single(text)
    if EMBED_CACHE_ENABLED:
        cached = _load_cache(key)
        if cached:
            return _norm(np.array(cached, dtype=np.float32))

    # 2) Tentativa via Ollama com retries
    last_exc: Optional[Exception] = None
    for attempt in range(EMBED_MAX_RETRIES + 1):
        try:
            vec = _ollama_embed_single(text)
            _save_cache(key, vec)
            return _norm(np.array(vec, dtype=np.float32))
        except Exception as exc:
            last_exc = exc
            logger.debug(f"[ollama] tentativa {attempt+1} falhou: {exc}")
            _backoff_sleep(attempt)

    # 3) Fallback OpenAI
    try:
        client = _openai_client()
    except Exception as exc:
        # Sem client OpenAI -> devolve erro original do Ollama para facilitar diagnóstico
        msg = f"Embeddings indisponíveis (Ollama falhou e OpenAI indisponível): {last_exc or 'erro desconhecido'}"
        logger.error(msg)
        raise RuntimeError(msg) from last_exc

    for attempt in range(EMBED_MAX_RETRIES + 1):
        try:
            vec = _openai_embed_single(client, text)
            _save_cache(key, vec)
            return _norm(np.array(vec, dtype=np.float32))
        except Exception as exc:
            last_exc = exc
            logger.debug(f"[openai] tentativa {attempt+1} falhou: {exc}")
            _backoff_sleep(attempt)

    msg = f"Falha ao obter embeddings (Ollama e OpenAI). Último erro: {last_exc}"
    logger.error(msg)
    raise RuntimeError(msg)


def get_embeddings(texts: List[str]) -> np.ndarray:
    """
    Retorna matriz (N, D) com embeddings para uma lista de textos.
    Aplica cache individual por item. Usa Ollama com retries; se
    fracassar totalmente, tenta fallback OpenAI por item.
    """
    texts = [t.strip() if isinstance(t, str) else "" for t in texts]
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)

    # Tenta carregar do cache primeiro
    keys = _redis_key_batch(texts)
    out: List[Optional[List[float]]] = [None] * len(texts)

    if EMBED_CACHE_ENABLED and _rds:
        try:
            pipe = _rds.pipeline()
            for k, _ in keys:
                pipe.get(k)
            raw_list = pipe.execute()
            for i, raw in enumerate(raw_list):
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                    if isinstance(obj, dict) and "v" in obj and isinstance(obj["v"], list):
                        out[i] = [float(x) for x in obj["v"]]
                except Exception:
                    continue
        except Exception as e:
            logger.debug(f"[cache] Falha em MGET pipeline: {e}")

    # Determina quais faltam
    missing_idx = [i for i, v in enumerate(out) if v is None]

    # Primeiro, tenta Ollama para os faltantes
    for i in missing_idx:
        txt = texts[i]
        if not txt:
            out[i] = [0.0]  # placeholder
            continue
        last_exc: Optional[Exception] = None
        for attempt in range(EMBED_MAX_RETRIES + 1):
            try:
                vec = _ollama_embed_single(txt)
                out[i] = vec
                # cache
                k = keys[i][0]
                _save_cache(k, vec)
                break
            except Exception as exc:
                last_exc = exc
                _backoff_sleep(attempt)
        # Se ainda None -> fallback OpenAI
        if out[i] is None:
            try:
                client = _openai_client()
            except Exception as exc:
                logger.error(f"[embeddings] OpenAI indisponível para fallback: {exc}")
                raise RuntimeError(f"OpenAI indisponível para fallback: {exc}") from last_exc
            for attempt in range(EMBED_MAX_RETRIES + 1):
                try:
                    vec = _openai_embed_single(client, txt)
                    out[i] = vec
                    k = keys[i][0]
                    _save_cache(k, vec)
                    break
                except Exception as exc:
                    last_exc = exc
                    _backoff_sleep(attempt)
            if out[i] is None:
                msg = f"Falha ao obter embedding para item {i} (Ollama e OpenAI). Último erro: {last_exc}"
                logger.error(msg)
                raise RuntimeError(msg)

    # Normaliza e empilha
    arrs = []
    for vec in out:
        v = np.array(vec if vec is not None else [0.0], dtype=np.float32)
        arrs.append(_norm(v))
    return np.vstack(arrs)


# ============================================================
# Helpers de alto nível (compat)
# ============================================================
def embed_text(text: str) -> np.ndarray:
    """Alias compatível com chamadas antigas."""
    return get_embedding(text)


def embed_many(texts: List[str]) -> np.ndarray:
    """Alias compatível com chamadas antigas."""
    return get_embeddings(texts)


# ============================================================
# Health/diagnóstico rápido
# ============================================================
def health() -> dict:
    """Retorna um snapshot leve do estado do provedor de embeddings."""
    return {
        "provider": EMBED_PROVIDER,
        "model": EMBED_MODEL,
        "ollama_host": OLLAMA_HOST,
        "cache_enabled": EMBED_CACHE_ENABLED,
        "normalize": EMBED_NORMALIZE,
        "max_retries": EMBED_MAX_RETRIES,
        "timeout_s": EMBED_TIMEOUT_S,
        "redis": bool(_rds),
        "openai_present": bool(OPENAI_API_KEY) and (OpenAIClient is not None),
    }


# Execução direta para teste manual rápido
if __name__ == "__main__":
    sample = "Teste de embedding híbrido."
    print("Health:", health())
    v = get_embedding(sample)
    print("Dimensão:", v.shape, "Norma:", float(np.linalg.norm(v)))
