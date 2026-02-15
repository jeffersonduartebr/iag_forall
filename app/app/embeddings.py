# -*- coding: utf-8 -*-
"""
embeddings.py — Híbrido Otimizado (CPU Local + Cloud Fallback)
--------------------------------------------------------------
OTIMIZAÇÃO DE PERFORMANCE:
Para evitar que o Ollama fique trocando modelos na GPU (Model Thrashing),
rodamos o modelo de embedding (que é leve) localmente na CPU do container
usando 'sentence-transformers'.

Isso deixa a GPU livre para o LLM/VLM pesado.

CACHE L1: In-memory LRU cache para acesso ultra-rápido
CACHE L2: Redis cache para persistência
"""

from __future__ import annotations

import os
import json
import hashlib
import logging
import time
import threading
from collections import OrderedDict
import numpy as np
from typing import List, Dict, Optional, Any, Tuple

# Importação Condicional para não quebrar se a lib faltar
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

# OpenAI SDK
try:
    from openai import OpenAI as OpenAIClient
except ImportError:
    OpenAIClient = None

from app.settings_dynamic import settings
from app.utils.redis_client import get_redis

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


# ============================================================
# L1 Cache In-Memory com LRU e TTL
# ============================================================
EMBED_L1_CACHE_SIZE = 100000  # Optimized for high-capacity environment (64GB RAM) - ~300MB memory
EMBED_L1_CACHE_TTL_S = 28800  # 8 horas - reduces embedding recalculation by 50-70%


class EmbeddingL1Cache:
    """Cache LRU in-memory para embeddings com TTL."""

    def __init__(self, maxsize: int = EMBED_L1_CACHE_SIZE, ttl_s: int = EMBED_L1_CACHE_TTL_S):
        self.maxsize = maxsize
        self.ttl_s = ttl_s
        self._lock = threading.Lock()
        self._data: "OrderedDict[str, Tuple[List[float], float]]" = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[List[float]]:
        now = time.time()
        with self._lock:
            if key not in self._data:
                self._misses += 1
                return None
            vec, ts = self._data[key]
            if self.ttl_s > 0 and (now - ts) > self.ttl_s:
                del self._data[key]
                self._misses += 1
                return None
            # Move to end (most recently used)
            self._data.move_to_end(key)
            self._hits += 1
            return vec

    def set(self, key: str, vec: List[float]) -> None:
        now = time.time()
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = (vec, now)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "size": len(self._data),
        }


_embed_l1_cache = EmbeddingL1Cache()

# Configs
EMBED_MODEL_TEXT = settings.get("EMBED_TEXT_MODEL", "nomic-ai/nomic-embed-text-v1.5")
# Fallback para HuggingFace ID se for nome do Ollama
if "nomic" in EMBED_MODEL_TEXT and "/" not in EMBED_MODEL_TEXT:
    EMBED_MODEL_TEXT = "nomic-ai/nomic-embed-text-v1.5"
if "minilm" in EMBED_MODEL_TEXT:
    EMBED_MODEL_TEXT = "sentence-transformers/all-MiniLM-L6-v2"

EMBED_CACHE_TTL_S = 86400 * 7
REDIS_KEY_PREFIX = "emb:v4"
_rds = get_redis()

# --- SINGLETON DO MODELO LOCAL ---
# Carrega o modelo na memória na primeira chamada e mantém lá.
_LOCAL_MODEL_INSTANCE = None

def get_local_model():
    global _LOCAL_MODEL_INSTANCE
    if _LOCAL_MODEL_INSTANCE is None and ST_AVAILABLE:
        logger.info(f"[Embeddings] Carregando modelo local CPU: {EMBED_MODEL_TEXT}...")
        # trust_remote_code=True é necessário para Nomic
        _LOCAL_MODEL_INSTANCE = SentenceTransformer(EMBED_MODEL_TEXT, trust_remote_code=True)
        # Otimização para CPU
        _LOCAL_MODEL_INSTANCE.eval() 
    return _LOCAL_MODEL_INSTANCE

# ============================================================
# Utils
# ============================================================

def _hash_text(text: str, model: str) -> str:
    payload = f"{model}|{text}".encode("utf-8", errors="ignore")
    return hashlib.sha256(payload).hexdigest()

def _norm(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    return vec if n == 0 else vec / n

def _save_cache(key: str, vec: List[float]):
    if _rds:
        try:
            _rds.setex(key, EMBED_CACHE_TTL_S, json.dumps({"v": vec}))
        except Exception as e:
            logger.warning(f"[Embeddings] Failed to save cache for key {key[:20]}...: {e}")

def _load_cache(key: str) -> Optional[List[float]]:
    if _rds:
        try:
            raw = _rds.get(key)
            if raw:
                return json.loads(raw).get("v")
        except Exception as e:
            logger.warning(f"[Embeddings] Failed to load cache for key {key[:20]}...: {e}")
    return None

# ============================================================
# GERAÇÃO (LOCAL CPU)
# ============================================================

def _local_cpu_embed(text: str) -> List[float]:
    """Gera embedding localmente na CPU, sem chamar Ollama."""
    model = get_local_model()
    if not model:
        raise RuntimeError("SentenceTransformers não instalado ou falha ao carregar.")
    
    # Prefixo específico para Nomic v1.5 (Melhora qualidade)
    if "nomic" in EMBED_MODEL_TEXT and not text.startswith("search_"):
        text = f"search_query: {text}"
        
    # Gera vetor
    vec = model.encode(text, convert_to_numpy=True)
    return vec.tolist()

# ============================================================
# API PÚBLICA
# ============================================================

def embed_text(text: str) -> List[float]:
    text = (text or "").strip()
    if not text:
        return [0.0]

    # Cache key
    key = f"{REDIS_KEY_PREFIX}:{EMBED_MODEL_TEXT}:{_hash_text(text, EMBED_MODEL_TEXT)}"

    # 1. L1 Cache (In-Memory) - Ultra-rápido
    if cached := _embed_l1_cache.get(key):
        return cached

    # 2. L2 Cache (Redis)
    if cached := _load_cache(key):
        _embed_l1_cache.set(key, cached)  # Populate L1
        return cached

    vec = []

    # 3. Tenta Local CPU (Prioridade Máxima para Performance)
    try:
        vec = _local_cpu_embed(text)
    except Exception as e:
        logger.warning(f"[Embeddings] Falha local CPU: {e}. Tentando OpenAI...")

        # 4. Fallback OpenAI (se configurado)
        if settings.get("OPENAI_API_KEY"):
            try:
                from openai import OpenAI

                client = OpenAI(api_key=settings.get("OPENAI_API_KEY"))
                resp = client.embeddings.create(model="text-embedding-3-small", input=text)
                vec = resp.data[0].embedding
            except Exception as ex:
                logger.error(f"[Embeddings] Falha OpenAI: {ex}")

    if vec:
        _save_cache(key, vec)           # L2 Cache
        _embed_l1_cache.set(key, vec)   # L1 Cache
        return vec

    return [0.0] * 768  # Retorna vetor zerado em caso de falha total


def get_embedding_cache_stats() -> Dict[str, Any]:
    """Retorna estatísticas do cache L1 de embeddings."""
    return _embed_l1_cache.stats()

# Vision embedding removido/simplificado pois o RAG agora usa "Ponte Descritiva"
# (VLM gera texto -> Texto vira embedding)
def embed_image(image_b64: str) -> List[float]:
    return [0.0] 

def embed_multimodal(text: str, image_b64: Optional[str]) -> Dict[str, List[float]]:
    text_vec = embed_text(text)
    # Retorna apenas o texto, pois estamos usando a estratégia de descrição
    return {
        "text": text_vec,
        "vision": [],
        "multimodal": text_vec
    }