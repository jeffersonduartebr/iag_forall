# -*- coding: utf-8 -*-
# Objective: Application runtime code for embeddings.
"""Generate and cache embeddings used by retrieval and semantic caching.

The embedding layer is optimized to keep heavy generation workloads away from
the main Ollama inference path when possible. It prefers a lightweight local
SentenceTransformer running on CPU and falls back to cloud embeddings only when
needed. The module also combines:

- an in-process L1 cache for repeated hot-path lookups
- a Redis-backed L2 cache for persistence across requests
- modality-specific helpers for text, image, and multimodal embeddings
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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
    # Optional dependency: fall back to a sentinel when the SDK is absent.
    OpenAIClient = None  # type: ignore[assignment,misc]

from app.services.rag_esquema import CarregadorComRecuo, com_prefixo
from app.settings_dynamic import settings
from app.utils.redis_client import get_redis

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


# ============================================================
# L1 Cache In-Memory com LRU e TTL
# ============================================================
# Vetores guardados em float32: ~3 KB por embedding de 768 dimensões (como list[float]
# seriam ~24 KB). No limite padrão, ~300 MB por processo.
EMBED_L1_CACHE_SIZE = int(os.getenv("EMBED_L1_CACHE_SIZE", "100000"))
EMBED_L1_CACHE_TTL_S = 28800  # 8 horas - reduces embedding recalculation by 50-70%


class EmbeddingL1Cache:
    """Store recent embeddings in process memory with LRU and TTL behavior.

    Vectors are kept as float32 arrays and returned as ``list[float]``, so the
    public contract of ``embed_text`` is unchanged.
    """

    def __init__(self, maxsize: int = EMBED_L1_CACHE_SIZE, ttl_s: int = EMBED_L1_CACHE_TTL_S):
        """Create an in-memory embedding cache with bounded size and freshness."""
        self.maxsize = maxsize
        self.ttl_s = ttl_s
        self._lock = threading.Lock()
        self._data: "OrderedDict[str, Tuple[np.ndarray, float]]" = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[List[float]]:
        """Return a cached embedding vector when the entry is still valid."""
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
        return vec.tolist()

    def set(self, key: str, vec: List[float]) -> None:
        """Store one embedding vector (copied as float32) in the in-memory cache."""
        arr = np.array(vec, dtype=np.float32)
        now = time.time()
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = (arr, now)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def stats(self) -> Dict[str, Any]:
        """Return hit, miss, and occupancy statistics for the L1 cache."""
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

# --- SINGLETON DO MODELO LOCAL (carga serializada, com recuo após falha) ---
_CARREGADOR = CarregadorComRecuo(f"embeddings {EMBED_MODEL_TEXT}")


def _carregar_modelo():
    # trust_remote_code=True é necessário para Nomic
    modelo = SentenceTransformer(EMBED_MODEL_TEXT, trust_remote_code=True)
    modelo.eval()
    return modelo


def get_local_model():
    """Lazily load and return the local SentenceTransformer embedding model (None while unavailable)."""
    return _CARREGADOR.obter(_carregar_modelo) if ST_AVAILABLE else None

# ============================================================
# Utils
# ============================================================

def _hash_text(text: str, model: str) -> str:
    """Hash one text/model pair into a stable cache key fragment."""
    payload = f"{model}|{text}".encode("utf-8", errors="ignore")
    return hashlib.sha256(payload).hexdigest()

def _norm(vec: np.ndarray) -> np.ndarray:
    """Normalize a vector while preserving zero vectors unchanged."""
    n = np.linalg.norm(vec)
    return vec if n == 0 else vec / n

# L2 (Redis): bytes float32 com prefixo mágico (3 KB por vetor de 768 dimensões, contra
# ~15 KB em JSON). Entradas legadas em JSON ({"v": [...]}) continuam legíveis até expirar.
_L2_BINARY_MAGIC = b"f32:"


def _encode_cached_vector(vec: List[float]) -> bytes:
    """Serialize one embedding for the L2 cache."""
    return _L2_BINARY_MAGIC + np.asarray(vec, dtype=np.float32).tobytes()


def _decode_cached_vector(raw: Any) -> Optional[List[float]]:
    """Deserialize an L2 entry (binary float32 or legacy JSON)."""
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    if raw.startswith(_L2_BINARY_MAGIC):
        payload = raw[len(_L2_BINARY_MAGIC):]
        if len(payload) % 4:
            return None
        return np.frombuffer(payload, dtype=np.float32).tolist()
    return json.loads(raw).get("v")


def _save_cache(key: str, vec: List[float]):
    """Persist one embedding vector to the Redis-backed L2 cache."""
    if _rds:
        try:
            _rds.setex(key, EMBED_CACHE_TTL_S, _encode_cached_vector(vec))
        except Exception as e:
            logger.warning(f"[Embeddings] Failed to save cache for key {key[:20]}...: {e}")

def _load_cache(key: str) -> Optional[List[float]]:
    """Load one embedding vector from the Redis-backed L2 cache."""
    if _rds:
        try:
            raw = _rds.get(key)
            if raw:
                return _decode_cached_vector(raw)
        except Exception as e:
            logger.warning(f"[Embeddings] Failed to load cache for key {key[:20]}...: {e}")
    return None

# ============================================================
# GERAÇÃO (LOCAL CPU)
# ============================================================

def _local_cpu_embed(text: str) -> List[float]:
    """Generate one embedding locally on CPU without involving Ollama."""
    model = get_local_model()
    if not model:
        raise RuntimeError("SentenceTransformers não instalado ou falha ao carregar.")

    # Gera vetor (o prefixo de tarefa do Nomic já vem aplicado por embed_text)
    vec = model.encode(text, convert_to_numpy=True, show_progress_bar=False)  # sem barra "Batches" nos logs
    return vec.tolist()

# ============================================================
# API PÚBLICA
# ============================================================

def embed_text(text: str, tarefa: str = "consulta") -> List[float]:
    """Generate a text embedding with layered cache lookup and CPU-first fallback.

    ``tarefa`` is ``"consulta"`` (queries, the default) or ``"documento"`` (corpus chunks at ingest).
    """
    text = (text or "").strip()
    if not text:
        return [0.0]
    text = com_prefixo(text, tarefa, EMBED_MODEL_TEXT)  # documento x consulta (Nomic)

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
        # Forma canônica float32: igual com ou sem acerto de cache (o modelo local já produz float32).
        vec = np.asarray(vec, dtype=np.float32).tolist()
        _save_cache(key, vec)           # L2 Cache
        _embed_l1_cache.set(key, vec)   # L1 Cache
        return vec

    _warn_null_embedding()
    return [0.0] * 768  # Retorna vetor zerado em caso de falha total


_last_null_embedding_log = 0.0


def _warn_null_embedding() -> None:
    """Log (at most once a minute) that no provider produced an embedding.

    The zero vector keeps the request alive, but uncertainty becomes 1.0 and the
    semantic cache stops matching; before this log the degradation was silent.
    """
    global _last_null_embedding_log
    now = time.monotonic()
    if now - _last_null_embedding_log >= 60.0:
        _last_null_embedding_log = now
        logger.error("[Embeddings] Nenhum provedor de embeddings disponível: usando vetor nulo (u(q) = 1, cache semântico inoperante).")


def get_embedding_cache_stats() -> Dict[str, Any]:
    """Expose L1 embedding-cache statistics for observability and diagnostics."""
    return _embed_l1_cache.stats()

# Vision embedding removido/simplificado pois o RAG agora usa "Ponte Descritiva"
# (VLM gera texto -> Texto vira embedding)
def embed_image(image_b64: str) -> List[float]:
    """Return a placeholder image embedding for the current text-bridge strategy.

    The project currently relies on image description followed by text
    embeddings, so direct image embeddings remain intentionally stubbed while
    preserving a stable public interface.
    """
    return [0.0]

def embed_multimodal(text: str, image_b64: Optional[str]) -> Dict[str, List[float]]:
    """Return the multimodal embedding payload expected by downstream callers.

    Under the current bridge strategy, multimodal requests are represented by
    the text embedding derived from the textual side of the prompt or image
    description pipeline.
    """
    text_vec = embed_text(text)
    # Retorna apenas o texto, pois estamos usando a estratégia de descrição
    return {
        "text": text_vec,
        "vision": [],
        "multimodal": text_vec
    }


# ============================================================
# ASYNC WRAPPERS (dedicated CPU pool — perf #23)
# ============================================================
# Route the GIL-bound encode() onto a CPU-isolated thread pool instead of the
# shared asyncio default executor, so embedding bursts do not starve I/O work
# (DB reads, Redis) that also rides asyncio.to_thread. Callers on the async
# hot path (semantic cache, RAG retrieval) should prefer these over
# asyncio.to_thread(embed_*, ...).

async def aembed_text(text: str) -> List[float]:
    """Async text embedding executed on the dedicated CPU pool."""
    from .utils.executors import run_cpu_bound

    return await run_cpu_bound(embed_text, text)


async def aembed_image(image_b64: str) -> List[float]:
    """Async image embedding executed on the dedicated CPU pool."""
    from .utils.executors import run_cpu_bound

    return await run_cpu_bound(embed_image, image_b64)


async def aembed_multimodal(text: str, image_b64: Optional[str]) -> Dict[str, List[float]]:
    """Async multimodal embedding executed on the dedicated CPU pool."""
    from .utils.executors import run_cpu_bound

    return await run_cpu_bound(embed_multimodal, text, image_b64)
