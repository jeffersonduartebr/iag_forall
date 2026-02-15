# -*- coding: utf-8 -*-
"""
semantic_cache.py — Cache Semântico via ChromaDB (Rápido)
---------------------------------------------------------
Substitui a busca linear no MariaDB por busca ANN no ChromaDB.

L1 Cache: Thread-safe in-memory TTL cache for exact-match queries
L2 Cache: ChromaDB semantic search for similar queries

Phase 5: Autonomous Behavior
- Adaptive cache threshold based on hit rate
- P-controller for automatic threshold tuning
"""

from __future__ import annotations
import hashlib
import logging
import asyncio
import time
import threading
from collections import OrderedDict
from typing import Optional, Dict, Any

from .settings_dynamic import settings
from .embeddings import embed_text, embed_image, embed_multimodal
from .vectorstore import query_embedding, add_document

logger = logging.getLogger(__name__)


def get_cache_threshold() -> float:
    """
    Get current cache threshold from settings (dynamic reload).

    Returns:
        float: Similarity threshold for cache hits (0.0 to 1.0)
    """
    return float(settings.get("CACHE_THRESHOLD", 0.92))


def _compute_sha256(text: str) -> str:
    """Executa compute sha256."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_modality(mod: str) -> str:
    """Executa normalize modality."""
    mod = mod.lower().strip()
    if mod in ("vision", "image"):
        return "vision"
    if mod == "multimodal":
        return "multimodal"
    return "text"


# ==============================================================================
# L1 Cache: Thread-safe TTL Cache (Replaces broken lru_cache implementation)
# ==============================================================================
class L1Cache:
    """
    Thread-safe in-memory cache with TTL and LRU eviction.

    This properly implements what the broken @lru_cache approach was trying to do:
    - Store key-value pairs in memory for fast exact-match lookups
    - Limit memory usage with maxsize
    - Auto-expire stale entries with TTL
    """

    def __init__(self, maxsize: int = 1024, ttl_seconds: int = 300):
        """Inicializa estado interno necessário para uso da classe."""
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = threading.Lock()
        self._maxsize = maxsize
        self._ttl = ttl_seconds
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value from cache.

        Returns None if key not found or expired.
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            value, timestamp = self._cache[key]

            # Check TTL
            if time.time() - timestamp > self._ttl:
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return value

    def store(self, key: str, value: Any) -> None:
        """
        Store a value in cache.

        Evicts oldest entries if maxsize is exceeded.
        """
        with self._lock:
            # Update existing or add new
            self._cache[key] = (value, time.time())
            self._cache.move_to_end(key)

            # Evict oldest if over capacity
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()

    def stats(self) -> Dict[str, int]:
        """Return cache statistics."""
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._cache),
                "maxsize": self._maxsize,
            }


# Global L1 cache instance
# Optimized for high-capacity environment (64GB RAM) - ~100MB memory
# TTL of 10 minutes for L1 (exact match), since L2 handles semantic similarity
_l1_cache = L1Cache(maxsize=20000, ttl_seconds=600)

async def _make_embedding(query: str, modality: str, image_b64: Optional[str]):
    # Reutiliza lógica de embeddings
    """Executa make embedding."""
    try:
        if modality == "text":
            return await asyncio.to_thread(embed_text, query)
        if modality == "vision":
            if not image_b64: return await asyncio.to_thread(embed_text, query)
            return await asyncio.to_thread(embed_image, image_b64)
        emb = await asyncio.to_thread(embed_multimodal, query, image_b64)
        return emb.get("multimodal") or emb.get("text")
    except Exception as e:
        logger.warning(f"[semantic_cache] Embed fail: {e}")
        return None

async def check_cache(query: str, modality: str="text", image_b64: str=None) -> Optional[Dict]:
    """
    Verifica se existe uma resposta similar no ChromaDB.
    """
    modality = _normalize_modality(modality)
    
    # 1. L1 Cache (Exact Match em RAM)
    # Hash da query + imagem para chave rápida
    img_hash = hashlib.sha256(image_b64.encode()).hexdigest() if image_b64 else "no_img"
    full_hash = f"{modality}:{_compute_sha256(query)}:{img_hash}"

    if cached := _l1_cache.get(full_hash):
        logger.info("[semantic_cache] L1 RAM Hit")
        return cached

    # 2. Gera Embedding
    q_emb = await _make_embedding(query, modality, image_b64)
    if q_emb is None: return None

    # 3. L2 Cache (Semantic Search no Chroma)
    try:
        # Busca na coleção de cache ("cache" é mapeado para semantic_cache_v2 no vectorstore.py)
        results = await query_embedding(
            modality="cache", 
            embedding=q_emb, 
            n_results=1
        )
        
        if not results or not results.get("documents"):
            return None

        # Chroma retorna 'distances' (Cosine Distance).
        # Similarity = 1 - Distance.
        distance = results["distances"][0][0]
        similarity = 1.0 - distance

        # Use dynamic threshold (Phase 5)
        threshold = get_cache_threshold()
        if similarity >= threshold:
            # O texto do documento é a Query original
            # A resposta está no metadata
            meta = results["metadatas"][0][0]
            answer_text = meta.get("answer_payload", "")
            
            if not answer_text:
                return None

            # Reconstrói objeto de resposta
            res = {
                "text": answer_text, 
                "similarity": float(similarity),
                "model_used": meta.get("model_used", "unknown"),
                "image_output_b64": meta.get("image_output_b64", None) # Se houver
            }
            
            # Atualiza L1
            _l1_cache.store(full_hash, res)
            logger.debug(f"[semantic_cache] L1 cache stored for hash {full_hash[:16]}...")
            return res
            
    except Exception as e:
        logger.warning(f"[semantic_cache] Chroma lookup fail: {e}")
        return None

    return None

async def store_cache(query: str, answer: str, modality: str="text", image_b64: str=None, model_used: str=None):
    """
    Armazena uma resposta de alta qualidade no ChromaDB e L1 cache.
    """
    modality = _normalize_modality(modality)

    # ID único para o documento de cache
    doc_id = f"cache_{_compute_sha256(query)}_{int(time.time())}"

    # Metadados para recuperação
    meta = {
        "model_used": model_used or "unknown",
        "original_query": query[:100],  # Snippet para debug
        "timestamp": int(time.time()),
        "modality": modality,
        "answer_payload": answer  # Armazena a resposta no metadata
    }

    # Store in L1 cache immediately for fast exact-match retrieval
    img_hash = hashlib.sha256(image_b64.encode()).hexdigest() if image_b64 else "no_img"
    full_hash = f"{modality}:{_compute_sha256(query)}:{img_hash}"
    l1_entry = {
        "text": answer,
        "similarity": 1.0,  # Exact match
        "model_used": model_used or "unknown",
        "image_output_b64": None
    }
    _l1_cache.store(full_hash, l1_entry)

    try:
        # Estratégia:
        # Documento (Text) = PERGUNTA (para match semântico com a nova pergunta)
        # Metadata = RESPOSTA (o que queremos devolver)

        await add_document(
            modality="cache",  # Vai para semantic_cache_v2
            doc_id=doc_id,
            text=query,  # O texto indexado é a pergunta
            metadata=meta
        )

        # Phase 5: Update judge calibration when response is cached
        try:
            from .judges import update_calibration_cache_status
            update_calibration_cache_status(query)
        except Exception:
            pass

    except Exception as e:
        logger.warning(f"[semantic_cache] Store fail: {e}")


def get_l1_cache_stats() -> Dict[str, int]:
    """
    Get L1 cache statistics for monitoring.

    Returns:
        Dict with hits, misses, size, maxsize
    """
    return _l1_cache.stats()


def get_cache_hit_rate() -> float:
    """
    Calculate current cache hit rate.

    Returns:
        float: Hit rate (0.0 to 1.0), or -1.0 if insufficient data
    """
    stats = _l1_cache.stats()
    total = stats["hits"] + stats["misses"]
    if total < 10:
        return -1.0  # Not enough data
    return stats["hits"] / total


async def tune_cache_threshold() -> Optional[float]:
    """
    Automatically tune cache threshold based on hit rate.

    Uses a P-controller to adjust threshold toward target hit rate:
    - If hit rate is too high -> increase threshold (more strict)
    - If hit rate is too low -> decrease threshold (more permissive)

    Returns:
        New threshold value if adjusted, None otherwise
    """
    if not settings.CACHE_THRESHOLD_ADAPT_ENABLED:
        return None

    stats = _l1_cache.stats()
    total = stats["hits"] + stats["misses"]

    # Need minimum samples to make meaningful adjustments
    if total < 100:
        logger.debug("[CacheTuning] Not enough samples for tuning")
        return None

    hit_rate = stats["hits"] / total
    current = get_cache_threshold()
    target = settings.CACHE_HIT_RATE_TARGET
    min_thresh = settings.CACHE_THRESHOLD_MIN
    max_thresh = settings.CACHE_THRESHOLD_MAX

    # P-controller: adjust toward target hit rate
    # Higher hit rate -> increase threshold (be more strict)
    # Lower hit rate -> decrease threshold (be more permissive)
    error = hit_rate - target
    adjustment = error * 0.01  # Small adjustment factor

    new_threshold = current + adjustment
    new_threshold = max(min_thresh, min(max_thresh, new_threshold))

    # Only update if change is significant
    if abs(new_threshold - current) > 0.005:
        try:
            settings.set(
                "CACHE_THRESHOLD",
                str(round(new_threshold, 3)),
                actor="cache-tuner",
                source="adaptive",
            )
            logger.info(
                f"[CacheTuning] Threshold adjusted: {current:.3f} -> {new_threshold:.3f} "
                f"(hit_rate={hit_rate:.2%}, target={target:.2%})"
            )

            # Update Prometheus metrics
            try:
                from .observability import (
                    CACHE_THRESHOLD_CURRENT,
                    CACHE_HIT_RATE,
                    CACHE_THRESHOLD_ADJUSTMENTS,
                )
                CACHE_THRESHOLD_CURRENT.set(new_threshold)
                CACHE_HIT_RATE.set(hit_rate)
                CACHE_THRESHOLD_ADJUSTMENTS.inc()
            except Exception:
                pass

            return new_threshold

        except Exception as e:
            logger.warning(f"[CacheTuning] Failed to update threshold: {e}")

    return None


def reset_cache_stats() -> None:
    """
    Reset L1 cache statistics.

    Useful after threshold adjustment to get fresh data.
    """
    with _l1_cache._lock:
        _l1_cache._hits = 0
        _l1_cache._misses = 0
