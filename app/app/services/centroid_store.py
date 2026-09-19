# -*- coding: utf-8 -*-
# Objective: Semantic centroid state for the contextual bandit (Redis-backed, learned online).
"""Semantic centroids: storage, online learning and nearest-centroid labels.

Queries are grouped into up to ``CENTROIDS_K`` unit centroids (768-d) kept as
one JSON document in Redis (``meta:bandit:centroids``) plus a metadata hash
with a monotonic revision. ``centroids_online_update`` moves the nearest
centroid towards the query (rate ``CENTROIDS_LEARN_RATE``) or creates one when
nothing is similar enough; ``_nearest_centroid_label`` is the read-only lookup
used for routing context. Extracted from ``app.bandits`` (re-exported there).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, List, Optional, Tuple

import numpy as np

from app.embeddings import embed_text
from app.services.bandit_centroids import (
    load_centroid_matrix,
    nearest_centroid_from_array,
    normalize_centroid_vec,
)
from app.settings_dynamic import settings
from app.utils.redis_client import ensure_redis_connected, get_redis_async_safe

logger = logging.getLogger("app.bandits")

R_CENTROIDS = "meta:bandit:centroids"  # lista completa de centróides (UQ)
R_CENTROIDS_META = "meta:bandit:centroids:meta"
R_CENTROIDS_LOCK = "meta:bandit:centroids:lock"


def _setting_number(key: str, default: float, cast=float):
    try:
        return cast(settings.get(key, default))
    except Exception:
        return cast(default)


CENTROIDS_K = _setting_number("CENTROIDS_K", 20, int)
CENTROIDS_DIM = _setting_number("CENTROIDS_DIM", 768, int)
CENTROIDS_LEARN_RATE = _setting_number("CENTROIDS_LEARN_RATE", 0.15)
CENTROIDS_MIN_SIM_CREATE = _setting_number("CENTROIDS_MIN_SIM_CREATE", 0.35)


def _get_rds():
    """Redis client for centroid storage (never blocks the hot path)."""
    return get_redis_async_safe() or ensure_redis_connected(max_wait_s=0.0, min_retry_interval_s=2.0)


# ============================================================
# Utils NumPy
# ============================================================
def _unit(v: np.ndarray) -> np.ndarray:
    """Return a normalized vector, preserving zero vectors unchanged."""
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors safely."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _ensure_dim(v: np.ndarray) -> np.ndarray:
    """Project a vector to the configured centroid dimensionality."""
    return normalize_centroid_vec(v, CENTROIDS_DIM)


# ============================================================
# Pre-computed Centroid Matrix Cache (Performance Optimization)
# ============================================================
class CentroidMatrixCache:
    """Cache a dense centroid matrix for fast nearest-neighbor lookup.

    Centroid search is a hot path during routing. Precomputing the stacked
    matrix avoids repeated array construction and enables a single vectorized dot
    product for nearest-centroid selection.
    """

    def __init__(self):
        """Initialize the centroid cache with no precomputed matrix."""
        self._lock = threading.Lock()
        self._matrix: Optional[np.ndarray] = None  # (K, D)
        self._ids: List[int] = []
        self._version: int = 0
        self._last_update: float = 0.0

    def update(self, cents: List[dict]) -> None:
        """Atualiza a matriz cache com os centróides atuais."""
        if not cents:
            with self._lock:
                self._matrix = None
                self._ids = []
                self._version += 1
            return

        with self._lock:
            self._matrix = np.stack([c["vec"] for c in cents], axis=0).astype(np.float32)
            self._ids = [c["id"] for c in cents]
            self._version += 1
            self._last_update = time.time()

    def nearest(self, v: np.ndarray) -> Tuple[Optional[int], float, int]:
        """
        Busca o centróide mais próximo usando a matriz pré-computada.
        Retorna: (índice no array original, similaridade, centroid_id)
        """
        with self._lock:
            if self._matrix is None or len(self._ids) == 0:
                return None, 0.0, -1

            # Produto escalar vetorizado (unit vectors → cosine)
            sims = self._matrix @ v
            idx = int(np.argmax(sims))
            return idx, float(sims[idx]), self._ids[idx]

    def is_stale(self, max_age_s: float = 60.0) -> bool:
        """Verifica se o cache está desatualizado."""
        return time.time() - self._last_update > max_age_s


_centroid_matrix_cache = CentroidMatrixCache()

# ============================================================
# Centróides semânticos (NumPy vetorizado) + clustering dinâmico
# ============================================================

def _acquire_lock(key: str, ttl: int = 10) -> bool:
    """Acquire a short-lived Redis lock for centroid mutation."""
    rds = _get_rds()
    if not rds:
        return False
    try:
        return bool(rds.set(key, "1", nx=True, ex=ttl))
    except Exception:
        return False


def _release_lock(key: str) -> None:
    """Release a Redis lock previously acquired for centroid mutation."""
    rds = _get_rds()
    if not rds:
        return
    try:
        rds.delete(key)
    except Exception:
        pass


def _parse_centroid(item: Any, now_ts: int) -> Optional[dict]:
    """One stored centroid as ``{"id", "vec" (unit, CENTROIDS_DIM), "count", "last"}``; ``None`` if malformed."""
    if not isinstance(item, dict) or "id" not in item or "vec" not in item:
        return None
    return {
        "id": int(item["id"]),
        "vec": _ensure_dim(np.array(item["vec"], dtype=np.float32)),
        "count": int(item.get("count", 0)),
        "last": int(item.get("last", now_ts)),
    }


def _load_centroids(update_matrix_cache: bool = True) -> List[dict]:
    """
    Carrega centróides de Redis:
    [
      {"id": int, "vec": np.ndarray(D,), "count": int, "last": int}
    ]
    """
    rds = _get_rds()
    if not rds:
        return []
    try:
        raw = rds.get(R_CENTROIDS)
        if not raw:
            return []
        now_ts = int(time.time())
        cents = [c for c in (_parse_centroid(it, now_ts) for it in json.loads(raw)) if c is not None]
        if update_matrix_cache and cents:
            _centroid_matrix_cache.update(cents)
        return cents
    except Exception as e:
        logger.warning(f"[centroids] Falha ao carregar: {e}")
        return []


def _save_centroids(cents: List[dict]) -> None:
    """
    Persiste centróides com reinicialização automática de degenerados.
    """
    rds = _get_rds()
    if not rds:
        return
    serial = []
    normalized_cents = []
    now_ts = int(time.time())

    for it in cents:
        vec = np.array(it["vec"], dtype=np.float32).reshape(-1)
        # reinicialização de degenerados
        if not np.isfinite(vec).all() or np.linalg.norm(vec) < 1e-4:
            vec = np.random.normal(size=(CENTROIDS_DIM,)).astype(np.float32)
            vec = _unit(vec)
            cnt = 0
        else:
            vec = _unit(vec)
            cnt = int(it.get("count", 0))

        serial.append(
            {
                "id": int(it["id"]),
                "vec": vec.tolist(),
                "count": cnt,
                "last": int(it.get("last", now_ts)),
            }
        )
        normalized_cents.append({"id": int(it["id"]), "vec": vec, "count": cnt, "last": int(it.get("last", now_ts))})

    try:
        pipe = rds.pipeline()
        pipe.set(R_CENTROIDS, json.dumps(serial))
        pipe.hset(
            R_CENTROIDS_META,
            mapping={
                "updated_at": str(now_ts),
                "k": str(CENTROIDS_K),
                "dim": str(CENTROIDS_DIM),
                "count": str(len(serial)),
            },
        )
        # Revisão monotônica: leitores (incerteza, rótulo de contexto) só reprocessam o JSON quando muda.
        pipe.hincrby(R_CENTROIDS_META, "rev", 1)
        pipe.execute()

        # Update the pre-computed matrix cache
        _centroid_matrix_cache.update(normalized_cents)

    except Exception as e:
        logger.warning(f"[centroids] Falha ao salvar: {e}")


def _new_centroid_id(cents: List[dict]) -> int:
    """Return the first unused integer identifier for a new centroid."""
    used = {c["id"] for c in cents}
    cid = 0
    while cid in used:
        cid += 1
    return cid


def _nearest_centroid_vec(
    v: np.ndarray, cents: List[dict], use_cache: bool = True
) -> Tuple[Optional[int], float]:
    """
    Versão vetorizada: empilha centróides e faz produto escalar.
    Assumimos vetores unitários.

    Se use_cache=True e o cache da matriz está atualizado, usa o cache.
    """
    if not cents:
        return None, 0.0

    # Try to use pre-computed matrix cache for faster lookup
    if use_cache and not _centroid_matrix_cache.is_stale(max_age_s=120.0):
        idx, sim, _ = _centroid_matrix_cache.nearest(v)
        if idx is not None:
            return idx, sim

    # Fallback: compute on-the-fly
    return nearest_centroid_from_array(v, cents)


def centroids_online_update(query_text: str) -> Optional[int]:
    """
    Update semantic centroids online using the current query embedding.

    The procedure either creates a new centroid for sufficiently novel queries or
    nudges the nearest existing centroid toward the new embedding using an
    exponential moving update. Returning ``None`` means the update failed or was
    skipped because another process already holds the mutation lock.
    """
    try:
        v = embed_text(query_text)
        if not isinstance(v, np.ndarray):
            v = np.array(v, dtype=np.float32)
        v = _ensure_dim(v)
    except Exception as e:
        logger.debug(f"[centroids] Falha ao gerar embedding: {e}")
        return None

    if not _acquire_lock(R_CENTROIDS_LOCK, ttl=5):
        return None

    try:
        cents = _load_centroids()
        if not cents:
            cid = 0
            cents = [{"id": cid, "vec": v, "count": 1, "last": int(time.time())}]
            _save_centroids(cents)
            return cid

        idx, sim = _nearest_centroid_vec(v, cents)

        if (idx is None) or (sim < CENTROIDS_MIN_SIM_CREATE and len(cents) < CENTROIDS_K):
            cid = _new_centroid_id(cents)
            cents.append(
                {"id": cid, "vec": v, "count": 1, "last": int(time.time())}
            )
            _save_centroids(cents)
            return cid

        # Update exponencial no centróide existente
        c = cents[idx]
        new_vec = (1.0 - CENTROIDS_LEARN_RATE) * c["vec"] + CENTROIDS_LEARN_RATE * v
        new_vec = _unit(new_vec.astype(np.float32))
        c["vec"] = new_vec
        c["count"] = int(c.get("count", 0)) + 1
        c["last"] = int(time.time())
        _save_centroids(cents)
        return c["id"]
    finally:
        _release_lock(R_CENTROIDS_LOCK)


def _nearest_centroid_label(query_text: str) -> Optional[str]:
    """
    Return a read-only semantic context label for the nearest known centroid.

    Unlike ``centroids_online_update()``, this helper never mutates centroid
    state. It exists for logging, inspection, and other non-learning paths.
    """
    rds = _get_rds()
    if not rds:
        return None
    try:
        centroids = load_centroid_matrix(rds, R_CENTROIDS, R_CENTROIDS_META, CENTROIDS_DIM)
        if centroids is None:
            return None
        v = embed_text(query_text)
        if not isinstance(v, np.ndarray):
            v = np.array(v, dtype=np.float32)
        v = _ensure_dim(v)
        idx = int(np.argmax(centroids.matrix @ v))
    except Exception:
        return None
    return f"semctx:{centroids.ids[idx]}"
