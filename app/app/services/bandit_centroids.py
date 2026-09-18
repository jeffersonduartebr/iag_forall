# -*- coding: utf-8 -*-
# Objective: Service-layer helpers for bandit centroids.
"""Centroid helpers for bandit clustering and epistemic uncertainty.

The bandit stores its semantic centroids as one JSON document in Redis. Both
the routing path (nearest-centroid context label) and the uncertainty score
read it on every request, so :func:`load_centroid_matrix` keeps the parsed,
unit-normalized float32 matrix in memory and reloads it only when the
centroid revision published in the metadata hash changes.
"""

from __future__ import annotations

import json
import threading
from typing import Any, List, NamedTuple, Optional, Tuple

import numpy as np


def normalize_centroid_vec(vec: np.ndarray, dim: int) -> np.ndarray:
    """Execute the normalize centroid vec routine.

This helper encapsulates one focused step used by the surrounding workflow."""
    v = vec.astype(np.float32).reshape(-1)
    if len(v) != dim:
        if len(v) > dim:
            v = v[:dim]
        else:
            v = np.concatenate([v, np.zeros(dim - len(v), dtype=np.float32)])
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


def nearest_centroid_from_array(v: np.ndarray, cents: List[dict]) -> Tuple[Optional[int], float]:
    """Execute the nearest centroid from array routine.

This helper encapsulates one focused step used by the surrounding workflow."""
    if not cents:
        return None, 0.0
    c = np.stack([c_["vec"] for c_ in cents], axis=0)
    sims = c @ v
    idx = int(np.argmax(sims))
    return idx, float(sims[idx])


class CentroidMatrix(NamedTuple):
    """Unit-normalized centroid matrix ``(K, D)`` and the centroid ids per row."""

    matrix: np.ndarray
    ids: List[int]


_snapshot_lock = threading.Lock()
_snapshot: Optional[Tuple[str, CentroidMatrix]] = None  # ("key@version", matrix)


def parse_centroid_matrix(raw: Any, dim: Optional[int] = None) -> Optional[CentroidMatrix]:
    """Parse the centroid JSON document into a normalized matrix (``None`` if empty).

    ``dim`` pads/truncates every vector; by default the first vector's size is used.
    """
    items = json.loads(raw)
    vecs: List[np.ndarray] = []
    ids: List[int] = []
    for pos, item in enumerate(items or []):
        if not isinstance(item, dict) or not item.get("vec"):
            continue
        vec = np.asarray(item["vec"], dtype=np.float32).reshape(-1)
        if dim is None:
            dim = int(vec.shape[0])
        vecs.append(normalize_centroid_vec(vec, dim))
        ids.append(int(item.get("id", pos)))
    if not vecs:
        return None
    return CentroidMatrix(np.stack(vecs, axis=0), ids)


def _revision(rds: Any, meta_key: str) -> Optional[str]:
    """Centroid revision from the metadata hash, or ``None`` when unavailable."""
    try:
        rev, updated_at = rds.hmget(meta_key, ["rev", "updated_at"])
    except Exception:
        return None
    parts = [p.decode() if isinstance(p, bytes) else p for p in (rev, updated_at)]
    if not all(isinstance(p, (str, int)) for p in parts if p is not None) or all(p is None for p in parts):
        return None
    return f"{parts[0]}:{parts[1]}"


def load_centroid_matrix(rds: Any, key: str, meta_key: str, dim: Optional[int] = None) -> Optional[CentroidMatrix]:
    """Return the centroid matrix, reusing the in-memory copy while the revision is unchanged.

    Without a readable revision (metadata missing, test doubles) the document is
    read and parsed on every call, exactly like the previous behavior.
    """
    global _snapshot
    revision = _revision(rds, meta_key)
    cache_key = f"{key}@{revision}:{dim}" if revision is not None else None
    if cache_key is not None:
        with _snapshot_lock:
            if _snapshot is not None and _snapshot[0] == cache_key:
                return _snapshot[1]
    raw = rds.get(key)
    if not raw:
        return None
    parsed = parse_centroid_matrix(raw, dim)
    if parsed is not None and cache_key is not None:
        with _snapshot_lock:
            _snapshot = (cache_key, parsed)
    return parsed


def reset_centroid_matrix_cache() -> None:
    """Drop the in-memory centroid matrix (tests, centroid resets)."""
    global _snapshot
    with _snapshot_lock:
        _snapshot = None
