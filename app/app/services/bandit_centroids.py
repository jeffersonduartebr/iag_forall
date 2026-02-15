# -*- coding: utf-8 -*-
"""Centroid helpers for bandit clustering."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def normalize_centroid_vec(vec: np.ndarray, dim: int) -> np.ndarray:
    """Resumo do comportamento desta função.

    Args:
        vec: Parâmetro de entrada.
        dim: Parâmetro de entrada.

    Returns:
        Valor retornado pela função.
    """
    v = vec.astype(np.float32).reshape(-1)
    if len(v) != dim:
        if len(v) > dim:
            v = v[:dim]
        else:
            v = np.concatenate([v, np.zeros(dim - len(v), dtype=np.float32)])
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


def nearest_centroid_from_array(v: np.ndarray, cents: List[dict]) -> Tuple[Optional[int], float]:
    """Resumo do comportamento desta função.

    Args:
        v: Parâmetro de entrada.
        cents: Parâmetro de entrada.

    Returns:
        Valor retornado pela função.
    """
    if not cents:
        return None, 0.0
    c = np.stack([c_["vec"] for c_ in cents], axis=0)
    sims = c @ v
    idx = int(np.argmax(sims))
    return idx, float(sims[idx])
