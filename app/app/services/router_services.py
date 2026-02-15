# -*- coding: utf-8 -*-
"""Shared helpers for router_core routing/feedback/maintenance."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple


def normalize_modality(modality: str, image_b64: Optional[str]) -> str:
    """Executa normalize modality."""
    m = modality or "text"
    if bool(image_b64) and m == "text":
        return "vision"
    return m


def build_final_prompt(
    query: str,
    system_prompt: str,
    use_rag: bool,
    rag_text: Optional[str],
) -> str:
    """Executa build final prompt."""
    if use_rag:
        if rag_text is None:
            rag_text = query
        if system_prompt:
            return f"{system_prompt}\n\nUsuário: {rag_text}".strip()
        return rag_text

    if system_prompt:
        return f"{system_prompt}\n\nUsuário: {query}".strip()
    return query


def parse_meta_cost(
    meta: Any,
    chosen_model: str,
    cost_lookup,
) -> Tuple[int, int, float, float, Dict[str, Any]]:
    """Executa parse meta cost."""
    p_tok = 0
    c_tok = 0
    total_cost = 0.0
    load_time_s = 0.0

    if isinstance(meta, dict):
        load_time_s = float(meta.get("load_time", 0.0) or 0.0)
        p_tok = int(meta.get("prompt_tokens", 0) or 0)
        c_tok = int(meta.get("completion_tokens", 0) or 0)

        if meta.get("cost_per_1k") is not None and float(meta.get("cost_per_1k", 0)) > 0:
            total_cost = float(meta["cost_per_1k"])
        else:
            try:
                total_cost = float(cost_lookup(chosen_model, p_tok, c_tok))
            except Exception:
                total_cost = 0.0

    meta_safe = meta if isinstance(meta, dict) else {}
    return p_tok, c_tok, total_cost, load_time_s, meta_safe


def should_enable_dedup(settings_get, requested: bool) -> bool:
    """Executa should enable dedup."""
    return requested and str(settings_get("REQUEST_DEDUP_ENABLED", "0")).strip() in ("1", "true", "True")


def compute_judge_probability(
    n_samples: int,
    predicted_error_prob: float,
    chosen_model: str,
    min_sample_rate: float,
) -> float:
    """Executa compute judge probability."""
    if n_samples < 5:
        base_prob = 1.0
    else:
        base_prob = 1.0 / math.sqrt(n_samples)

    prob_judge = max(base_prob, predicted_error_prob)

    if "gpt-5" in chosen_model or "gpt-4" in chosen_model or "claude" in chosen_model:
        prob_judge *= 0.1

    return max(min_sample_rate, prob_judge)
