# -*- coding: utf-8 -*-
# Objective: Shadow comparison between an explored OpenRouter model and the incumbent.
"""Replay a sample of explored queries on the incumbent model and compare quality.

Extracted from ``app.openrouter_explorer`` (re-exported there).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import numpy as np

from app.openrouter_exploration_state import _load_model_stats, _save_model_stats

logger = logging.getLogger("app.openrouter_explorer")


async def maybe_run_shadow_comparison(
    *,
    query: str,
    explored_model: str,
    explored_answer: str,
    explored_quality: float,
    explored_latency: float,
    explored_cost: float,
    incumbent_model: Optional[str],
    deps: Dict[str, Any],
    settings: Any,
) -> None:
    """Run incumbent model on the same query and store paired comparison deltas."""
    if not incumbent_model or incumbent_model == explored_model:
        return
    if not incumbent_model.startswith(("openrouter/", "openai/", "anthropic/", "gemini/", "ollama/")):
        return

    try:
        out, meta = await deps["call_model"](
            model=incumbent_model,
            prompt=query,
            modality="text",
            image_b64=None,
            temperature=float(getattr(settings, "TEMPERATURE_DEFAULT", 0.2) or 0.2),
            max_tokens=int(getattr(settings, "MAX_TOKENS_DEFAULT", 512) or 512),
        )
        incumbent_answer = out if isinstance(out, str) else str(out)
        p_tok, c_tok, incumbent_cost, _, _ = deps["parse_meta_cost"](
            meta=meta,
            chosen_model=incumbent_model,
            cost_lookup=deps["get_model_cost"],
        )
        judge_scores = await deps["judge_answer"](query, incumbent_answer)
        valid_scores = [s["score"] for s in judge_scores if "score" in s]
        incumbent_quality = round((float(np.mean(valid_scores)) if valid_scores else 5.0) * 10.0, 2)

        delta = {
            "delta_quality": round(explored_quality - incumbent_quality, 3),
            "delta_latency_s": round(explored_latency - 0.0, 3),
            "delta_cost_usd": round(explored_cost - float(incumbent_cost or 0.0), 6),
            "incumbent_model": incumbent_model,
            "explored_model": explored_model,
            "recorded_at": time.time(),
        }

        from app.openrouter_explorer import _get_redis  # import tardio: o explorer reexporta este módulo

        rds = await _get_redis()
        if rds:
            stats = await _load_model_stats(rds, explored_model)
            shadows = stats.get("shadow_comparisons") or []
            if not isinstance(shadows, list):
                shadows = []
            shadows.append(delta)
            stats["shadow_comparisons"] = shadows[-20:]
            stats["shadow_delta_quality_mean"] = round(
                sum(float(s.get("delta_quality", 0)) for s in stats["shadow_comparisons"]) / len(stats["shadow_comparisons"]),
                4,
            )
            await _save_model_stats(rds, explored_model, stats)

        logger.info(
            "[openrouter_explore] shadow %s vs %s delta_q=%.2f delta_cost=%.6f",
            explored_model,
            incumbent_model,
            delta["delta_quality"],
            delta["delta_cost_usd"],
        )
    except Exception as exc:
        logger.warning("[openrouter_explore] shadow comparison failed: %s", exc)
