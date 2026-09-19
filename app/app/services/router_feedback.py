# Objective: Service-layer helpers for router feedback.
"""Background feedback processing helper for router_core.

Orchestrates the stages in ``services.feedback_stages``: error-risk estimate,
judge sampling, quality, reward/bandit update, exploration bookkeeping, EMA,
semantic cache and query log.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from .feedback_stages import (
    FeedbackRequest,
    assess_error_risk,
    decide_judging,
    judge_quality,
    maybe_store_cache,
    observe_backlog_age,
    persist_log,
    proxy_quality,
    record_exploration,
    update_bandit,
    update_ema,
)
from .router_stages import quietly


async def process_background_feedback_impl(
    *,
    deps: Dict[str, Any],
    state: Dict[str, Any],
    query: str,
    answer: str,
    chosen_model: str,
    modality: str,
    latency_s: float,
    cost_val: float,
    image_b64: Optional[str] = None,
    raw_payload: Optional[Any] = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> None:
    """Process feedback using injected router_core dependencies and state."""
    started = time.time()
    fb = FeedbackRequest(
        query=query,
        answer=answer,
        chosen_model=chosen_model,
        modality=modality,
        latency_s=latency_s,
        cost_val=cost_val,
        image_b64=image_b64,
        raw_payload=raw_payload,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    observe_backlog_age(deps, fb, started)
    try:
        risk = await assess_error_risk(deps, fb)
        decision = decide_judging(deps, fb, risk)
        quality = await judge_quality(deps, fb, risk, decision) if decision.should_judge else proxy_quality(risk)
        reward = update_bandit(deps, fb, quality)
        await record_exploration(deps, fb, reward, quality)
        update_ema(deps, state, fb, quality)
        await maybe_store_cache(deps, fb, quality)
        persist_log(deps, fb, quality, decision.should_judge, risk, reward)
    except Exception as exc:
        quietly(lambda: deps["FEEDBACK_TASK_FAILURES"].labels(stage="persist").inc())
        deps["logger"].exception(f"[Background] Critical fail: {exc}")
    finally:
        deps["FEEDBACK_PROCESSING_LATENCY"].observe(time.time() - started)
