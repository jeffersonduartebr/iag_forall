# Objective: Service-layer helpers for router feedback.
"""Background feedback processing helper for router_core.

Orchestrates the stages in ``services.feedback_stages``: error-risk estimate,
judge sampling, quality, reward/bandit update, exploration bookkeeping, EMA,
semantic cache and query log.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from ..correlation import _correlation_id
from .feedback_stages import (
    ErrorRisk,
    FeedbackRequest,
    Quality,
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
    # Valores que a linha leva se um estágio falhar: a linha é escrita sempre, com o que se apurou até ali.
    risk = ErrorRisk(model_stats={}, predictor=None, query_embedding=None, predicted_error_prob=None)
    quality: Optional[Quality] = None
    judged, reward = False, None
    token = _correlation_id.set(fb.payload.get("correlation_id"))  # as notas dos juízes levam o id da requisição
    try:
        try:
            risk = await assess_error_risk(deps, fb)
            decision = decide_judging(deps, fb, risk)
            judged = decision.should_judge
            quality = await judge_quality(deps, fb, risk, decision) if judged else proxy_quality(risk)
            reward = update_bandit(deps, fb, quality)
            await record_exploration(deps, fb, reward, quality)
            update_ema(deps, state, fb, quality)
            await maybe_store_cache(deps, fb, quality)
        except Exception as exc:
            # Um erro de estágio não é repetido (repetir volta a pagar os juízes), mas também não apaga a linha.
            quietly(lambda: deps["FEEDBACK_TASK_FAILURES"].labels(stage="pipeline").inc())
            deps["logger"].exception(f"[Background] Critical fail: {exc}")
        # Falha de escrita propaga (FeedbackPersistError) e a tarefa fica em FAILURE.
        persist_log(deps, fb, quality, judged, risk, reward)
    finally:
        _correlation_id.reset(token)
        deps["FEEDBACK_PROCESSING_LATENCY"].observe(time.time() - started)
