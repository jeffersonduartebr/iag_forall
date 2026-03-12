# Objective: Service-layer helpers for router feedback.
"""Background feedback processing helper for router_core."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np


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
    feedback_start = time.time()
    raw_payload_dict = raw_payload if isinstance(raw_payload, dict) else {}
    enqueued_at = raw_payload_dict.get("queue_enqueued_at")
    if isinstance(enqueued_at, (int, float)):
        try:
            deps["FEEDBACK_BACKLOG_AGE"].set(max(0.0, time.time() - float(enqueued_at)))
        except Exception:
            pass
    try:
        stats = deps["_get_ctx_stats"]("global")
        model_stats = stats.get(chosen_model, {})
        n_samples = model_stats.get("count", 0)

        predictor = deps["get_predictor"](chosen_model)
        query_embedding = await deps["asyncio"].to_thread(deps["embed_text"], query)
        predicted_error_prob = predictor.predict_error_probability(query_embedding)

        prob_judge = deps["compute_judge_probability"](
            n_samples=n_samples,
            predicted_error_prob=predicted_error_prob,
            chosen_model=chosen_model,
            min_sample_rate=deps["settings"].JUDGE_MIN_SAMPLE_RATE,
        )

        should_judge = deps["random"].random() < prob_judge

        if should_judge:
            try:
                deps["logger"].info(
                    f"[Background] Sampling Judge for {chosen_model} (p={prob_judge:.2f}, pred_err={predicted_error_prob:.2f})"
                )
                judge_scores = await deps["judge_answer"](query, answer)
                valid_scores = [score["score"] for score in judge_scores if "score" in score]
                final_quality = round((float(np.mean(valid_scores)) if valid_scores else 5.0) * 10.0, 2)
                is_correct_label = final_quality >= 7.0
                predictor.learn(query_embedding, is_correct_label)
                predictor.record_outcome(predicted_error_prob, not is_correct_label)
                predictor.save()
            except Exception:
                final_quality = 5.0
                try:
                    deps["FEEDBACK_TASK_FAILURES"].labels(stage="judge").inc()
                except Exception:
                    pass
        else:
            final_quality = model_stats.get("mean", 0.5) * 10.0

        try:
            reward = deps["compute_reward"](chosen_model, final_quality, latency_s, cost_val)
        except Exception:
            reward = 0.0

        try:
            deps["bandit_update"](model=chosen_model, query=query, reward=reward, modality=modality)
        except Exception as exc:
            deps["logger"].warning(f"[Background] Bandit fail: {exc}")
            try:
                deps["FEEDBACK_TASK_FAILURES"].labels(stage="bandit_update").inc()
            except Exception:
                pass

        try:
            alpha = 0.2
            key = (modality, chosen_model)
            prev = state["EMA_HISTORY"].get(key)
            if prev is None:
                new_entry = {
                    "ema_latency": latency_s,
                    "ema_quality": final_quality,
                    "ema_cost": cost_val,
                    "ema_alignment": 1.0,
                    "updates": 1,
                }
            else:
                new_entry = {
                    "ema_latency": alpha * latency_s + (1 - alpha) * prev["ema_latency"],
                    "ema_quality": alpha * final_quality + (1 - alpha) * prev["ema_quality"],
                    "ema_cost": alpha * cost_val + (1 - alpha) * prev["ema_cost"],
                    "ema_alignment": prev.get("ema_alignment", 1.0),
                    "updates": prev.get("updates", 0) + 1,
                }
            state["EMA_HISTORY"].set(key, new_entry)
            deps["asyncio"].create_task(
                deps["asyncio"].to_thread(deps["_persist_ema"], modality, chosen_model, new_entry)
            )
        except Exception as exc:
            deps["logger"].warning(f"[Background] EMA update failed: {exc}")
            try:
                deps["FEEDBACK_TASK_FAILURES"].labels(stage="ema_update").inc()
            except Exception:
                pass

        if final_quality >= 7.0:
            try:
                await deps["store_cache"](
                    query=query,
                    answer=answer,
                    modality=modality,
                    image_b64=image_b64,
                    model_used=chosen_model,
                )
            except Exception as exc:
                deps["logger"].warning(f"[Background] Cache store failed: {exc}")
                try:
                    deps["FEEDBACK_TASK_FAILURES"].labels(stage="cache_write").inc()
                except Exception:
                    pass

        try:
            deps["ROUTER_QUALITY_AVG"].labels(model=chosen_model).set(final_quality)
            if "ollama" in chosen_model:
                deps["ROUTER_LOCAL_USAGE_RATIO"].set(1.0)
        except Exception:
            pass

        try:
            deps["insert_query_log"](
                query_text=query,
                model=chosen_model,
                modality=modality,
                image_provided=bool(image_b64),
                answer=answer,
                image_output_b64=None,
                latency_s=latency_s,
                cost_per_1k=cost_val,
                quality=final_quality,
                reward=reward,
                context_label="async_processed",
                raw_payload=raw_payload,
                query_embedding=query_embedding,
                answer_embedding=None,
            )
        except Exception as exc:
            deps["logger"].warning(f"[Background] Log fail: {exc}")
            try:
                deps["FEEDBACK_TASK_FAILURES"].labels(stage="persist").inc()
            except Exception:
                pass

        deps["FEEDBACK_PROCESSING_LATENCY"].observe(time.time() - feedback_start)
    except Exception as exc:
        deps["FEEDBACK_PROCESSING_LATENCY"].observe(time.time() - feedback_start)
        try:
            deps["FEEDBACK_TASK_FAILURES"].labels(stage="persist").inc()
        except Exception:
            pass
        deps["logger"].exception(f"[Background] Critical fail: {exc}")
