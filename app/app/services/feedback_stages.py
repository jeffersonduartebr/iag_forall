# -*- coding: utf-8 -*-
# Objective: Stages of the background feedback loop (judge -> reward -> bandit -> EMA -> log).
"""Stages used by ``process_background_feedback_impl``.

The feedback loop runs after a response is returned: estimate error risk,
decide whether to spend a judge call, score quality, turn it into a bandit
reward, update the bandit / exploration stats / EMA, cache good answers and
persist the query log. Each stage is a small function over the injected
``deps`` (same keys as before) and a :class:`FeedbackRequest`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .ema_store import next_ema, update_shared_ema
from .formative_observability import record_routing_cell
from .reward import cost_per_1k_from_total
from .router_services import spawn_via_deps
from .router_stages import quietly

CACHE_MIN_QUALITY = 7.0


@dataclass
class FeedbackRequest:
    """One served answer to learn from."""

    query: str
    answer: str
    chosen_model: str
    modality: str
    latency_s: float
    cost_val: float
    image_b64: Optional[str] = None
    raw_payload: Any = None
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def payload(self) -> Dict[str, Any]:
        return self.raw_payload if isinstance(self.raw_payload, dict) else {}

    @property
    def explored(self) -> bool:
        return bool(self.payload.get("openrouter_exploration"))


@dataclass
class ErrorRisk:
    model_stats: Dict[str, Any]
    predictor: Any
    query_embedding: Any
    predicted_error_prob: Optional[float]


@dataclass
class Quality:
    value: float
    source: str
    judge_rubric: Optional[Dict[str, Any]] = None


@dataclass
class JudgeDecision:
    should_judge: bool
    probability: float = 0.0
    reasons: List[str] = field(default_factory=list)


def _setting(deps: Dict[str, Any], key: str, default: Any) -> Any:
    return getattr(deps["settings"], "get", lambda k, d=None: d)(key, default)


async def assess_error_risk(deps: Dict[str, Any], fb: FeedbackRequest) -> ErrorRisk:
    """Global bandit stats for the model plus the online predictor's error probability."""
    model_stats = deps["_get_ctx_stats"]("global").get(fb.chosen_model, {})
    predictor = deps["get_predictor"](fb.chosen_model)
    embedding = await deps["asyncio"].to_thread(deps["embed_text"], fb.query)
    return ErrorRisk(model_stats, predictor, embedding, predictor.predict_error_probability(embedding))


def _pending_query_jobs(deps: Dict[str, Any]) -> int:
    try:
        return max(0, int(deps.get("get_pending_query_jobs_count", lambda: 0)() or 0))
    except Exception:
        return 0


def _backlog_threshold(deps: Dict[str, Any]) -> int:
    try:
        return max(1, int(_setting(deps, "QUERY_JOB_BACKGROUND_THROTTLE_PENDING_THRESHOLD", "1")))
    except Exception:
        return 1


def judge_throttle_reason(deps: Dict[str, Any]) -> Optional[str]:
    """Why the judge must be skipped now (queued interactive work or provider pressure), if at all."""
    pending, threshold = _pending_query_jobs(deps), _backlog_threshold(deps)
    if pending >= threshold:
        deps["logger"].info(
            "[Background] Judge throttled because queued query backlog=%s exceeds threshold=%s", pending, threshold
        )
        return "query_backlog"
    throttle_enabled = str(_setting(deps, "JUDGE_BACKGROUND_THROTTLE_ENABLED", "1")).strip() == "1"
    if throttle_enabled and deps.get("should_throttle_background_judge", lambda: False)():
        deps["logger"].info("[Background] Judge throttled to preserve interactive Ollama capacity")
        return "provider_pressure"
    return None


def decide_judging(deps: Dict[str, Any], fb: FeedbackRequest, risk: ErrorRisk) -> JudgeDecision:
    """Sample the judge with a risk-aware probability; always judge exploration picks."""
    probability = deps["compute_judge_probability"](
        n_samples=risk.model_stats.get("count", 0),
        predicted_error_prob=risk.predicted_error_prob,
        chosen_model=fb.chosen_model,
        min_sample_rate=deps["settings"].JUDGE_MIN_SAMPLE_RATE,
    )
    reason = judge_throttle_reason(deps)
    if reason is not None:
        quietly(lambda: deps["BACKGROUND_JUDGE_SKIPPED"].labels(reason=reason).inc())
        decision = JudgeDecision(False, probability, [reason])
    else:
        decision = JudgeDecision(deps["random"].random() < probability, probability)
    if fb.explored:
        deps["logger"].info("[Background] Forcing judge for OpenRouter exploration on %s", fb.chosen_model)
        decision.should_judge = True
    return decision


def _rubric_summary(judge_scores: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    entry = next((s for s in judge_scores if s.get("judge_id") == "llm_rubric"), None)
    if entry is None:
        return None
    keys = (
        "score", "dimensions", "dispersion", "n_judges", "judges",
        "q_tech", "q_calibrado", "p_entrega", "calibration_status",
    )
    return {k: entry.get(k) for k in keys}


def _learn_from_judgment(risk: ErrorRisk, quality: float) -> None:
    is_correct = quality >= 7.0
    risk.predictor.learn(risk.query_embedding, is_correct)
    risk.predictor.record_outcome(risk.predicted_error_prob, not is_correct)
    risk.predictor.maybe_save()


async def judge_quality(deps: Dict[str, Any], fb: FeedbackRequest, risk: ErrorRisk, decision: JudgeDecision) -> Quality:
    """Judge score on 0-10 (heuristic fallback flagged); predictor learns only from real judgments."""
    deps["logger"].info(
        f"[Background] Sampling Judge for {fb.chosen_model} "
        f"(p={decision.probability:.2f}, pred_err={risk.predicted_error_prob:.2f})"
    )
    try:
        scores = await deps["judge_answer"](fb.query, fb.answer, evaluated_model=fb.chosen_model)
        valid = [s["score"] for s in scores if "score" in s]
        value = round((float(np.mean(valid)) if valid else 5.0) * 10.0, 2)
        heuristic_only = bool(scores) and all(s.get("judge_id") == "heuristic_fallback" for s in scores)
        # Nenhum juiz LLM respondeu: medição grosseira, sinalizada e fora do preditor.
        source = "heuristic_fallback" if heuristic_only else "judge"
        if source == "judge":
            _learn_from_judgment(risk, value)
        return Quality(value, source, _rubric_summary(scores))
    except Exception:
        quietly(lambda: deps["FEEDBACK_TASK_FAILURES"].labels(stage="judge").inc())
        return Quality(5.0, "fallback_default")


def proxy_quality(risk: ErrorRisk) -> Quality:
    """Unjudged answers: the bandit mean (a reward, not literal quality) as a labelled proxy."""
    return Quality(max(0.0, min(10.0, risk.model_stats.get("mean", 0.5) * 10.0)), "bandit_proxy")


def learned_quality(quality: Quality) -> float:
    """The score the bandit learns from: calibrated under formative semantics.

    This is the only place the switch actually changes what is optimised. The
    error predictor and the cache gate keep reading ``quality.value``, because
    they answer "does this answer serve?" and not "did it teach?" — feeding
    usurpation into the error predictor would make it forecast pedagogical
    failure, which is not what the router uses to decide effort.

    Falls back to ``quality.value`` whenever the calibrated score is missing, so
    a judged row never loses its reward because the usurpation judge was down.
    """
    from .quality_semantics import is_formative

    if not is_formative():
        return quality.value
    rubric = quality.judge_rubric or {}
    calibrated = rubric.get("q_calibrado")
    if calibrated is None or rubric.get("calibration_status") != "calibrated":
        return quality.value
    return float(calibrated)


def update_bandit(deps: Dict[str, Any], fb: FeedbackRequest, quality: Quality) -> float:
    """Reward from quality/latency/cost (per 1k tokens) and the contextual bandit update."""
    try:
        cost_per_1k = cost_per_1k_from_total(fb.cost_val, fb.prompt_tokens, fb.completion_tokens)
        reward = deps["compute_reward"](
            fb.chosen_model,
            learned_quality(quality),
            fb.latency_s,
            cost_per_1k,
            modality=fb.modality,
            completion_tokens=fb.completion_tokens,
        )
    except Exception:
        reward = 0.0
    try:
        deps["bandit_update"](model=fb.chosen_model, query=fb.query, reward=reward, modality=fb.modality)
    except Exception as exc:
        deps["logger"].warning(f"[Background] Bandit fail: {exc}")
        quietly(lambda: deps["FEEDBACK_TASK_FAILURES"].labels(stage="bandit_update").inc())
    return reward


async def _record_exploration_stats(deps: Dict[str, Any], fb: FeedbackRequest, reward: float, quality: Quality) -> None:
    record_fn = deps.get("record_openrouter_exploration")
    if record_fn is None:
        return
    try:
        promotion = await record_fn(
            model=fb.chosen_model,
            reward=reward,
            latency_s=fb.latency_s,
            cost_usd=float(fb.cost_val or 0.0),
            settings=deps["settings"],
            prompt_tokens=int(fb.prompt_tokens or 0),
            completion_tokens=int(fb.completion_tokens or 0),
            success=bool(fb.answer and str(fb.answer).strip()),
            judge_quality=quality.value if quality.source == "judge" else None,
        )
        if promotion and promotion.get("auto_promoted"):
            deps["logger"].info("[Background] Auto-promoted exploration model %s to candidates", fb.chosen_model)
    except Exception as exc:
        deps["logger"].warning("[Background] OpenRouter exploration record failed: %s", exc)


async def _shadow_compare(deps: Dict[str, Any], fb: FeedbackRequest, quality: Quality) -> None:
    shadow_fn = deps.get("maybe_run_shadow_comparison")
    info = fb.payload.get("exploration_info") or {}
    if shadow_fn is None or not info.get("shadow_compare"):
        return
    try:
        await shadow_fn(
            query=fb.query,
            explored_model=fb.chosen_model,
            explored_answer=fb.answer,
            explored_quality=quality.value,
            explored_latency=fb.latency_s,
            explored_cost=float(fb.cost_val or 0.0),
            incumbent_model=info.get("incumbent_model"),
            deps=deps,
            settings=deps["settings"],
        )
    except Exception as exc:
        deps["logger"].warning("[Background] Shadow comparison failed: %s", exc)


async def record_exploration(deps: Dict[str, Any], fb: FeedbackRequest, reward: float, quality: Quality) -> None:
    """OpenRouter exploration bookkeeping (stats/promotion and shadow comparison)."""
    if not fb.explored:
        return
    await _record_exploration_stats(deps, fb, reward, quality)
    await _shadow_compare(deps, fb, quality)


def _share_and_persist_ema(deps: Dict[str, Any], fb: FeedbackRequest, quality: float, local: Dict[str, Any]) -> None:
    """Fold the observation into the shared (Redis) EMA and persist that entry to ema_history."""
    shared = deps.get("update_shared_ema", update_shared_ema)(
        fb.modality, fb.chosen_model, fb.latency_s, quality, fb.cost_val
    )
    deps["_persist_ema"](fb.modality, fb.chosen_model, shared or local)


def update_ema(deps: Dict[str, Any], state: Dict[str, Any], fb: FeedbackRequest, quality: Quality) -> None:
    """EMA per (modality, model): in-process copy, shared Redis copy (routing) and ema_history."""
    try:
        key = (fb.modality, fb.chosen_model)
        entry = next_ema(state["EMA_HISTORY"].get(key), fb.latency_s, quality.value, fb.cost_val)
        state["EMA_HISTORY"].set(key, entry)
        spawn_via_deps(
            deps,
            deps["asyncio"].to_thread(_share_and_persist_ema, deps, fb, quality.value, entry),
            name="ema_persist",
        )
    except Exception as exc:
        deps["logger"].warning(f"[Background] EMA update failed: {exc}")
        quietly(lambda: deps["FEEDBACK_TASK_FAILURES"].labels(stage="ema_update").inc())


async def maybe_store_cache(deps: Dict[str, Any], fb: FeedbackRequest, quality: Quality) -> None:
    """Semantic-cache only answers judged/proxied at CACHE_MIN_QUALITY or above."""
    if quality.value < CACHE_MIN_QUALITY:
        return
    try:
        await deps["store_cache"](
            query=fb.query,
            answer=fb.answer,
            modality=fb.modality,
            image_b64=fb.image_b64,
            model_used=fb.chosen_model,
            tenant_id=fb.payload.get("tenant_id"),
        )
    except Exception as exc:
        deps["logger"].warning(f"[Background] Cache store failed: {exc}")
        quietly(lambda: deps["FEEDBACK_TASK_FAILURES"].labels(stage="cache_write").inc())


from .log_fields import _formative_fields, _reliability_fields, _research_fields  # noqa: E402  (evita ciclo)


class FeedbackPersistError(RuntimeError):
    """The query_log row could not be written.

    Raised so the Celery task is marked FAILURE instead of SUCCESS. It is
    deliberately *not* retried: the judges have already run and been paid for,
    so repeating the pipeline would buy the same row twice.
    """


def persist_log(
    deps: Dict[str, Any],
    fb: FeedbackRequest,
    quality: Optional[Quality],
    judged: bool,
    risk: ErrorRisk,
    reward: Optional[float],
) -> None:
    """Quality metrics and the query_log row (with the rubric breakdown when judged).

    ``quality`` is None when a stage failed before any quality was known: the row is still written, with
    ``quality`` NULL and ``quality_source = 'feedback_error'``.
    """
    if quality is None:
        _gravar_linha(deps, fb, None, "feedback_error", None, judged, risk, reward)
        return
    quietly(lambda: deps["ROUTER_QUALITY_AVG"].labels(model=fb.chosen_model).set(quality.value))
    if "ollama" in fb.chosen_model:
        quietly(lambda: deps["ROUTER_LOCAL_USAGE_RATIO"].set(1.0))
    quietly(
        lambda: record_routing_cell(
            fb.chosen_model,
            fb.payload.get("detected_complexity"),
            quality.value,
            quality.source,
        )
    )
    _gravar_linha(deps, fb, quality.value, quality.source, quality, judged, risk, reward)


def _gravar_linha(deps, fb, valor, fonte, quality, judged, risk, reward) -> None:
    raw_payload = fb.raw_payload
    if quality is not None and quality.judge_rubric is not None and isinstance(raw_payload, dict):
        raw_payload = {**raw_payload, "judge_rubric": quality.judge_rubric}
    try:
        deps["insert_query_log"](
            query_text=fb.query,
            model=fb.chosen_model,
            modality=fb.modality,
            image_provided=bool(fb.image_b64),
            answer=fb.answer,
            image_output_b64=None,
            latency_s=fb.latency_s,
            estimated_cost_usd=fb.cost_val,
            quality=valor,
            quality_source=fonte,
            judge_sampled=judged,
            predicted_error_prob=None if risk.predicted_error_prob is None else float(risk.predicted_error_prob),
            reward=reward,
            **_formative_fields(quality, fb),
            context_label="async_processed",
            tenant_id=fb.payload.get("tenant_id"),
            raw_payload=raw_payload,
            query_embedding=risk.query_embedding,
            answer_embedding=None,
            **_reliability_fields(fb.payload),
            **_research_fields(fb.payload),
        )
    except Exception as exc:
        # Não engolir: com o MariaDB em baixo, a tarefa Celery terminava em
        # SUCCESS e nenhuma linha de query_log era escrita. O ciclo de
        # aprendizagem parava em silêncio e o único sinal era um contador que
        # ninguém tinha alertado.
        deps["logger"].error(f"[Background] query_log NÃO foi escrito: {exc}")
        quietly(lambda: deps["FEEDBACK_TASK_FAILURES"].labels(stage="persist").inc())
        raise FeedbackPersistError(str(exc)) from exc


def observe_backlog_age(deps: Dict[str, Any], fb: FeedbackRequest, now: float) -> None:
    enqueued_at = fb.payload.get("queue_enqueued_at")
    if isinstance(enqueued_at, (int, float)):
        quietly(lambda: deps["FEEDBACK_BACKLOG_AGE"].set(max(0.0, now - float(enqueued_at))))
