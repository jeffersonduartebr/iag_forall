# Objective: Closed-loop adversarial governance — cluster risk memory + online feedback.
"""Adversarial governance service (roadmap #17).

Turns adversarial red-teaming from an offline measurement into an active
governance loop. It provides three capabilities:

1. **Cluster-level risk memory** — accumulates per-knowledge-cluster attack
   outcomes (Redis-backed, in-memory fallback) so the system remembers *where*
   the Tutor is fragile, not just *which model* is globally weak.
2. **Online-loop closure** — routes adversarial audit verdicts into the very same
   learning path organic traffic uses in ``services.router_feedback``: it updates
   the contextual bandits (``bandit_update``) and retrains the per-model
   ``online_predictor`` of failure probability.
3. **Risk-based escalation** — when a cluster is high-risk (or epistemic
   uncertainty is high), suggests escalating the chosen model to a stronger
   (typically cloud) candidate.

Everything is gated by ``ADVGOV_ENABLED`` (off by default) and is written to never
raise into callers: the hot path degrades gracefully to a no-op.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# In-memory fallback store used when Redis is unavailable (also exercised by tests).
_MEM_CLUSTERS: Dict[str, Dict[str, float]] = {}

_LOCAL_PREFIXES = ("ollama/",)
_CLUSTER_KEY_PREFIX = "advgov:cluster:"


# --------------------------------------------------------------------------- #
# Settings helpers (defensive: never raise, always return a usable default)
# --------------------------------------------------------------------------- #
def _settings() -> Any:
    from ..settings_dynamic import settings

    return settings


def _cfg_bool(key: str, default: bool) -> bool:
    try:
        raw = _settings().get(key, "1" if default else "0")
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}
    except Exception:
        return default


def _cfg_float(key: str, default: float) -> float:
    try:
        return float(_settings().get(key, default))
    except Exception:
        return default


def _cfg_int(key: str, default: int) -> int:
    try:
        return int(float(_settings().get(key, default)))
    except Exception:
        return default


def _cfg_list(key: str) -> List[str]:
    try:
        raw = _settings().get(key, "[]")
    except Exception:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw if str(x).strip()]
    text = str(raw).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x) for x in parsed if str(x).strip()]
    except Exception:
        pass
    return [part.strip() for part in text.split(",") if part.strip()]


def _enabled() -> bool:
    return _cfg_bool("ADVGOV_ENABLED", False)


def _is_local(model: Optional[str]) -> bool:
    return bool(model) and any(str(model).startswith(p) for p in _LOCAL_PREFIXES)


# --------------------------------------------------------------------------- #
# Cluster risk store
# --------------------------------------------------------------------------- #
def _cluster_key(cluster_id: str) -> str:
    return f"{_CLUSTER_KEY_PREFIX}{cluster_id}"


def _read_redis() -> Any:
    try:
        from ..utils.redis_client import get_redis_sync_nonblocking

        return get_redis_sync_nonblocking()
    except Exception:
        return None


def _load_cluster(cluster_id: str) -> Dict[str, float]:
    rds = _read_redis()
    if rds is not None:
        try:
            raw = rds.get(_cluster_key(cluster_id))
            if raw:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                return dict(json.loads(raw))
        except Exception as exc:
            logger.debug("[advgov] redis load failed for %s: %s", cluster_id, exc)
    return dict(_MEM_CLUSTERS.get(cluster_id) or {})


def _store_cluster(cluster_id: str, data: Dict[str, float]) -> None:
    _MEM_CLUSTERS[cluster_id] = data
    rds = _read_redis()
    if rds is not None:
        try:
            ttl = max(60, _cfg_int("ADVGOV_CLUSTER_TTL_S", 2592000))
            rds.set(_cluster_key(cluster_id), json.dumps(data), ex=ttl)
        except Exception as exc:
            logger.debug("[advgov] redis store failed for %s: %s", cluster_id, exc)


def get_cluster_risk(cluster_id: Optional[str]) -> Dict[str, Any]:
    """Return the current risk snapshot for a knowledge cluster.

    ``high_risk`` requires both a minimum sample count and a failure rate at or
    above the configured threshold, so a single unlucky duel cannot flip a cluster.
    """
    if not cluster_id:
        return {"cluster_id": None, "n": 0, "failures": 0, "failure_rate": 0.0, "mean_score": None, "high_risk": False}
    data = _load_cluster(str(cluster_id))
    n = int(data.get("n", 0) or 0)
    failures = int(data.get("failures", 0) or 0)
    score_sum = float(data.get("score_sum", 0.0) or 0.0)
    failure_rate = (failures / n) if n > 0 else 0.0
    mean_score = (score_sum / n) if n > 0 else None
    min_samples = max(1, _cfg_int("ADVGOV_CLUSTER_MIN_SAMPLES", 5))
    threshold = _cfg_float("ADVGOV_CLUSTER_FAILURE_RATE_THRESHOLD", 0.5)
    high_risk = n >= min_samples and failure_rate >= threshold
    return {
        "cluster_id": str(cluster_id),
        "n": n,
        "failures": failures,
        "failure_rate": failure_rate,
        "mean_score": mean_score,
        "high_risk": high_risk,
        "last_strategy": data.get("last_strategy"),
    }


# --------------------------------------------------------------------------- #
# Online-loop closure (same path organic traffic uses in router_feedback)
# --------------------------------------------------------------------------- #
def _close_online_loop(
    *,
    model: str,
    score: float,
    is_failure: bool,
    query: Optional[str],
    embedding: Optional[Sequence[float]],
    latency_s: Optional[float],
    cost_per_1k: Optional[float],
) -> None:
    """Feed one adversarial verdict into bandits + online error predictor."""
    # 1) Contextual bandit reward (quality on the 0..10 scale, like router_feedback).
    try:
        from ..bandits import bandit_update, compute_reward

        reward = compute_reward(model, float(score), float(latency_s or 0.0), cost_per_1k)
        bandit_update(model=model, query=query or "", reward=reward)
    except Exception as exc:
        logger.debug("[advgov] bandit update failed for %s: %s", model, exc)

    # 2) Online error predictor (River). Needs an embedding; compute lazily.
    try:
        emb = list(embedding) if embedding is not None else None
        if emb is None and query:
            from ..embeddings import embed_text

            emb = embed_text(query)
        if emb:
            from ..online_predictor import get_predictor

            predictor = get_predictor(model)
            # Predict BEFORE learning to record the pre-update calibration point,
            # mirroring services.router_feedback.
            predicted_error_prob = predictor.predict_error_probability(emb)
            predictor.learn(emb, not is_failure)
            predictor.record_outcome(predicted_error_prob, is_failure)
            predictor.save()
    except Exception as exc:
        logger.debug("[advgov] predictor learn failed for %s: %s", model, exc)


def record_adversarial_outcome(
    *,
    cluster_id: str,
    model: str,
    score: float,
    strategy: Optional[str] = None,
    query: Optional[str] = None,
    embedding: Optional[Sequence[float]] = None,
    latency_s: Optional[float] = None,
    cost_per_1k: Optional[float] = None,
    learn: bool = True,
) -> Optional[Dict[str, Any]]:
    """Record one adversarial duel outcome and (optionally) close the online loop.

    ``score`` is the auditor/judge quality on a 0..10 scale. Returns the updated
    cluster risk snapshot, or ``None`` when governance is disabled.
    """
    if not _enabled():
        return None
    try:
        fail_threshold = _cfg_float("ADVGOV_FAIL_SCORE_THRESHOLD", 7.0)
        is_failure = float(score) < fail_threshold

        data = _load_cluster(str(cluster_id))
        data["n"] = int(data.get("n", 0) or 0) + 1
        data["failures"] = int(data.get("failures", 0) or 0) + (1 if is_failure else 0)
        data["score_sum"] = float(data.get("score_sum", 0.0) or 0.0) + float(score)
        if strategy:
            data["last_strategy"] = strategy
        _store_cluster(str(cluster_id), data)

        if learn:
            _close_online_loop(
                model=model,
                score=score,
                is_failure=is_failure,
                query=query,
                embedding=embedding,
                latency_s=latency_s,
                cost_per_1k=cost_per_1k,
            )
        return get_cluster_risk(str(cluster_id))
    except Exception as exc:
        logger.warning("[advgov] record_adversarial_outcome failed: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# Risk-based escalation
# --------------------------------------------------------------------------- #
def _pick_escalation_target(candidates: Sequence[str], current_model: Optional[str]) -> Optional[str]:
    """Choose a stronger candidate: configured priority first, else first cloud model."""
    configured = _cfg_list("ADVGOV_ESCALATION_MODELS")
    candidate_set = {c for c in candidates}
    for preferred in configured:
        if preferred in candidate_set and preferred != current_model:
            return preferred
    for candidate in candidates:
        if not _is_local(candidate) and candidate != current_model:
            return candidate
    return None


def suggest_escalation(
    *,
    cluster_id: Optional[str],
    candidate_models: Sequence[str],
    current_model: Optional[str] = None,
    uncertainty: Optional[float] = None,
) -> Optional[str]:
    """Suggest a stronger model when the cluster is high-risk or UQ is high.

    Returns the model to escalate to, or ``None`` to keep ``current_model``.
    """
    if not _enabled() or not _cfg_bool("ADVGOV_ESCALATION_ENABLED", True):
        return None
    candidates = [m for m in (candidate_models or []) if m]
    if not candidates:
        return None

    risk = get_cluster_risk(cluster_id)
    uq_threshold = _cfg_float("UNCERTAINTY_THRESHOLD", 0.7)
    high_uq = uncertainty is not None and float(uncertainty) >= uq_threshold
    if not (risk.get("high_risk") or high_uq):
        return None

    # Nothing to gain by escalating away from an already-cloud model unless an
    # explicitly-preferred stronger target is available.
    target = _pick_escalation_target(candidates, current_model)
    if target and target != current_model:
        return target
    return None


def advgov_escalate(
    deps: Dict[str, Any],
    chosen: str,
    top2: List[str],
    candidate_models: Sequence[str],
    uncertainty: Optional[float],
    runtime_hints: Optional[Dict[str, Any]],
) -> Any:
    """Router integration point: return ``(chosen, top2)`` after optional escalation.

    Self-gated (no-op unless ``ADVGOV_ENABLED``) and never raises into the hot path.
    """
    try:
        hints = runtime_hints or {}
        cluster_id = hints.get("benchmark_theme") or hints.get("theme")
        escalated = suggest_escalation(
            cluster_id=cluster_id,
            candidate_models=candidate_models,
            current_model=chosen,
            uncertainty=uncertainty,
        )
        if escalated and escalated != chosen:
            try:
                deps["logger"].info("[router] adv-governance escalation %s -> %s", chosen, escalated)
            except Exception:
                pass
            return escalated, [escalated]
    except Exception as exc:
        logger.debug("[advgov] route escalation skipped: %s", exc)
    return chosen, top2


def reset_state() -> None:
    """Clear in-memory cluster state (test helper)."""
    _MEM_CLUSTERS.clear()
