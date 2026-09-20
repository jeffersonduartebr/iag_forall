# -*- coding: utf-8 -*-
# Objective: NSGA-II-coupled reward function for the contextual bandit.
"""Compute the bandit reward with objective weights published by the NSGA-II cycle.

The NSGA-II updater optimizes a per-model portfolio and tunes the routing
weights ``NSGA_W_QUALITY/LATENCY/COST``, which live on raw scales (quality 0-10,
latency in seconds, cost in USD per request). The reward, in contrast, combines
three transfer functions already normalized to [0, 1]. To couple both layers the
updater converts the raw weights into dimensionless *shares*: the fraction each
objective contributes to the routing score at the operating point chosen by the
optimizer (``W_Q*q, W_L*l, W_C*c``). The shares are published per modality in
Redis (``nsga:reward_weights:<modality>``) and read here. When they are missing
the reward falls back to fixed defaults and reports it through Prometheus, so a
decoupled reward is observable instead of silent.
"""

from __future__ import annotations

import json
import logging
import math
import time
from typing import Any, Dict, Iterable, Optional, Tuple

from app.settings_dynamic import settings

logger = logging.getLogger(__name__)

Weights = Tuple[float, float, float]

DEFAULT_REWARD_WEIGHTS: Weights = (0.55, 0.30, 0.15)
REWARD_WEIGHTS_KEY = "nsga:reward_weights:{modality}"
DEFAULT_MIN_SHARE = 0.05

# Cost baseline C_base (USD per 1k tokens), recalibrated 2026-09-18 with
# ``calibrate_cost_baseline`` over the CANDIDATE_MODELS_LIST pool at OpenRouter
# list prices (in/out per 1k): gpt-5.5 0.005/0.030, claude-opus-4.8 0.005/0.025,
# claude-sonnet-5 0.002/0.010, claude-haiku-4.5 0.001/0.005, claude-fable-5
# 0.010/0.050, with a 3:1 input:output token mix (RAG context dominates the
# prompt). The former 0.12 sat above every candidate, so the cost term was 1.0
# for all models. Recalibrate whenever the pool or its prices change.
DEFAULT_COST_BASELINE_PER_1K = 0.007
DEFAULT_COST_INPUT_SHARE = 0.75

# Latency transfer: inverted logistic with slope k and inflection x0 (seconds).
LATENCY_K = 0.12
LATENCY_X0_S = 20.0

# Dynamic inflection: x0(N) = TTFT budget + N / target rate, i.e. the *fair
# deadline* for an answer of that length. With a fixed x0 the same 20 s window
# was given to an 80-token arithmetic answer and to an 800-token discursive one,
# so a humanities item was penalised for being long rather than for being slow.
#
# DEFAULT_COMPLETION_TOKENS is chosen so that x0(375) = 5.0 + 375/25 = 20.0 s,
# exactly the constant above. Every call site that does not know the token count
# therefore keeps the previous behaviour bit for bit, and the new normalisation
# is the identity at the point where the old calibration sat.
LATENCY_TTFT_BUDGET_S = 5.0
LATENCY_TOKENS_PER_S = 25.0
DEFAULT_COMPLETION_TOKENS = 375

# Beyond this, more tokens stop buying deadline: without a ceiling a model could
# pad its answer to widen its own window, and latency would stop discriminating.
MAX_BUDGETED_COMPLETION_TOKENS = 8000
# Cost transfer floor: expensive models are never excluded from exploration.
COST_SCORE_FLOOR = 0.3

_FALLBACK_WARN_INTERVAL_S = 300.0
_last_fallback_warn = 0.0


def _get_rds():
    """Return the shared Redis client without blocking the hot path."""
    from app.utils.redis_client import ensure_redis_connected, get_redis_async_safe

    return get_redis_async_safe() or ensure_redis_connected(max_wait_s=0.0, min_retry_interval_s=2.0)


def _setting_float(key: str, default: float) -> float:
    try:
        value = float(settings.get(key, default))
        return value if math.isfinite(value) else default
    except Exception:
        return default


def _normalize(values: Iterable[Any]) -> Optional[Weights]:
    """Normalize three non-negative values to sum 1; ``None`` when degenerate."""
    clean = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        clean.append(max(0.0, f) if math.isfinite(f) else 0.0)
    total = sum(clean)
    if len(clean) != 3 or total <= 0:
        return None
    return (clean[0] / total, clean[1] / total, clean[2] / total)


def _apply_floor(shares: Weights, floor: float) -> Weights:
    """Raise shares below ``floor`` to it and rescale the others to keep sum 1."""
    out = list(shares)
    fixed: set[int] = set()
    while True:
        new_low = {i for i in range(3) if i not in fixed and out[i] < floor}
        if not new_low:
            break
        fixed |= new_low
        free = [i for i in range(3) if i not in fixed]
        budget = 1.0 - floor * len(fixed)
        free_total = sum(shares[i] for i in free)
        for i in fixed:
            out[i] = floor
        for i in free:
            out[i] = shares[i] / free_total * budget if free_total > 0 else budget / len(free)
    return (out[0], out[1], out[2])


def derive_reward_weights(
    w_quality: float,
    w_latency: float,
    w_cost: float,
    quality_bar: float,
    latency_bar: float,
    cost_bar: float,
    min_share: float = DEFAULT_MIN_SHARE,
) -> Weights:
    """Convert raw NSGA routing weights into reward shares at one operating point.

    Each share is the contribution of one objective to the routing score
    ``W_Q*q - W_L*l - W_C*c`` evaluated at the system metrics of the portfolio
    selected by the NSGA-II (``quality_bar`` on 0-10, ``latency_bar`` in seconds,
    ``cost_bar`` in USD per request). Shares are floored at ``min_share`` so no
    objective silently drops out of the reward.
    """
    shares = _normalize((w_quality * quality_bar, w_latency * latency_bar, w_cost * cost_bar))
    floor = max(0.0, min(float(min_share), 1.0 / 3.0))
    # Ponto de operação degenerado: usa os pesos padrão, mas o piso vale também para eles.
    return _apply_floor(shares if shares is not None else DEFAULT_REWARD_WEIGHTS, floor)


def _parse_weights(raw: Any) -> Optional[Weights]:
    if not raw:
        return None
    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        obj = json.loads(raw)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    return _normalize((obj.get("quality"), obj.get("latency"), obj.get("cost")))


def _observe_weights(weights: Weights, modality: str, source: str) -> None:
    try:
        from app.observability import REWARD_WEIGHT, REWARD_WEIGHTS_SOURCE

        REWARD_WEIGHTS_SOURCE.labels(source=source).inc()
        for name, value in zip(("quality", "latency", "cost"), weights):
            REWARD_WEIGHT.labels(objective=name, modality=modality).set(value)
    except Exception:
        pass


def _warn_fallback(modality: str) -> None:
    global _last_fallback_warn
    now = time.time()
    if now - _last_fallback_warn >= _FALLBACK_WARN_INTERVAL_S:
        _last_fallback_warn = now
        logger.warning(
            "[reward] Sem pesos publicados pelo NSGA-II (%s); usando padrão %s",
            REWARD_WEIGHTS_KEY.format(modality=modality),
            DEFAULT_REWARD_WEIGHTS,
        )


def load_reward_weights(modality: str = "text") -> Tuple[Weights, str]:
    """Return ``((w_q, w_l, w_c), source)`` for one modality.

    Lookup order: the modality's published shares, then the ``text`` shares,
    then ``DEFAULT_REWARD_WEIGHTS``. ``source`` is ``"nsga"`` or ``"default"``.
    """
    modality = (modality or "text").strip().lower() or "text"
    rds = _get_rds()
    if rds:
        for mod in dict.fromkeys((modality, "text")):
            try:
                weights = _parse_weights(rds.get(REWARD_WEIGHTS_KEY.format(modality=mod)))
            except Exception as exc:
                logger.debug("[reward] Falha ao ler pesos (%s): %s", mod, exc)
                break
            if weights:
                _observe_weights(weights, modality, "nsga")
                return weights, "nsga"
    _warn_fallback(modality)
    _observe_weights(DEFAULT_REWARD_WEIGHTS, modality, "default")
    return DEFAULT_REWARD_WEIGHTS, "default"


def publish_reward_weights(
    rds: Any,
    modality: str,
    weights: Weights,
    sys_metrics: Optional[Dict[str, float]] = None,
) -> bool:
    """Publish reward shares for one modality; returns whether the write succeeded."""
    if not rds:
        return False
    payload = {
        "quality": round(float(weights[0]), 6),
        "latency": round(float(weights[1]), 6),
        "cost": round(float(weights[2]), 6),
        "updated_at": time.time(),
        "source": "nsga-updater",
        "sys_metrics": dict(sys_metrics or {}),
    }
    try:
        rds.set(REWARD_WEIGHTS_KEY.format(modality=modality), json.dumps(payload))
        return True
    except Exception as exc:
        logger.warning("[reward] Falha ao publicar pesos (%s): %s", modality, exc)
        return False


def cost_per_1k_from_total(total_cost_usd: Any, prompt_tokens: Any, completion_tokens: Any) -> Optional[float]:
    """Convert the total cost of one call into USD per 1k tokens (``None`` if unknown)."""
    try:
        tokens = int(prompt_tokens or 0) + int(completion_tokens or 0)
        cost = float(total_cost_usd or 0.0)
    except (TypeError, ValueError):
        return None
    if tokens <= 0 or not math.isfinite(cost):
        return None
    return max(0.0, cost) / tokens * 1000.0


def blended_cost_per_1k(input_per_1k: float, output_per_1k: float, input_share: float) -> float:
    """Price per 1k tokens for a traffic mix with ``input_share`` of prompt tokens."""
    share = max(0.0, min(1.0, float(input_share)))
    return share * float(input_per_1k) + (1.0 - share) * float(output_per_1k)


def calibrate_cost_baseline(
    prices: Iterable[Tuple[float, float]],
    input_share: float = DEFAULT_COST_INPUT_SHARE,
) -> float:
    """Geometric mean of the pool's blended prices, the reference point for C_base.

    The cost transfer penalizes the *ratio* to the baseline, so the geometric
    mean is its natural centre: cheaper models keep the full score, pricier ones
    decay until the floor at ``1 / COST_SCORE_FLOOR`` times the baseline.
    """
    blended = [blended_cost_per_1k(i, o, input_share) for i, o in prices]
    positive = [b for b in blended if b > 0]
    if not positive:
        return DEFAULT_COST_BASELINE_PER_1K
    return math.exp(sum(math.log(b) for b in positive) / len(positive))


def latency_threshold_s(
    completion_tokens: Optional[int] = None,
    *,
    ttft: Optional[float] = None,
    tokens_per_s: Optional[float] = None,
) -> float:
    """The fair deadline for an answer of this length: TTFT budget + generation time.

    An unknown, negative or non-finite token count falls back to
    ``DEFAULT_COMPLETION_TOKENS``, which reproduces the former static threshold
    exactly. The count is capped so a verbose answer cannot buy itself an
    unbounded window.
    """
    budget = LATENCY_TTFT_BUDGET_S if ttft is None else ttft
    rate = LATENCY_TOKENS_PER_S if tokens_per_s is None else tokens_per_s
    if rate <= 0 or not math.isfinite(rate):
        rate = LATENCY_TOKENS_PER_S
    try:
        tokens = int(completion_tokens) if completion_tokens is not None else DEFAULT_COMPLETION_TOKENS
    except (TypeError, ValueError):
        tokens = DEFAULT_COMPLETION_TOKENS
    if tokens < 0:
        tokens = DEFAULT_COMPLETION_TOKENS
    tokens = min(tokens, MAX_BUDGETED_COMPLETION_TOKENS)
    return float(budget) + tokens / float(rate)


def latency_score(
    latency_s: float,
    completion_tokens: Optional[int] = None,
    *,
    k: Optional[float] = None,
    x0: Optional[float] = None,
) -> float:
    """Inverted logistic: ~1 when well inside the deadline, 0.5 at it, → 0 beyond.

    ``x0`` may be passed in already resolved. :func:`compute_reward` does that so
    the settings are read once per reward rather than once per transfer — this
    function is on the hot path and is covered by a benchmark.
    """
    slope = LATENCY_K if k is None else k
    midpoint = latency_threshold_s(completion_tokens) if x0 is None else x0
    z = slope * (float(latency_s) - midpoint)
    if z > 700:  # math.exp overflow guard
        return 0.0
    return max(0.0, min(1.0, 1.0 / (1.0 + math.exp(z))))


def cost_score(cost_per_1k: Optional[float], baseline_per_1k: float) -> float:
    """Ratio penalty above the baseline, floored at ``COST_SCORE_FLOOR``; neutral if unknown."""
    if cost_per_1k is None:
        return 1.0
    ratio = float(cost_per_1k) / baseline_per_1k if baseline_per_1k > 0 else 1.0
    score = 1.0 / (1.0 + max(0.0, ratio - 1.0))
    return max(COST_SCORE_FLOOR, min(1.0, score))


def compute_reward(
    model: str,
    quality: float,
    latency_s: float,
    cost_per_1k: Optional[float] = None,
    modality: str = "text",
    *,
    completion_tokens: Optional[int] = None,
) -> float:
    """Convert observed metrics into a reward in [0, 1] (thesis eq. ``eq:recompensa``).

    ``quality`` is on the 0-10 judge scale, ``latency_s`` in seconds and
    ``cost_per_1k`` in USD per 1k tokens (``None`` means unknown and is neutral).
    Weights come from :func:`load_reward_weights`.

    ``completion_tokens`` is keyword-only on purpose: every existing caller, and
    every test double that mirrors this signature positionally, keeps working
    untouched. When it is omitted the latency threshold falls back to the value
    that reproduces the former static calibration.

    The dynamic threshold is applied only when ``REWARD_DYNAMIC_LATENCY_ENABLED``
    is on. It shifts the reward distribution substantially — a short answer gets
    a much tighter deadline than before — and the promotion gate downstream is an
    absolute threshold, so the two have to be moved together.
    """
    try:
        (w_q, w_l, w_c), _source = load_reward_weights(modality)
        q = max(0.0, min(10.0, float(quality))) / 10.0
        baseline = _setting_float("REWARD_COST_BASELINE_PER_1K", DEFAULT_COST_BASELINE_PER_1K)

        # Resolved once per reward: latency_score sits on the hot path.
        slope = _setting_float("REWARD_LATENCY_K", LATENCY_K)
        if _setting_float("REWARD_DYNAMIC_LATENCY_ENABLED", 0.0) >= 1.0:
            midpoint = latency_threshold_s(
                completion_tokens,
                ttft=_setting_float("REWARD_LATENCY_TTFT_BUDGET_S", LATENCY_TTFT_BUDGET_S),
                tokens_per_s=_setting_float("REWARD_LATENCY_TOKENS_PER_S", LATENCY_TOKENS_PER_S),
            )
        else:
            midpoint = LATENCY_X0_S

        reward = (
            w_q * q
            + w_l * latency_score(latency_s, k=slope, x0=midpoint)
            + w_c * cost_score(cost_per_1k, baseline)
        )
        return max(0.0, min(1.0, float(reward)))
    except Exception as exc:
        logger.warning("[reward] Falha em compute_reward (%s): %s", model, exc)
        return 0.0
