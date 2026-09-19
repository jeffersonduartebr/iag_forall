# -*- coding: utf-8 -*-
# Objective: Application runtime code for router strategy.
"""
router_strategy.py (Versão Final: Filtros Bilaterais de Segurança)
------------------------------------------------------------------
Estratégia de seleção sensível ao risco e otimizada por NSGA-II.
Inclui Hard Filters para impedir:
1. Modelos de Texto processarem Imagens.
2. Modelos de Visão processarem Texto simples (opcional para economia).

Atualizado:
- Integração com Circuit Breakers para filtrar modelos indisponíveis
- Penalização gradual baseada em fail_counter
- Detecção de cascata para emergency routing
- Fatores de risco dinâmicos via settings
"""

import logging
from typing import Dict, List, Tuple

# Importa helpers do bandits.py
from app.bandits import get_snapshot, sample_metrics_from_snapshot
from app.config.constants import DEFAULT_UNCERTAINTY_THRESHOLD
from app.model_registry import is_vision_only_model, model_supports_vision
from app.reliability import get_cascade_detector, get_circuit_breaker_manager
from app.services.ema_store import load_ema_snapshot, routing_latency_cost
from app.settings_dynamic import settings

logger = logging.getLogger(__name__)

SOTA_MARKERS = ["gpt-5", "opus", "sonnet", "gemini-3-pro"]

def _is_sota(model_name: str) -> bool:
    """Execute the is sota routine.

This helper encapsulates one focused step used by the surrounding workflow."""
    return any(m in model_name.lower() for m in SOTA_MARKERS)

def _is_local(model_name: str) -> bool:
    """Execute the is local routine.

This helper encapsulates one focused step used by the surrounding workflow."""
    return "ollama" in model_name.lower()

def _get_circuit_breaker_penalty(model: str) -> float:
    """
    Calculate a penalty factor based on circuit breaker state.

    Returns a multiplier (0.0 - 1.0) where:
    - 1.0 = no penalty (model healthy)
    - 0.0 = full penalty (circuit open)
    """
    try:
        manager = get_circuit_breaker_manager()
        status = manager.get_status(model)

        state = status.get("state", "closed")

        if state == "open":
            return 0.0  # Fully penalized
        elif state == "half-open":
            return 0.5  # Partially penalized

        # Gradual penalty based on fail_counter
        fail_counter = status.get("fail_counter", 0)
        fail_max = status.get("fail_max", 5)

        if fail_counter > 0 and fail_max > 0:
            # Linear penalty as failures accumulate
            penalty_ratio = fail_counter / fail_max
            return max(0.3, 1.0 - (penalty_ratio * 0.5))

        return 1.0  # No penalty
    except Exception:
        return 1.0  # Default to no penalty on error


def _filter_candidates(candidates: List[str], modality: str, breaker_manager) -> List[str]:
    """Drop models with an open circuit breaker, then apply the modality hard filters."""
    available = [m for m in candidates if breaker_manager.is_available(m)]
    if not available:
        logger.warning("[Strategy] ⚠️ All models have open circuit breakers! Using original candidates.")
        available = candidates
    elif len(available) < len(candidates):
        logger.info(f"[Strategy] Circuit breaker filtered {len(candidates) - len(available)} model(s)")

    # Visão exige capacidade multimodal; a fonte única de verdade é app.model_registry.
    if modality in ("vision", "multimodal"):
        vision = [m for m in available if model_supports_vision(m)]
        if vision:
            return vision
        # Se o filtro removeu tudo (ex: lista mal configurada), força um fallback seguro
        logger.warning("[Strategy] ⚠️ NENHUM modelo de visão encontrado na lista! Usando fallback Qwen3-VL.")
        return ["ollama/qwen3-vl:4b"]
    # Texto: remove modelos estritamente de visão (economia de VRAM/custo); "Omni" (gpt-4o) fica.
    if modality == "text":
        return [m for m in available if not is_vision_only_model(m)] or ["ollama/gemma3:4b"]
    return available


def _risk_factor(model: str, is_high_uncertainty: bool, risks: Tuple[float, float, float]) -> float:
    """UQ safety mode: trust frontier models on unknown ground, favor local ones on known ground."""
    sota_high_uq, local_high_uq, local_low_uq = risks
    if is_high_uncertainty:
        if _is_sota(model):
            return sota_high_uq
        return local_high_uq if _is_local(model) else 1.0
    return local_low_uq if _is_local(model) else 1.0


def model_score(quality: float, latency_s: float, cost_usd: float, weights: Tuple[float, float, float]) -> float:
    """S(m) = Q*w_Q - L*w_L - C*w_C (NSGA-II weights on raw scales)."""
    w_q, w_l, w_c = weights
    return quality * w_q - latency_s * w_l - cost_usd * w_c


def choose_top2_models(
    candidates: List[str],
    weights: Dict[str, float],  # Pesos do NSGA-II (w_quality, w_latency, w_cost)
    query_text: str,
    modality: str = "text",
    uncertainty_score: float = 0.0,
    min_quality: float = 0.0,  # Legacy
) -> List[str]:

    # ==================================================================
    # 🚨 CASCADE DETECTION - Emergency routing check
    # ==================================================================
    """Execute the choose top2 models routine.

This helper encapsulates one focused step used by the surrounding workflow."""
    cascade_detector = get_cascade_detector()
    if cascade_detector.is_emergency_mode:
        emergency_model = cascade_detector.get_emergency_fallback()
        if emergency_model:
            logger.warning(f"[Strategy] 🚨 EMERGENCY MODE: Using fallback {emergency_model}")
            return [emergency_model]

    snapshot = get_snapshot()
    ema_snapshot = load_ema_snapshot(modality)
    # Amostra qualidade probabilística (Thompson Sampling)
    sampled_qs = sample_metrics_from_snapshot(snapshot)

    # Limiar dinâmico definido pelo NSGA-II
    uq_threshold = float(settings.get("UNCERTAINTY_THRESHOLD", DEFAULT_UNCERTAINTY_THRESHOLD))

    is_high_uncertainty = uncertainty_score > uq_threshold

    candidates = _filter_candidates(candidates, modality, get_circuit_breaker_manager())

    # Define pesos padrão caso o NSGA-II ainda não tenha rodado ou retornado vazio
    w = (weights.get("w_quality", 1.0), weights.get("w_latency", 0.5), weights.get("w_cost", 50.0))
    risks = (settings.RISK_FACTOR_SOTA_HIGH_UQ, settings.RISK_FACTOR_LOCAL_HIGH_UQ, settings.RISK_FACTOR_LOCAL_LOW_UQ)

    scores = []
    for model in candidates:
        risk = _risk_factor(model, is_high_uncertainty, risks) * _get_circuit_breaker_penalty(model)
        # Latência e custo estimados: EMA compartilhada (feedback de todos os workers);
        # heurísticas só enquanto o modelo tiver poucas observações.
        avg_latency, est_cost = routing_latency_cost(
            ema_snapshot.get(model), is_local=_is_local(model), is_sota=_is_sota(model)
        )
        # Qualidade base (0-10) via Thompson Sampling, ajustada pelo risco
        score = model_score(sampled_qs.get(model, 5.0) * risk, avg_latency, est_cost, w)
        scores.append((model, score))

    # Ordena e pega top 2
    scores.sort(key=lambda x: x[1], reverse=True)
    top2 = [s[0] for s in scores[:2]]

    if is_high_uncertainty:
        logger.info(f"[Strategy] ⚠️ Alta Incerteza ({uncertainty_score:.2f}). Top2: {top2}")

    return top2
