# -*- coding: utf-8 -*-
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
from typing import List, Dict, Any

# Importa helpers do bandits.py
from app.bandits import get_snapshot, sample_metrics_from_snapshot
from app.settings_dynamic import settings
from app.reliability import get_circuit_breaker_manager, get_cascade_detector

logger = logging.getLogger(__name__)

SOTA_MARKERS = ["gpt-5", "opus", "sonnet", "gemini-3-pro"]

def _is_sota(model_name: str) -> bool:
    """Executa is sota."""
    return any(m in model_name.lower() for m in SOTA_MARKERS)

def _is_local(model_name: str) -> bool:
    """Executa is local."""
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
    """Executa choose top2 models."""
    cascade_detector = get_cascade_detector()
    if cascade_detector.is_emergency_mode:
        emergency_model = cascade_detector.get_emergency_fallback()
        if emergency_model:
            logger.warning(f"[Strategy] 🚨 EMERGENCY MODE: Using fallback {emergency_model}")
            return [emergency_model]

    snapshot = get_snapshot()
    # Amostra qualidade probabilística (Thompson Sampling)
    sampled_qs = sample_metrics_from_snapshot(snapshot)

    # Limiar dinâmico definido pelo NSGA-II
    uq_threshold = float(settings.get("UNCERTAINTY_THRESHOLD", 0.45))

    is_high_uncertainty = uncertainty_score > uq_threshold

    # ==================================================================
    # 🔒 CIRCUIT BREAKER FILTER - Remove models with open circuit breakers
    # ==================================================================
    breaker_manager = get_circuit_breaker_manager()
    available_candidates = [m for m in candidates if breaker_manager.is_available(m)]

    if not available_candidates:
        logger.warning("[Strategy] ⚠️ All models have open circuit breakers! Using original candidates.")
        available_candidates = candidates
    elif len(available_candidates) < len(candidates):
        excluded = len(candidates) - len(available_candidates)
        logger.info(f"[Strategy] Circuit breaker filtered {excluded} model(s)")

    candidates = available_candidates

    # ==================================================================
    # 🔒 HARD FILTERS (SEGURANÇA DE TIPO)
    # ==================================================================

    # 1. SE FOR VISÃO: Exige capacidade multimodal
    if modality in ("vision", "multimodal"):
        # Lista de termos que identificam modelos que "enxergam"
        vision_markers = ["vision", "vl", "llava", "moondream", "gpt-4o", "gemini", "claude"]

        # Filtra: Mantém apenas se tiver algum marcador visual no nome
        vision_candidates = [
            m for m in candidates
            if any(tag in m.lower() for tag in vision_markers)
        ]

        if vision_candidates:
            candidates = vision_candidates
        else:
            # Se o filtro removeu tudo (ex: lista mal configurada), força um fallback seguro
            logger.warning("[Strategy] ⚠️ NENHUM modelo de visão encontrado na lista! Usando fallback Qwen3-VL.")
            candidates = ["ollama/qwen3-vl:4b"]

    # 2. SE FOR TEXTO: Remove modelos de visão (para economizar VRAM/Custo)
    elif modality == "text":
        # Remove modelos que tenham tags de visão explícitas,
        # a menos que sejam modelos "Omni" (GPT-4o) que são ótimos em texto também.
        candidates = [
            m for m in candidates
            if not any(tag in m.lower() for tag in ["vision", "vl", "llava", "moondream"])
        ]
        # Se sobrar vazio, fallback para texto
        if not candidates:
            candidates = ["ollama/gemma3:4b"]

    # ==================================================================

    scores = []

    # Define pesos padrão caso o NSGA-II ainda não tenha rodado ou retornado vazio
    w_q = weights.get("w_quality", 1.0)
    w_l = weights.get("w_latency", 0.5)
    w_c = weights.get("w_cost", 50.0)

    # Get dynamic risk factors from settings
    risk_sota_high_uq = settings.RISK_FACTOR_SOTA_HIGH_UQ
    risk_local_high_uq = settings.RISK_FACTOR_LOCAL_HIGH_UQ
    risk_local_low_uq = settings.RISK_FACTOR_LOCAL_LOW_UQ

    for model in candidates:
        # Chave para buscar stats no snapshot
        key_options = [f"{model}::{modality}", model]
        stats = {}
        for k in key_options:
            if k in snapshot:
                stats = snapshot[k]
                break

        # Base quality (0-10) via Thompson Sampling
        q_base = sampled_qs.get(model, 5.0)

        # --- LÓGICA UQ (Safety Mode) with Dynamic Risk Factors ---
        risk_factor = 1.0

        if is_high_uncertainty:
            # Em terreno desconhecido:
            if _is_sota(model):
                risk_factor = risk_sota_high_uq  # Confia nos modelos fortes
            elif _is_local(model):
                risk_factor = risk_local_high_uq  # Desconfia dos locais
        else:
            # Em terreno conhecido:
            if _is_local(model):
                risk_factor = risk_local_low_uq  # Bônus de eficiência para locais

        # --- CIRCUIT BREAKER PENALTY ---
        cb_penalty = _get_circuit_breaker_penalty(model)
        risk_factor *= cb_penalty

        # Qualidade Ajustada pelo Risco
        q_final = q_base * risk_factor

        # Heurística de Latência (se não tiver dados reais)
        avg_latency = stats.get("avg_latency", 2.0)
        if _is_local(model) and avg_latency == 2.0:
            avg_latency = 0.5

        # Heurística de Custo (se não tiver dados reais)
        est_cost = stats.get("avg_cost", 0.000001)
        if _is_sota(model) and est_cost < 0.001:
            est_cost = 0.01

        # --- FÓRMULA DE SCORE (NSGA-II) ---
        # Score = (Qualidade * wQ) - (Latencia * wL) - (Custo * wC)
        score = (q_final * w_q) - (avg_latency * w_l) - (est_cost * w_c)

        scores.append((model, score))

    # Ordena e pega top 2
    scores.sort(key=lambda x: x[1], reverse=True)
    top2 = [s[0] for s in scores[:2]]

    if is_high_uncertainty:
        logger.info(f"[Strategy] ⚠️ Alta Incerteza ({uncertainty_score:.2f}). Top2: {top2}")

    return top2
