# -*- coding: utf-8 -*-
"""
router_strategy.py (Com Lógica de Incerteza UQ)
-----------------------------------------------
Estratégia de seleção sensível ao risco.
"""

import logging
from typing import List, Dict, Any

# Importa helpers do novo bandits.py
from app.bandits import get_snapshot, sample_metrics_from_snapshot
from app.settings_dynamic import settings

logger = logging.getLogger(__name__)

SOTA_MARKERS = ["gpt-4", "opus", "sonnet", "gemini-1.5-pro"]

def _is_sota(model_name: str) -> bool:
    return any(m in model_name.lower() for m in SOTA_MARKERS)

def _is_local(model_name: str) -> bool:
    return "ollama" in model_name.lower()

def choose_top2_models(
    candidates: List[str],
    min_quality: float, # Mantido para compatibilidade, não usado diretamente
    query_text: str,
    modality: str = "text",
    uncertainty_score: float = 0.0, # <-- Input do UQ
) -> List[str]:
    
    snapshot = get_snapshot()
    # Amostra qualidade probabilística (Thompson Sampling)
    # Isso já dá variedade à escolha (Exploration)
    sampled_qs = sample_metrics_from_snapshot(snapshot)

    # Limiar dinâmico definido pelo NSGA-II
    uq_threshold = float(settings.get("UNCERTAINTY_THRESHOLD", 0.45))
    
    is_high_uncertainty = uncertainty_score > uq_threshold
    # Hard Filter: Se a modalidade é texto, expulsa modelos com 'vision', 'vl', 'llava' no nome
    if modality == "text":
        candidates = [
            m for m in candidates 
            if not any(tag in m.lower() for tag in ["vision", "vl", "llava", "moondream"])
        ]
    scores = []
    for model in candidates:
        # Base quality (0-10)
        q_val = sampled_qs.get(model, 5.0) # 5.0 = prior neutro

        # --- LÓGICA UQ (Safety Mode) ---
        risk_factor = 1.0
        
        if is_high_uncertainty:
            # Em terreno desconhecido:
            if _is_sota(model):
                risk_factor = 1.3 # Confia nos modelos fortes
            elif _is_local(model):
                risk_factor = 0.6 # Desconfia dos locais
        else:
            # Em terreno conhecido:
            if _is_local(model):
                risk_factor = 1.1 # Bônus de eficiência para locais

        # Penalidade de Custo (Hardcoded simples ou vindo de pesos globais)
        # Em tese real, esses pesos viriam do NSGA também
        cost_penalty = 0.1 if _is_local(model) else 0.8

        final_score = (q_val * risk_factor) - cost_penalty
        scores.append((model, final_score))

    # Ordena e pega top 2
    scores.sort(key=lambda x: x[1], reverse=True)
    top2 = [s[0] for s in scores[:2]]

    if is_high_uncertainty:
        logger.info(f"[Strategy] ⚠️ Alta Incerteza ({uncertainty_score:.2f}). Modo Segurança. Top2: {top2}")
    
    return top2