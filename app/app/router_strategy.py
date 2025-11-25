# -*- coding: utf-8 -*-
"""
router_strategy.py (multimodal-aware)
----------------------------------------------------
Estratégias de seleção de modelos para o Router Multimodal:
- Seleção por modalidade (text/vision/multimodal)
- Snapshot multimodal-aware: chave (modelo, modalidade)
- Rankeamento híbrido: qualidade ajustada × latência × custo
"""

import logging
from typing import List, Dict, Tuple

from .settings_dynamic import settings
from .metrics_collector import get_snapshot, update_model_metrics
from .observability import ROUTER_CHOSEN

logger = logging.getLogger(__name__)


# ============================================================
# 🔥 Utilitários
# ============================================================

def _get_candidate_sets() -> Dict[str, List[str]]:
    """Retorna listas de modelos por modalidade."""
    return {
        "text": getattr(settings, "CANDIDATE_MODELS_LIST", []),
        "vision": getattr(settings, "CANDIDATE_VISION_MODELS_LIST", []),
        "multimodal": getattr(settings, "CANDIDATE_MULTIMODAL_MODELS_LIST", []),
    }


def _pick_models_by_modality(request_modality: str) -> List[str]:
    """
    Seleciona subconjuntos coerentes com a modalidade.
    """
    msets = _get_candidate_sets()

    if request_modality == "vision":
        return msets["vision"] or msets["multimodal"] or msets["text"]

    if request_modality == "multimodal":
        return msets["multimodal"] + msets["vision"] + msets["text"]

    # texto
    return msets["text"] or msets["multimodal"] or msets["vision"]


# ============================================================
# 🎯 Escolha inteligente top-2 (multimodal-aware)
# ============================================================

def choose_top2_models(
    candidates: List[str],
    min_quality: float,
    query_text: str,
    modality: str = "text",
) -> List[str]:
    """
    Seleciona os 2 melhores modelos com base em métricas dinâmicas
    e compatibilidade multimodal.
    """
    snapshot = get_snapshot() or {}

    # subconjunto coerente com modalidade
    modal_candidates = _pick_models_by_modality(modality)
    filtered = [c for c in candidates if c in modal_candidates]

    if not filtered:
        logger.warning("[router_strategy] Nenhum modelo multimodal compatível — fallback para lista original.")
        filtered = candidates

    results = []

    for model in filtered:
        # chave multimodal
        key = (model, modality)

        # snapshot multimodal-aware
        m = snapshot.get(key)

        if not m:
            # fallback para quando modelo ainda não tem histórico nesta modalidade
            m = {"quality": 5.0, "latency": 1.8, "cost": 0.25}

        # Peso de ajuste dependente do tipo do modelo
        lower = model.lower()
        if modality == "vision":
            modality_weight = 1.4 if ("vision" in lower or "vl" in lower) else 0.6
        elif modality == "multimodal":
            # vlms tendem a performar melhor aqui
            modality_weight = 1.25 if ("vision" in lower or "vl" in lower) else 1.0
        else:  # text
            modality_weight = 1.15 if ("text" in lower or "llama" in lower or "phi" in lower) else 0.75

        adj_quality = m["quality"] * modality_weight

        results.append((model, adj_quality, m["latency"], m["cost"]))

    # Ordenação: maior qualidade ajustada, menor latência, menor custo
    ranked = sorted(results, key=lambda r: (-r[1], r[2], r[3]))

    top = [r[0] for r in ranked[:2]]

    logger.info(
        f"[router_strategy] top2={top} (modality={modality}, query='{query_text[:40]}...')"
    )

    return top


# ============================================================
# 📈 Atualização de métricas por modalidade
# ============================================================

def update_metrics(
    model_name: str,
    cost: float,
    latency: float,
    quality: float,
    modality: str = "text",
    **kwargs
):
    """
    Atualiza métricas multimodais no collector.
    kwargs pode incluir:
        - cost_per_1k
        - tokens_in
        - tokens_out
        - embedding_dim
        - vision_usage
    """
    update_model_metrics(
        model_name=model_name,
        latency=latency,
        quality=quality,
        cost=cost,
        modality=modality,
        **kwargs
    )
