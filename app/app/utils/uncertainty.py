# -*- coding: utf-8 -*-
# Objective: Utility helpers for uncertainty.
"""
app/app/utils/uncertainty.py
------------------------------------------------------
Módulo de Quantificação de Incerteza Epistêmica (UQ).
Calcula quão "nova" ou "estranha" uma query é baseada na sua distância
semântica em relação aos clusters de conhecimento prévio (centróides)
armazenados pelo sistema Bandit.
"""

import json
import logging
from typing import Dict, List

import numpy as np
from app.embeddings import embed_text
from app.utils.redis_client import get_redis

logger = logging.getLogger(__name__)

# Chave Redis onde o bandits.py armazena os vetores de queries bem-sucedidas
R_CENTROIDS_KEY = "meta:bandit:centroids"

def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calcula similaridade de cosseno entre dois vetores numpy."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-9 or norm2 < 1e-9: # Evita divisão por zero
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))

def get_uncertainty_score(query_text: str, modality: str = "text") -> float:
    """
    Calcula o score de incerteza para uma query.
    Retorna: 0.0 (Máxima Confiança/Certeza) a 1.0 (Máxima Incerteza).
    """
    # Incerteza só faz sentido para texto ou multimodal com texto rico.
    # Se for só imagem ou texto muito curto, a comparação semântica é frágil.
    if modality == "vision" and len(query_text) < 5:
        # Assume incerteza média/alta para visão pura sem contexto
        return 0.7

    rds = get_redis()
    if not rds:
        logger.warning("[UQ] Sem conexão Redis. Assumindo incerteza máxima.")
        return 1.0

    try:
        # 1. Carrega Centróides de Conhecimento do Redis
        raw_data = rds.get(R_CENTROIDS_KEY)
        if not raw_data:
            # Cold Start: Se não há histórico, tudo é incerto.
            # logger.debug("[UQ] Cold start (sem centróides). Incerteza = 1.0")
            return 1.0

        centroids_data: List[Dict] = json.loads(raw_data)
        if not centroids_data:
             return 1.0

        # 2. Gera Embedding da Query Atual
        # Nota: Embeddings são cacheados internamente pelo lru_cache no embeddings.py,
        # então chamar aqui não deve gerar recomputação excessiva se o router já chamou.
        query_vec_list = embed_text(query_text)
        if not query_vec_list or all(v == 0 for v in query_vec_list):
             logger.warning("[UQ] Falha no embedding da query. Incerteza = 1.0")
             return 1.0

        q_np = np.array(query_vec_list, dtype=np.float32)

        # 3. Encontra a maior similaridade com o conhecimento existente
        max_similarity = -1.0

        for centroid in centroids_data:
            # Formato esperado do centróide: {"vec": [floats], "text": "..."}
            c_vec_list = centroid.get("vec")
            if not c_vec_list: continue

            c_np = np.array(c_vec_list, dtype=np.float32)
            sim = _cosine_similarity(q_np, c_np)

            if sim > max_similarity:
                max_similarity = sim

        # Clampa a similaridade entre 0 e 1 por segurança matemática
        max_similarity = max(0.0, min(1.0, max_similarity))

        # 4. A Incerteza é o inverso da Similaridade (Familiaridade)
        # Se similaridade é alta (1.0), incerteza é baixa (0.0).
        uncertainty = 1.0 - max_similarity

        # Logging para análise (pode ser ruidoso em produção)
        # logger.debug(f"[UQ] Query: '{query_text[:20]}...' | MaxSim: {max_similarity:.3f} | UQ: {uncertainty:.3f}")

        return uncertainty

    except Exception as e:
        logger.error(f"[UQ] Erro crítico no cálculo de incerteza: {e}", exc_info=True)
        # Em caso de erro, assume postura conservadora (incerteza média/alta)
        return 0.5
