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

import logging

import numpy as np

from app.embeddings import embed_text
from app.services.bandit_centroids import load_centroid_matrix, normalize_centroid_vec
from app.utils.redis_client import get_redis

logger = logging.getLogger(__name__)

# Chave Redis onde o bandits.py armazena os vetores de queries bem-sucedidas
R_CENTROIDS_KEY = "meta:bandit:centroids"
# Hash de metadados (revisão) que o bandits.py atualiza a cada gravação dos centróides
R_CENTROIDS_META_KEY = "meta:bandit:centroids:meta"

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

    u(q) = 1 - max_j cos(v_q, c_j) sobre os centróides do bandit. A matriz de
    centróides (normalizada, float32) vem de ``load_centroid_matrix``, que só
    relê o JSON do Redis quando a revisão dos centróides muda; a similaridade
    é um único produto matriz-vetor. Função síncrona: o roteador a executa em
    thread (``asyncio.to_thread``) para não bloquear o event loop.
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
        # 1. Centróides de conhecimento (cold start: sem histórico, tudo é incerto)
        centroids = load_centroid_matrix(rds, R_CENTROIDS_KEY, R_CENTROIDS_META_KEY)
        if centroids is None:
            return 1.0

        # 2. Embedding da query (cache L1/L2 em embeddings.py; o cache semântico já o calculou)
        query_vec_list = embed_text(query_text)
        if not query_vec_list or all(v == 0 for v in query_vec_list):
            logger.warning("[UQ] Falha no embedding da query. Incerteza = 1.0")
            return 1.0
        q_np = normalize_centroid_vec(np.asarray(query_vec_list, dtype=np.float32), centroids.matrix.shape[1])

        # 3. Maior similaridade de cosseno (linhas e query unitárias), limitada a [0, 1]
        max_similarity = max(0.0, min(1.0, float(np.max(centroids.matrix @ q_np))))

        # 4. A Incerteza é o inverso da Similaridade (Familiaridade)
        return 1.0 - max_similarity

    except Exception as e:
        logger.error(f"[UQ] Erro crítico no cálculo de incerteza: {e}", exc_info=True)
        # Em caso de erro, assume postura conservadora (incerteza média/alta)
        return 0.5
