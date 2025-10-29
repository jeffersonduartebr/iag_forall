"""
semantic_cache.py
-----------------
Cache semântico híbrido: ChromaDB + Redis + Prometheus
Evita reprocessar consultas semelhantes e mede eficiência do cache.
"""

import time
import json
import logging
import numpy as np
from prometheus_client import Counter, Gauge
from app.vectorstore import get_or_create_collection, query_embedding, insert_embedding
from app.embeddings import embed_text
from app.utils.redis_client import get_redis

logger = logging.getLogger(__name__)

# ============================================================
# ⚙️ Configurações básicas
# ============================================================
COLLECTION_NAME = "semantic_cache"
SIM_THRESHOLD = 0.90
MAX_RESULTS = 1
CACHE_TTL = 60 * 60 * 24  # 24h

# ============================================================
# 📊 Métricas Prometheus
# ============================================================
CACHE_HITS = Counter("semantic_cache_hits_total", "Total de acertos no cache semântico.")
CACHE_MISSES = Counter("semantic_cache_misses_total", "Total de falhas no cache semântico.")
CACHE_ENTRIES = Gauge("semantic_cache_entries", "Número de respostas armazenadas no cache.")

# ============================================================
# 🧠 Utilitários
# ============================================================
def cosine_similarity(a, b):
    """Cálculo de similaridade coseno."""
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ============================================================
# 🔍 Consulta ao cache (Redis → Chroma)
# ============================================================
def check_cache(query: str) -> dict | None:
    """
    Verifica se existe uma resposta semanticamente semelhante.
    1️⃣ Checa Redis (cache rápido).
    2️⃣ Se não houver, consulta ChromaDB.
    """
    try:
        redis_client = get_redis()
        redis_key = f"semantic:{hash(query)}"

        # 1️⃣ Verifica cache direto no Redis
        if redis_client and redis_client.exists(redis_key):
            cached = json.loads(redis_client.get(redis_key))
            logger.info(f"[semantic_cache] Redis HIT: {redis_key}")
            CACHE_HITS.inc()
            return cached

        # 2️⃣ Fallback: consulta ChromaDB
        emb = embed_text(query)
        results = query_embedding(COLLECTION_NAME, emb, n_results=MAX_RESULTS)
        if not results or "embeddings" not in results or not results["documents"]:
            CACHE_MISSES.inc()
            return None

        doc = results["documents"][0][0]
        score = cosine_similarity(emb, results["embeddings"][0][0])

        if score >= SIM_THRESHOLD:
            result = {"text": doc, "similarity": score}
            CACHE_HITS.inc()
            logger.info(f"[semantic_cache] Chroma HIT (sim={score:.2f})")

            # Armazena em Redis para acelerar próximos acessos
            if redis_client:
                redis_client.setex(redis_key, CACHE_TTL, json.dumps(result))
            return result

        CACHE_MISSES.inc()
        logger.info(f"[semantic_cache] MISS (sim={score:.2f})")
        return None

    except Exception as e:
        logger.error(f"[semantic_cache] Erro ao consultar cache: {e}")
        CACHE_MISSES.inc()
        return None


# ============================================================
# 💾 Armazenamento (Chroma + Redis)
# ============================================================
def store_cache(query: str, answer: str):
    """Armazena uma nova entrada no cache."""
    try:
        emb = embed_text(query)
        get_or_create_collection(COLLECTION_NAME)
        insert_embedding(COLLECTION_NAME, str(time.time()), answer, emb)

        redis_client = get_redis()
        redis_key = f"semantic:{hash(query)}"
        payload = json.dumps({"text": answer, "similarity": 1.0})

        if redis_client:
            redis_client.setex(redis_key, CACHE_TTL, payload)

        CACHE_ENTRIES.inc()
        logger.info(f"[semantic_cache] Resposta armazenada no cache: {redis_key}")

    except Exception as e:
        logger.error(f"[semantic_cache] Falha ao salvar no cache: {e}")
