"""
semantic_cache.py
----------------------------------------------------
Cache semântico híbrido: ChromaDB + Redis + Prometheus.
Evita reprocessar consultas semelhantes e mede eficiência do cache.

Atualizado para:
✅ Usar o registry global do módulo observability
✅ Operar de forma compatível com Prometheus multiprocess
✅ Manter logs consistentes e métricas confiáveis
"""

import time
import json
import logging
import numpy as np
from app.vectorstore import (
    get_or_create_collection_async,
    query_embedding,
    insert_embedding,
)
from app.embeddings import embed_text
from app.utils.redis_client import get_redis
from app.observability import registry, logger  # integração centralizada

from prometheus_client import Counter, Gauge

# ============================================================
# ⚙️ Configurações básicas
# ============================================================
COLLECTION_NAME = "semantic_cache"
SIM_THRESHOLD = 0.90
MAX_RESULTS = 1
CACHE_TTL = 60 * 60 * 24  # 24h

# ============================================================
# 📊 Métricas Prometheus (registradas no registry global)
# ============================================================
CACHE_HITS = Counter(
    "semantic_cache_hits_total",
    "Total de acertos no cache semântico.",
    registry=registry,
)
CACHE_MISSES = Counter(
    "semantic_cache_misses_total",
    "Total de falhas no cache semântico.",
    registry=registry,
)
CACHE_ENTRIES = Gauge(
    "semantic_cache_entries",
    "Número de respostas armazenadas no cache.",
    registry=registry,
)
CACHE_HIT_RATIO = Gauge(
    "semantic_cache_hit_ratio",
    "Taxa de acerto do cache semântico (0–1).",
    registry=registry,
)

# ============================================================
# 🧠 Utilitários
# ============================================================
def cosine_similarity(a, b):
    """Cálculo de similaridade coseno com tratamento seguro."""
    if a is None or b is None:
        return 0.0
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _update_hit_ratio():
    """Atualiza a métrica de taxa de acerto (hits / total)."""
    try:
        # Acessa valores via collectors em vez de atributos privados
        hits = CACHE_HITS._value.get() if hasattr(CACHE_HITS, "_value") else 0
        misses = CACHE_MISSES._value.get() if hasattr(CACHE_MISSES, "_value") else 0
        total = hits + misses
        ratio = hits / total if total > 0 else 0.0
        CACHE_HIT_RATIO.set(round(ratio, 3))
    except Exception as e:
        logger.warning(f"[semantic_cache] Falha ao atualizar hit ratio: {e}")

# ============================================================
# 🔍 Consulta ao cache (Redis → Chroma)
# ============================================================
async def check_cache(query: str) -> dict | None:
    """
    Verifica se existe uma resposta semanticamente semelhante.
    1️⃣ Checa Redis (cache rápido).
    2️⃣ Se não houver, consulta ChromaDB.
    Retorna um dicionário {"text": ..., "similarity": ...} ou None.
    """
    try:
        if not query or not isinstance(query, str):
            logger.warning("[semantic_cache] Consulta inválida (query vazia ou None).")
            CACHE_MISSES.inc()
            _update_hit_ratio()
            return None

        redis_client = get_redis()
        redis_key = f"semantic:{hash(query)}"

        # 1️⃣ Verifica cache direto no Redis
        if redis_client:
            try:
                cached_data = redis_client.get(redis_key)
                if cached_data:
                    cached = json.loads(cached_data)
                    if isinstance(cached, dict) and "text" in cached:
                        CACHE_HITS.inc()
                        _update_hit_ratio()
                        logger.info(f"[semantic_cache] Redis HIT: {redis_key}")
                        return cached
            except Exception as e:
                logger.warning(f"[semantic_cache] Falha ao ler Redis: {e}")

        # 2️⃣ Fallback: consulta ChromaDB
        emb = await embed_text(query)
        if emb is None:
            logger.warning("[semantic_cache] Falha ao gerar embedding — abortando consulta.")
            CACHE_MISSES.inc()
            _update_hit_ratio()
            return None

        results = await query_embedding(COLLECTION_NAME, emb, n_results=MAX_RESULTS)
        if not results:
            CACHE_MISSES.inc()
            _update_hit_ratio()
            logger.info("[semantic_cache] MISS — Chroma retornou None.")
            return None

        docs = results.get("documents")
        embs = results.get("embeddings")
        if not docs or not isinstance(docs, list) or len(docs) == 0:
            CACHE_MISSES.inc()
            _update_hit_ratio()
            logger.info("[semantic_cache] MISS — Nenhum documento encontrado.")
            return None

        try:
            doc_text = docs[0][0] if isinstance(docs[0], list) else docs[0]
        except Exception:
            logger.warning("[semantic_cache] Estrutura de documentos inesperada.")
            doc_text = None

        # Cálculo da similaridade
        try:
            if embs and embs[0]:
                score = cosine_similarity(emb, embs[0][0])
            else:
                score = 0.0
        except Exception as e:
            logger.warning(f"[semantic_cache] Falha ao calcular similaridade: {e}")
            score = 0.0

        if not doc_text:
            CACHE_MISSES.inc()
            _update_hit_ratio()
            return None

        if score >= SIM_THRESHOLD:
            result = {"text": doc_text, "similarity": score}
            CACHE_HITS.inc()
            _update_hit_ratio()
            logger.info(f"[semantic_cache] Chroma HIT (sim={score:.2f})")

            # Armazena em Redis para acelerar próximos acessos
            if redis_client:
                try:
                    redis_client.setex(redis_key, CACHE_TTL, json.dumps(result))
                except Exception as e:
                    logger.warning(f"[semantic_cache] Falha ao salvar Redis HIT: {e}")
            return result

        CACHE_MISSES.inc()
        _update_hit_ratio()
        logger.info(f"[semantic_cache] MISS (sim={score:.2f})")
        return None

    except Exception as e:
        logger.error(f"[semantic_cache] Erro ao consultar cache: {e}")
        CACHE_MISSES.inc()
        _update_hit_ratio()
        return None


# ============================================================
# 💾 Armazenamento (Chroma + Redis)
# ============================================================
async def store_cache(query: str, answer: str):
    """Armazena uma nova entrada no cache (Chroma + Redis)."""
    try:
        if not query or not answer:
            logger.warning("[semantic_cache] store_cache chamado com dados vazios.")
            return

        emb = await embed_text(query)
        if emb is None:
            logger.warning("[semantic_cache] Falha ao gerar embedding — cache não salvo.")
            return

        await get_or_create_collection_async(COLLECTION_NAME)
        await insert_embedding(COLLECTION_NAME, str(time.time()), answer, emb)

        redis_client = get_redis()
        redis_key = f"semantic:{hash(query)}"
        payload = json.dumps({"text": answer, "similarity": 1.0})

        if redis_client:
            try:
                redis_client.setex(redis_key, CACHE_TTL, payload)
            except Exception as e:
                logger.warning(f"[semantic_cache] Falha ao salvar no Redis: {e}")

        CACHE_ENTRIES.inc()
        logger.info(f"[semantic_cache] Resposta armazenada no cache: {redis_key}")

    except Exception as e:
        logger.error(f"[semantic_cache] Falha ao salvar no cache: {e}")
