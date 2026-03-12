# -*- coding: utf-8 -*-
# Objective: Test coverage for performance behavior and regressions.
"""
test_performance.py - Testes de Performance para Quick Wins
------------------------------------------------------------
Testes para validar as otimizações de performance implementadas:
1. Connection Pool Tuning
2. EMA History Cleanup (TTL/LRU)
3. Embedding Cache Layer (L1)
4. Judge Verdict Caching
5. Centroid Lookup Optimization
"""

import pytest
import time
import threading
import numpy as np
from unittest.mock import MagicMock, patch


class TestEMAHistoryCache:
    """Testes para o cache EMA com TTL e LRU."""

    def test_ema_cache_basic_operations(self):
        """Testa operações básicas de get/set no cache EMA."""
        # Import dentro do teste para usar os mocks
        from app.router_core import EMAHistoryCache

        cache = EMAHistoryCache(maxsize=10, ttl_s=60)

        # Set e Get
        key = ("text", "model_a")
        value = {"ema_latency": 1.5, "ema_quality": 8.0, "ema_cost": 0.001}

        cache.set(key, value)
        result = cache.get(key)

        assert result is not None
        assert result["ema_latency"] == 1.5
        assert result["ema_quality"] == 8.0

    def test_ema_cache_lru_eviction(self):
        """Testa evicção LRU quando cache está cheio."""
        from app.router_core import EMAHistoryCache

        cache = EMAHistoryCache(maxsize=3, ttl_s=60)

        # Adiciona 4 itens (um a mais que maxsize)
        for i in range(4):
            cache.set(("text", f"model_{i}"), {"value": i})

        # O primeiro item deve ter sido evicted
        assert cache.get(("text", "model_0")) is None
        # Os outros devem existir
        assert cache.get(("text", "model_1")) is not None
        assert cache.get(("text", "model_2")) is not None
        assert cache.get(("text", "model_3")) is not None

    def test_ema_cache_ttl_expiration(self):
        """Testa expiração por TTL."""
        from app.router_core import EMAHistoryCache

        cache = EMAHistoryCache(maxsize=10, ttl_s=1)  # TTL de 1 segundo

        key = ("text", "model_a")
        cache.set(key, {"value": 1})

        # Imediatamente deve existir
        assert cache.get(key) is not None

        # Após TTL, deve expirar
        time.sleep(1.1)
        assert cache.get(key) is None

    def test_ema_cache_size_tracking(self):
        """Testa tracking de tamanho do cache."""
        from app.router_core import EMAHistoryCache

        cache = EMAHistoryCache(maxsize=100, ttl_s=60)

        for i in range(50):
            cache.set(("text", f"model_{i}"), {"value": i})

        assert cache.size() == 50

    def test_ema_cache_cleanup_expired(self):
        """Testa cleanup de entradas expiradas."""
        from app.router_core import EMAHistoryCache

        cache = EMAHistoryCache(maxsize=100, ttl_s=1)

        for i in range(10):
            cache.set(("text", f"model_{i}"), {"value": i})

        assert cache.size() == 10

        time.sleep(1.1)
        removed = cache.cleanup_expired()

        assert removed == 10
        assert cache.size() == 0


class TestEmbeddingL1Cache:
    """Testes para o cache L1 de embeddings."""

    def test_embedding_cache_basic(self):
        """Testa operações básicas do cache de embeddings."""
        from app.embeddings import EmbeddingL1Cache

        cache = EmbeddingL1Cache(maxsize=100, ttl_s=60)

        key = "test_key_123"
        vec = [0.1, 0.2, 0.3, 0.4]

        cache.set(key, vec)
        result = cache.get(key)

        assert result is not None
        assert result == vec

    def test_embedding_cache_miss_tracking(self):
        """Testa tracking de hits e misses."""
        from app.embeddings import EmbeddingL1Cache

        cache = EmbeddingL1Cache(maxsize=100, ttl_s=60)

        # Miss
        cache.get("nonexistent")
        stats = cache.stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 0

        # Hit
        cache.set("key1", [1.0, 2.0])
        cache.get("key1")
        stats = cache.stats()
        assert stats["hits"] == 1

    def test_embedding_cache_ttl(self):
        """Testa TTL do cache de embeddings."""
        from app.embeddings import EmbeddingL1Cache

        cache = EmbeddingL1Cache(maxsize=100, ttl_s=1)

        cache.set("key1", [1.0, 2.0])
        assert cache.get("key1") is not None

        time.sleep(1.1)
        assert cache.get("key1") is None


class TestVerdictCache:
    """Testes para o cache de verdicts de juízes."""

    def test_verdict_cache_basic(self):
        """Testa operações básicas do cache de verdicts."""
        from app.judges import VerdictCache

        cache = VerdictCache(maxsize=100, ttl_s=60)

        query = "What is the capital of France?"
        answer = "The capital of France is Paris."
        score = 0.9

        cache.set(query, answer, score)
        result = cache.get(query, answer)

        assert result is not None
        assert result == score

    def test_verdict_cache_key_generation(self):
        """Testa que diferentes queries/answers geram chaves diferentes."""
        from app.judges import VerdictCache

        cache = VerdictCache(maxsize=100, ttl_s=60)

        query1 = "Question 1"
        answer1 = "Answer 1"
        query2 = "Question 2"
        answer2 = "Answer 2"

        cache.set(query1, answer1, 0.8)
        cache.set(query2, answer2, 0.6)

        assert cache.get(query1, answer1) == 0.8
        assert cache.get(query2, answer2) == 0.6
        assert cache.get(query1, answer2) is None  # Different combination

    def test_verdict_cache_stats(self):
        """Testa estatísticas do cache de verdicts."""
        from app.judges import VerdictCache

        cache = VerdictCache(maxsize=100, ttl_s=60)

        cache.set("q1", "a1", 0.9)
        cache.get("q1", "a1")  # hit
        cache.get("q2", "a2")  # miss

        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1


class TestCentroidMatrixCache:
    """Testes para o cache de matriz de centróides."""

    def test_centroid_matrix_basic(self):
        """Testa operações básicas do cache de matriz de centróides."""
        from app.bandits import CentroidMatrixCache

        cache = CentroidMatrixCache()

        # Cria centróides de teste
        cents = [
            {"id": 0, "vec": np.array([1.0, 0.0, 0.0], dtype=np.float32)},
            {"id": 1, "vec": np.array([0.0, 1.0, 0.0], dtype=np.float32)},
            {"id": 2, "vec": np.array([0.0, 0.0, 1.0], dtype=np.float32)},
        ]

        cache.update(cents)

        # Query vector mais próximo do centróide 0
        query = np.array([0.9, 0.1, 0.0], dtype=np.float32)
        idx, sim, cid = cache.nearest(query)

        assert idx == 0
        assert cid == 0
        assert sim >= 0.89  # Floating-point tolerance

    def test_centroid_matrix_nearest_accuracy(self):
        """Testa precisão do nearest neighbor."""
        from app.bandits import CentroidMatrixCache

        cache = CentroidMatrixCache()

        # Cria centróides ortogonais
        cents = [
            {"id": 0, "vec": np.array([1.0, 0.0], dtype=np.float32)},
            {"id": 1, "vec": np.array([0.0, 1.0], dtype=np.float32)},
        ]

        cache.update(cents)

        # Query exatamente no centróide 1
        query = np.array([0.0, 1.0], dtype=np.float32)
        idx, sim, cid = cache.nearest(query)

        assert idx == 1
        assert cid == 1
        assert abs(sim - 1.0) < 1e-6

    def test_centroid_matrix_staleness(self):
        """Testa detecção de cache stale."""
        from app.bandits import CentroidMatrixCache

        cache = CentroidMatrixCache()

        # Sem update, deve estar stale
        assert cache.is_stale(max_age_s=0.1)

        cents = [{"id": 0, "vec": np.array([1.0, 0.0], dtype=np.float32)}]
        cache.update(cents)

        # Imediatamente após update, não deve estar stale
        assert not cache.is_stale(max_age_s=60.0)


class TestConnectionPoolConfig:
    """Testes para verificar configuração do pool de conexões."""

    def test_pool_settings_applied(self):
        """Verifica que as configurações do pool estão corretas."""
        # Verifica os valores configurados no módulo de conexão.
        import app.db as db_module

        import inspect
        source = inspect.getsource(db_module)

        assert "pool_recycle=300" in source
        assert "pool_size=20" in source
        assert "max_overflow=10" in source


class TestPerformanceMetrics:
    """Testes para verificar que as métricas de performance existem."""

    def test_metrics_exist(self):
        """Verifica que as novas métricas estão definidas."""
        from app.observability import (
            JUDGE_CACHE_HITS,
            JUDGE_CACHE_MISSES,
            JUDGE_CACHE_HIT_RATE,
            EMBEDDING_CACHE_HITS,
            EMBEDDING_CACHE_MISSES,
            EMBEDDING_CACHE_HIT_RATE,
            CENTROID_LOOKUP_DURATION,
            EMA_HISTORY_SIZE,
            EMA_HISTORY_EVICTIONS,
        )

        # Verifica que são instâncias válidas de métricas
        assert JUDGE_CACHE_HITS is not None
        assert JUDGE_CACHE_MISSES is not None
        assert JUDGE_CACHE_HIT_RATE is not None
        assert EMBEDDING_CACHE_HITS is not None
        assert EMBEDDING_CACHE_MISSES is not None
        assert EMBEDDING_CACHE_HIT_RATE is not None
        assert CENTROID_LOOKUP_DURATION is not None
        assert EMA_HISTORY_SIZE is not None
        assert EMA_HISTORY_EVICTIONS is not None


class TestCacheIntegration:
    """Testes de integração para os caches."""

    def test_get_verdict_cache_stats_function(self):
        """Testa função de estatísticas do cache de verdicts."""
        from app.judges import get_verdict_cache_stats

        stats = get_verdict_cache_stats()

        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        assert "size" in stats

    def test_get_embedding_cache_stats_function(self):
        """Testa função de estatísticas do cache de embeddings."""
        from app.embeddings import get_embedding_cache_stats

        stats = get_embedding_cache_stats()

        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        assert "size" in stats


class TestThreadSafety:
    """Testes de thread safety para os caches."""

    def test_ema_cache_thread_safety(self):
        """Testa que o cache EMA é thread-safe."""
        from app.router_core import EMAHistoryCache

        cache = EMAHistoryCache(maxsize=1000, ttl_s=60)
        errors = []

        def writer(thread_id):
            """Execute the writer routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            try:
                for i in range(100):
                    cache.set(("text", f"model_{thread_id}_{i}"), {"value": i})
            except Exception as e:
                errors.append(e)

        def reader(thread_id):
            """Execute the reader routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            try:
                for i in range(100):
                    cache.get(("text", f"model_{thread_id}_{i}"))
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            t1 = threading.Thread(target=writer, args=(i,))
            t2 = threading.Thread(target=reader, args=(i,))
            threads.extend([t1, t2])

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"

    def test_embedding_cache_thread_safety(self):
        """Testa que o cache de embeddings é thread-safe."""
        from app.embeddings import EmbeddingL1Cache

        cache = EmbeddingL1Cache(maxsize=1000, ttl_s=60)
        errors = []

        def worker(thread_id):
            """Execute the worker routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            try:
                for i in range(100):
                    key = f"key_{thread_id}_{i}"
                    cache.set(key, [float(i)] * 768)
                    cache.get(key)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"
