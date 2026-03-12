# Objective: Test coverage for semantic cache behavior and regressions.
"""Test coverage for semantic cache behavior and regressions.

This test module verifies expected behavior, regression boundaries, and failure
handling for the corresponding runtime component.
"""


import pytest
import time
from unittest.mock import patch, AsyncMock
from app.semantic_cache import (
    check_cache,
    _compute_sha256,
    L1Cache,
    get_l1_cache_stats,
    _l1_cache,
)


class TestL1Cache:
    """Tests for the L1Cache class."""

    def test_l1_cache_store_and_get(self):
        """L1 cache should store and retrieve values."""
        cache = L1Cache(maxsize=10, ttl_seconds=60)

        cache.store("key1", {"value": "test"})
        result = cache.get("key1")

        assert result is not None
        assert result["value"] == "test"

    def test_l1_cache_returns_none_for_missing_key(self):
        """L1 cache should return None for missing keys."""
        cache = L1Cache(maxsize=10, ttl_seconds=60)

        result = cache.get("nonexistent_key")

        assert result is None

    def test_l1_cache_respects_ttl(self):
        """L1 cache should expire entries after TTL."""
        cache = L1Cache(maxsize=10, ttl_seconds=1)

        cache.store("key1", {"value": "test"})

        # Should be available immediately
        assert cache.get("key1") is not None

        # Wait for TTL to expire
        time.sleep(1.1)

        # Should be expired now
        assert cache.get("key1") is None

    def test_l1_cache_evicts_oldest_when_full(self):
        """L1 cache should evict oldest entries when maxsize exceeded."""
        cache = L1Cache(maxsize=3, ttl_seconds=60)

        cache.store("key1", "value1")
        cache.store("key2", "value2")
        cache.store("key3", "value3")
        cache.store("key4", "value4")  # Should evict key1

        assert cache.get("key1") is None  # Evicted
        assert cache.get("key2") is not None
        assert cache.get("key3") is not None
        assert cache.get("key4") is not None

    def test_l1_cache_updates_lru_on_access(self):
        """L1 cache should move accessed items to end (LRU)."""
        cache = L1Cache(maxsize=3, ttl_seconds=60)

        cache.store("key1", "value1")
        cache.store("key2", "value2")
        cache.store("key3", "value3")

        # Access key1 to make it recently used
        cache.get("key1")

        # Add new key - should evict key2 (oldest unused)
        cache.store("key4", "value4")

        assert cache.get("key1") is not None  # Still there (was accessed)
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") is not None
        assert cache.get("key4") is not None

    def test_l1_cache_stats_tracks_hits_and_misses(self):
        """L1 cache should track hits and misses correctly."""
        cache = L1Cache(maxsize=10, ttl_seconds=60)

        cache.store("key1", "value1")

        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("missing")  # Miss

        stats = cache.stats()

        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["size"] == 1

    def test_l1_cache_clear(self):
        """L1 cache clear should remove all entries."""
        cache = L1Cache(maxsize=10, ttl_seconds=60)

        cache.store("key1", "value1")
        cache.store("key2", "value2")

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.stats()["size"] == 0


class TestGlobalL1Cache:
    """Tests for the global L1 cache instance."""

    def test_global_l1_cache_exists(self):
        """Global L1 cache instance should exist."""
        assert _l1_cache is not None
        assert isinstance(_l1_cache, L1Cache)

    def test_get_l1_cache_stats_returns_dict(self):
        """get_l1_cache_stats should return statistics dictionary."""
        stats = get_l1_cache_stats()

        assert "hits" in stats
        assert "misses" in stats
        assert "size" in stats
        assert "maxsize" in stats


class TestHashConsistency:
    """Tests for hash functions."""

    def test_hash_consistency(self):
        """O hash deve ser determinístico."""
        t1 = "Como fazer bolo?"
        h1 = _compute_sha256(t1)
        h2 = _compute_sha256(t1)
        assert h1 == h2
        assert len(h1) == 64  # SHA256 hex


class TestCacheLogic:
    """Tests for cache hit/miss logic."""

    @pytest.mark.asyncio
    async def test_cache_hit_logic(self):
        """Simula um HIT no cache vetorial."""

        # Mock do retorno do ChromaDB
        mock_results = {
            "distances": [[0.05]],  # Similaridade = 1 - 0.05 = 0.95
            "documents": [["Resposta cacheada"]],
            "metadatas": [[{"model_used": "gpt-4", "answer_payload": "Resposta cacheada"}]]
        }

        with patch("app.semantic_cache._make_embedding", new_callable=AsyncMock) as mock_emb, \
             patch("app.semantic_cache.query_embedding", new_callable=AsyncMock) as mock_query, \
             patch("app.semantic_cache._l1_cache") as mock_l1:

            mock_emb.return_value = [0.1, 0.2, 0.3]
            mock_query.return_value = mock_results
            mock_l1.get.return_value = None  # L1 miss, force L2 lookup

            result = await check_cache("pergunta")

            assert result is not None
            assert result["text"] == "Resposta cacheada"
            assert result["similarity"] > 0.90

    @pytest.mark.asyncio
    async def test_cache_miss_logic(self):
        """Simula um MISS no cache (distância alta)."""

        mock_results = {
            "distances": [[0.5]],  # Similaridade = 0.5 (Baixa)
            "documents": [["Outra coisa"]],
            "metadatas": [[{}]]
        }

        with patch("app.semantic_cache._make_embedding", new_callable=AsyncMock) as mock_emb, \
             patch("app.semantic_cache.query_embedding", new_callable=AsyncMock) as mock_query, \
             patch("app.semantic_cache._l1_cache") as mock_l1:

            mock_emb.return_value = [0.1, 0.2, 0.3]
            mock_query.return_value = mock_results
            mock_l1.get.return_value = None

            result = await check_cache("pergunta")

            assert result is None

    @pytest.mark.asyncio
    async def test_l1_cache_hit_skips_chroma(self):
        """L1 cache hit should skip ChromaDB lookup."""

        cached_response = {
            "text": "L1 cached answer",
            "similarity": 1.0,
            "model_used": "gpt-4",
            "image_output_b64": None
        }

        with patch("app.semantic_cache._l1_cache") as mock_l1, \
             patch("app.semantic_cache._make_embedding", new_callable=AsyncMock) as mock_emb, \
             patch("app.semantic_cache.query_embedding", new_callable=AsyncMock) as mock_query:

            mock_l1.get.return_value = cached_response

            result = await check_cache("pergunta")

            assert result is not None
            assert result["text"] == "L1 cached answer"
            # ChromaDB should not be called
            mock_emb.assert_not_called()
            mock_query.assert_not_called()