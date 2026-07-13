# -*- coding: utf-8 -*-
# Objective: Test coverage for embeddings behavior and regressions.
"""
test_embeddings.py — Tests for Embeddings Module
--------------------------------------------------
Tests for the embeddings.py utility module covering:
- Text embedding generation
- L1 and L2 cache functionality
- Fallback behavior
- Cache statistics
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestEmbeddingL1Cache:
    """Tests for the L1 in-memory embedding cache."""

    def test_cache_initialization(self):
        """Test that cache initializes with correct defaults."""
        from app.embeddings import EmbeddingL1Cache

        cache = EmbeddingL1Cache(maxsize=100, ttl_s=60)

        assert cache.maxsize == 100
        assert cache.ttl_s == 60

    def test_cache_set_and_get(self):
        """Test basic set and get operations."""
        from app.embeddings import EmbeddingL1Cache

        cache = EmbeddingL1Cache(maxsize=100, ttl_s=300)

        test_vec = [0.1, 0.2, 0.3]
        cache.set("test_key", test_vec)

        result = cache.get("test_key")

        assert result == test_vec

    def test_cache_miss_returns_none(self):
        """Test that cache miss returns None."""
        from app.embeddings import EmbeddingL1Cache

        cache = EmbeddingL1Cache(maxsize=100, ttl_s=300)

        result = cache.get("nonexistent_key")

        assert result is None

    def test_cache_eviction_on_maxsize(self):
        """Test that oldest entries are evicted when maxsize is reached."""
        from app.embeddings import EmbeddingL1Cache

        cache = EmbeddingL1Cache(maxsize=3, ttl_s=300)

        cache.set("key1", [1.0])
        cache.set("key2", [2.0])
        cache.set("key3", [3.0])
        cache.set("key4", [4.0])  # Should evict key1

        assert cache.get("key1") is None  # Evicted
        assert cache.get("key4") == [4.0]  # Present

    def test_cache_stats(self):
        """Test that cache statistics are tracked correctly."""
        from app.embeddings import EmbeddingL1Cache

        cache = EmbeddingL1Cache(maxsize=100, ttl_s=300)

        cache.set("key1", [1.0])
        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("missing")  # Miss

        stats = cache.stats()

        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["size"] == 1

    def test_cache_lru_behavior(self):
        """Test LRU behavior - recently used items are kept."""
        from app.embeddings import EmbeddingL1Cache

        cache = EmbeddingL1Cache(maxsize=3, ttl_s=300)

        cache.set("key1", [1.0])
        cache.set("key2", [2.0])
        cache.set("key3", [3.0])

        # Access key1 to make it recently used
        cache.get("key1")

        # Add key4, should evict key2 (least recently used)
        cache.set("key4", [4.0])

        assert cache.get("key1") is not None  # Still present (was accessed)
        assert cache.get("key2") is None  # Evicted


class TestHashText:
    """Tests for the text hashing function."""

    def test_hash_text_deterministic(self):
        """Test that same input produces same hash."""
        from app.embeddings import _hash_text

        text = "Hello world"
        model = "test-model"

        hash1 = _hash_text(text, model)
        hash2 = _hash_text(text, model)

        assert hash1 == hash2

    def test_hash_text_different_texts(self):
        """Test that different texts produce different hashes."""
        from app.embeddings import _hash_text

        model = "test-model"

        hash1 = _hash_text("Hello", model)
        hash2 = _hash_text("World", model)

        assert hash1 != hash2

    def test_hash_text_different_models(self):
        """Test that same text with different models produces different hashes."""
        from app.embeddings import _hash_text

        text = "Hello"

        hash1 = _hash_text(text, "model-a")
        hash2 = _hash_text(text, "model-b")

        assert hash1 != hash2


class TestNorm:
    """Tests for vector normalization."""

    def test_norm_normalizes_vector(self):
        """Test that _norm normalizes a vector to unit length."""
        from app.embeddings import _norm

        vec = np.array([3.0, 4.0])  # Magnitude = 5
        result = _norm(vec)

        assert np.isclose(np.linalg.norm(result), 1.0)

    def test_norm_zero_vector(self):
        """Test that _norm handles zero vectors."""
        from app.embeddings import _norm

        vec = np.array([0.0, 0.0])
        result = _norm(vec)

        # Should return the zero vector unchanged
        assert np.array_equal(result, vec)


class TestEmbedText:
    """Tests for the main embed_text function."""

    def test_embed_text_empty_string(self):
        """Test that empty string returns minimal vector."""
        from app.embeddings import embed_text

        result = embed_text("")

        assert result == [0.0]

    def test_embed_text_none(self):
        """Test that None returns minimal vector."""
        from app.embeddings import embed_text

        result = embed_text(None)

        assert result == [0.0]

    def test_embed_text_whitespace_only(self):
        """Test that whitespace-only string returns minimal vector."""
        from app.embeddings import embed_text

        result = embed_text("   ")

        assert result == [0.0]

    @patch('app.embeddings._local_cpu_embed')
    def test_embed_text_returns_list(self, mock_embed):
        """Test that embed_text returns a list of floats."""
        mock_embed.return_value = [0.1, 0.2, 0.3]

        from app.embeddings import _embed_l1_cache, embed_text

        # Clear cache to ensure we hit the mock
        _embed_l1_cache._data.clear()

        result = embed_text("test text")

        assert isinstance(result, list)
        assert all(isinstance(x, (int, float)) for x in result)

    @patch("app.embeddings._load_cache", return_value=None)
    @patch("app.embeddings._save_cache")
    @patch("app.embeddings._local_cpu_embed")
    def test_embed_text_caches_result(self, mock_embed, _mock_save_cache, _mock_load_cache):
        """Test that results are cached."""
        mock_embed.return_value = [0.1, 0.2, 0.3]

        from app.embeddings import _embed_l1_cache, embed_text

        # Clear cache
        _embed_l1_cache._data.clear()

        # First call
        result1 = embed_text("test text for caching")

        # Second call should use cache
        result2 = embed_text("test text for caching")

        # Mock should only be called once
        assert mock_embed.call_count == 1
        assert result1 == result2


class TestEmbedImage:
    """Tests for image embedding function."""

    def test_embed_image_returns_minimal_vector(self):
        """Test that embed_image returns a minimal vector (feature disabled)."""
        from app.embeddings import embed_image

        result = embed_image("fake_base64_data")

        assert result == [0.0]


class TestEmbedMultimodal:
    """Tests for multimodal embedding function."""

    @patch('app.embeddings.embed_text')
    def test_embed_multimodal_returns_dict(self, mock_embed_text):
        """Test that embed_multimodal returns a dictionary."""
        mock_embed_text.return_value = [0.1, 0.2, 0.3]

        from app.embeddings import embed_multimodal

        result = embed_multimodal("test text", "fake_image_b64")

        assert isinstance(result, dict)
        assert "text" in result
        assert "vision" in result
        assert "multimodal" in result

    @patch('app.embeddings.embed_text')
    def test_embed_multimodal_text_only(self, mock_embed_text):
        """Test embed_multimodal with text only."""
        mock_embed_text.return_value = [0.1, 0.2, 0.3]

        from app.embeddings import embed_multimodal

        result = embed_multimodal("test text", None)

        assert result["text"] == [0.1, 0.2, 0.3]


class TestGetEmbeddingCacheStats:
    """Tests for cache statistics function."""

    def test_get_embedding_cache_stats_returns_dict(self):
        """Test that get_embedding_cache_stats returns a dictionary."""
        from app.embeddings import get_embedding_cache_stats

        stats = get_embedding_cache_stats()

        assert isinstance(stats, dict)
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        assert "size" in stats


class TestLocalCpuEmbed:
    """Tests for local CPU embedding generation."""

    @patch('app.embeddings.get_local_model')
    def test_local_cpu_embed_adds_prefix_for_nomic(self, mock_get_model):
        """Test that Nomic models get search_query prefix."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2])
        mock_get_model.return_value = mock_model

        from app.embeddings import _local_cpu_embed

        # This assumes EMBED_MODEL_TEXT contains "nomic"
        with patch('app.embeddings.EMBED_MODEL_TEXT', 'nomic-ai/nomic-embed-text-v1.5'):
            _local_cpu_embed("test query")

        # Verify the prefix was added
        call_args = mock_model.encode.call_args
        assert call_args[0][0].startswith("search_query:")

    @patch('app.embeddings.get_local_model')
    def test_local_cpu_embed_returns_list(self, mock_get_model):
        """Test that _local_cpu_embed returns a list."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_get_model.return_value = mock_model

        from app.embeddings import _local_cpu_embed

        result = _local_cpu_embed("test text")

        assert isinstance(result, list)

    @patch('app.embeddings.get_local_model')
    def test_local_cpu_embed_raises_when_no_model(self, mock_get_model):
        """Test that _local_cpu_embed raises when model unavailable."""
        mock_get_model.return_value = None

        from app.embeddings import _local_cpu_embed

        with pytest.raises(RuntimeError):
            _local_cpu_embed("test text")
