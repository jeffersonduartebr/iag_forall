# -*- coding: utf-8 -*-
# Objective: Test coverage for token utils behavior and regressions.
"""
test_token_utils.py — Tests for Token Utilities
------------------------------------------------
Tests for the token counting functionality with LRU cache.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestCountTokens:
    """Tests for the count_tokens function."""

    def test_count_tokens_empty_string(self):
        """Test that empty string returns 0 tokens."""
        from app.utils.token_utils import count_tokens

        result = count_tokens("")
        assert result == 0

    def test_count_tokens_none_returns_zero(self):
        """Test that None input returns 0 tokens."""
        from app.utils.token_utils import count_tokens

        result = count_tokens(None)
        assert result == 0

    def test_count_tokens_heuristic_for_non_openai(self):
        """Test heuristic token counting for non-OpenAI models."""
        from app.utils.token_utils import count_tokens

        text = "Hello world this is a test"  # 26 chars
        result = count_tokens(text, model_name="ollama/llama3")

        # Heuristic: len(text) // 3
        expected = max(1, len(text) // 3)
        assert result == expected

    def test_count_tokens_heuristic_for_gemma(self):
        """Test heuristic for Gemma model."""
        from app.utils.token_utils import count_tokens

        text = "This is a test sentence for Gemma"
        result = count_tokens(text, model_name="gemma3:4b")

        # Should use heuristic (not tiktoken)
        assert result == max(1, len(text) // 3)

    def test_count_tokens_uses_tiktoken_for_gpt(self):
        """Test that tiktoken is used for GPT models."""
        from app.utils.token_utils import count_tokens

        text = "Hello world"
        result = count_tokens(text, model_name="gpt-4o")

        # Should return a reasonable token count
        assert result > 0
        assert result < len(text)  # Tokens should be fewer than chars

    def test_count_tokens_uses_tiktoken_for_o1(self):
        """Test that tiktoken is used for o1 models."""
        from app.utils.token_utils import count_tokens

        text = "This is a test"
        result = count_tokens(text, model_name="o1-preview")

        assert result > 0

    def test_count_tokens_minimum_one(self):
        """Test that count_tokens returns at least 1 for non-empty text."""
        from app.utils.token_utils import count_tokens

        text = "a"  # Very short text
        result = count_tokens(text, model_name="ollama/test")

        assert result >= 1


class TestEncoderCache:
    """Tests for the encoder LRU cache."""

    def test_get_encoder_cache_info(self):
        """Test that cache info is available."""
        from app.utils.token_utils import get_encoder_cache_info

        info = get_encoder_cache_info()

        assert isinstance(info, dict)
        assert "hits" in info
        assert "misses" in info
        assert "maxsize" in info
        assert "currsize" in info

    def test_clear_encoder_cache(self):
        """Test that cache can be cleared."""
        from app.utils.token_utils import (
            count_tokens,
            clear_encoder_cache,
            get_encoder_cache_info,
        )

        # Use the function to populate cache
        count_tokens("test", model_name="gpt-4o")

        # Clear cache
        clear_encoder_cache()

        # Verify cache is empty
        info = get_encoder_cache_info()
        assert info["currsize"] == 0

    def test_cache_hit_after_repeated_calls(self):
        """Test that repeated calls result in cache hits."""
        from app.utils.token_utils import (
            count_tokens,
            clear_encoder_cache,
            get_encoder_cache_info,
        )

        clear_encoder_cache()

        # First call - cache miss
        count_tokens("test", model_name="gpt-4o")
        info_after_first = get_encoder_cache_info()

        # Second call - should be cache hit
        count_tokens("another test", model_name="gpt-4o")
        info_after_second = get_encoder_cache_info()

        # Hits should have increased
        assert info_after_second["hits"] >= info_after_first["hits"]

    def test_cache_respects_maxsize(self):
        """Test that cache respects the maximum size limit."""
        from app.utils.token_utils import get_encoder_cache_info
        from app.config.constants import TOKEN_ENCODER_CACHE_SIZE

        info = get_encoder_cache_info()

        # Cache size should never exceed maxsize
        assert info["currsize"] <= info["maxsize"]
        assert info["maxsize"] == TOKEN_ENCODER_CACHE_SIZE


class TestEncoderFallback:
    """Tests for encoder fallback behavior."""

    @patch('app.utils.token_utils.tiktoken')
    def test_fallback_when_tiktoken_unavailable(self, mock_tiktoken):
        """Test fallback when tiktoken is not available."""
        mock_tiktoken.encoding_for_model.side_effect = KeyError("Unknown model")
        mock_tiktoken.get_encoding.return_value = MagicMock(
            encode=lambda x: list(x.split())
        )

        from app.utils.token_utils import _get_encoder, clear_encoder_cache

        clear_encoder_cache()
        encoder = _get_encoder("unknown-model-xyz")

        # Should fall back to cl100k_base
        assert encoder is not None

    def test_heuristic_fallback_on_error(self):
        """Test that heuristic is used when tiktoken fails."""
        from app.utils.token_utils import count_tokens

        # For non-OpenAI models, should always use heuristic
        text = "Test text for heuristic"
        result = count_tokens(text, model_name="local-model")

        # Should be approximately len(text) // 3
        expected = max(1, len(text) // 3)
        assert result == expected

    def test_get_encoder_returns_none_when_tiktoken_missing(self, monkeypatch):
        """Missing tiktoken should return None and force heuristic fallback."""
        from app.utils import token_utils as tu

        tu.clear_encoder_cache()
        monkeypatch.setattr(tu, "tiktoken", None)
        assert tu._get_encoder("gpt-4o") is None

    def test_get_encoder_returns_none_on_unexpected_error(self, monkeypatch):
        """Unexpected tokenizer initialization errors should be swallowed."""
        from app.utils import token_utils as tu

        tu.clear_encoder_cache()

        class _Tik:
            def encoding_for_model(self, _model):
                raise RuntimeError("boom")

        monkeypatch.setattr(tu, "tiktoken", _Tik())
        assert tu._get_encoder("gpt-4o") is None

    def test_count_tokens_falls_back_when_encoder_encode_fails(self, monkeypatch):
        """Encoding failures should fall back to the 4-char heuristic."""
        from app.utils import token_utils as tu

        class _Enc:
            def encode(self, _text):
                raise RuntimeError("encode fail")

        monkeypatch.setattr(tu, "_get_encoder", lambda model: _Enc())
        text = "abcdefghij"
        assert tu.count_tokens(text, model_name="gpt-4o") == max(1, len(text) // 4)


class TestTokenCountingAccuracy:
    """Tests for token counting accuracy."""

    def test_longer_text_more_tokens(self):
        """Test that longer text produces more tokens."""
        from app.utils.token_utils import count_tokens

        short_text = "Hello"
        long_text = "Hello world, this is a much longer text for testing"

        short_count = count_tokens(short_text, model_name="gpt-4o")
        long_count = count_tokens(long_text, model_name="gpt-4o")

        assert long_count > short_count

    def test_special_characters_counted(self):
        """Test that special characters are handled."""
        from app.utils.token_utils import count_tokens

        text_with_special = "Hello! @#$%^&*() World?"
        result = count_tokens(text_with_special, model_name="gpt-4o")

        assert result > 0

    def test_unicode_text_handled(self):
        """Test that unicode text is handled correctly."""
        from app.utils.token_utils import count_tokens

        unicode_text = "こんにちは世界"  # Japanese "Hello World"
        result = count_tokens(unicode_text, model_name="gpt-4o")

        assert result > 0

    def test_emoji_counted(self):
        """Test that emojis are counted correctly."""
        from app.utils.token_utils import count_tokens

        emoji_text = "Hello 👋 World 🌍"
        result = count_tokens(emoji_text, model_name="gpt-4o")

        assert result > 0
