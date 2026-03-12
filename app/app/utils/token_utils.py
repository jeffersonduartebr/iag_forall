# -*- coding: utf-8 -*-
# Objective: Utility helpers for token utils.
"""
token_utils.py — Token Counting Utilities
------------------------------------------
Provides accurate token counting for LLM inputs/outputs.
Uses tiktoken for OpenAI models with LRU cache to prevent memory growth.
"""

import logging
from functools import lru_cache
from typing import Optional

try:
    import tiktoken
    from tiktoken import Encoding
except ImportError:
    tiktoken = None
    Encoding = None

from app.config.constants import TOKEN_ENCODER_CACHE_SIZE

logger = logging.getLogger("token_utils")


@lru_cache(maxsize=TOKEN_ENCODER_CACHE_SIZE)
def _get_encoder(model_name: str) -> Optional["Encoding"]:
    """
    Get tiktoken encoder for a model with LRU cache.

    The cache is limited to TOKEN_ENCODER_CACHE_SIZE entries to prevent
    unbounded memory growth.

    Args:
        model_name: Name of the model to get encoder for

    Returns:
        Tiktoken Encoding object or None if not available
    """
    if tiktoken is None:
        return None

    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Fallback to cl100k_base for newer/unknown OpenAI models
        logger.debug(f"[token_utils] Unknown model '{model_name}', using cl100k_base")
        return tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        logger.warning(f"[token_utils] Error getting encoder for {model_name}: {e}")
        return None


def count_tokens(text: str, model_name: str = "gpt-4o") -> int:
    """
    Count tokens with precision using tiktoken for OpenAI models,
    or heuristic fallback for others.

    Args:
        text: Text to count tokens for
        model_name: Name of the model (used to select tokenizer)

    Returns:
        Number of tokens in the text
    """
    if not text:
        return 0

    # 1. Heuristic for non-OpenAI models (Llama, Gemma, etc)
    # Generally 1 token ~= 3 to 4 characters
    if not model_name.startswith(("gpt-", "o1-", "text-embedding")):
        return max(1, len(text) // 3)

    # 2. Tiktoken for OpenAI
    encoder = _get_encoder(model_name)
    if encoder is not None:
        try:
            return len(encoder.encode(text))
        except Exception as e:
            logger.warning(f"[token_utils] Error encoding with tiktoken: {e}")
            return max(1, len(text) // 4)

    # 3. Fallback heuristic
    return max(1, len(text) // 4)


def get_encoder_cache_info() -> dict:
    """Get information about the encoder cache."""
    info = _get_encoder.cache_info()
    return {
        "hits": info.hits,
        "misses": info.misses,
        "maxsize": info.maxsize,
        "currsize": info.currsize,
    }


def clear_encoder_cache() -> None:
    """Clear the encoder cache."""
    _get_encoder.cache_clear()
    logger.info("[token_utils] Encoder cache cleared")