# -*- coding: utf-8 -*-
# Objective: Test coverage for router core behavior and regressions.
"""
test_router_core.py — Unit tests for router_core.py
----------------------------------------------------
Tests the core routing logic including:
- route_and_answer() function
- Cache hit/miss paths
- Model selection flow
- Error handling
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import asyncio


class TestRouteAndAnswer:
    """Tests for the route_and_answer function."""

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_response(self):
        """When cache hits, should return cached response without calling model."""
        cached_response = {
            "text": "Cached answer",
            "similarity": 0.95,
            "model_used": "gpt-4",
            "image_output_b64": None
        }

        with patch("app.router_core.check_cache", new_callable=AsyncMock) as mock_cache, \
             patch("app.router_core.call_model", new_callable=AsyncMock) as mock_call:

            mock_cache.return_value = cached_response

            from app.router_core import route_and_answer
            result = await route_and_answer(
                query="What is Python?",
                use_cache=True
            )

            assert result["model"] == "semantic_cache"
            assert result["answer"] == "Cached answer"
            assert result["cost_per_1k"] == 0.0
            mock_call.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_miss_calls_model(self):
        """When cache misses, should call the selected model."""
        model_response = ("Model answer", {
            "latency": 1.5,
            "cost_per_1k": 0.001,
            "quality": 8.0,
            "raw_payload": "{}",
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "load_time": 0.5
        })

        with patch("app.router_core.check_cache", new_callable=AsyncMock) as mock_cache, \
             patch("app.router_core.call_model", new_callable=AsyncMock) as mock_call, \
             patch("app.router_core.choose_top2_models") as mock_choose, \
             patch("app.router_core.select_model") as mock_select, \
             patch("app.router_core.get_uncertainty_score") as mock_uq, \
             patch("app.router_core._ensure_ollama_model"):

            mock_cache.return_value = None
            mock_call.return_value = model_response
            mock_choose.return_value = ["ollama/phi4:latest", "openai/gpt-4o"]
            mock_select.return_value = "ollama/phi4:latest"
            mock_uq.return_value = 0.5

            from app.router_core import route_and_answer
            result = await route_and_answer(
                query="What is Python?",
                use_cache=True
            )

            assert result["model"] == "ollama/phi4:latest"
            assert result["answer"] == "Model answer"
            mock_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_disabled_skips_cache_check(self):
        """When use_cache=False, should skip cache lookup."""
        model_response = ("Direct answer", {
            "latency": 1.0,
            "cost_per_1k": 0.002,
            "quality": 7.5,
            "raw_payload": "{}",
            "prompt_tokens": 5,
            "completion_tokens": 15,
            "load_time": 0.0
        })

        with patch("app.router_core.check_cache", new_callable=AsyncMock) as mock_cache, \
             patch("app.router_core.call_model", new_callable=AsyncMock) as mock_call, \
             patch("app.router_core.choose_top2_models") as mock_choose, \
             patch("app.router_core.select_model") as mock_select, \
             patch("app.router_core.get_uncertainty_score") as mock_uq, \
             patch("app.router_core._ensure_ollama_model"):

            mock_call.return_value = model_response
            mock_choose.return_value = ["ollama/phi4:latest"]
            mock_select.return_value = "ollama/phi4:latest"
            mock_uq.return_value = 0.3

            from app.router_core import route_and_answer
            result = await route_and_answer(
                query="What is Python?",
                use_cache=False
            )

            mock_cache.assert_not_called()
            assert result["answer"] == "Direct answer"

    @pytest.mark.asyncio
    async def test_image_input_sets_vision_modality(self):
        """When image is provided with text modality, should switch to vision."""
        model_response = ("Vision answer", {
            "latency": 2.0,
            "cost_per_1k": 0.005,
            "quality": 8.5,
            "raw_payload": "{}",
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "load_time": 0.0,
            "image_output_b64": None
        })

        with patch("app.router_core.check_cache", new_callable=AsyncMock) as mock_cache, \
             patch("app.router_core.call_model", new_callable=AsyncMock) as mock_call, \
             patch("app.router_core.choose_top2_models") as mock_choose, \
             patch("app.router_core.select_model") as mock_select, \
             patch("app.router_core.get_uncertainty_score") as mock_uq, \
             patch("app.router_core._ensure_ollama_model"):

            mock_cache.return_value = None
            mock_call.return_value = model_response
            mock_choose.return_value = ["ollama/llava:latest"]
            mock_select.return_value = "ollama/llava:latest"
            mock_uq.return_value = 0.6

            from app.router_core import route_and_answer
            result = await route_and_answer(
                query="Describe this image",
                modality="text",
                image_b64="base64encodedimage"
            )

            assert result["modality"] == "vision"

    @pytest.mark.asyncio
    async def test_model_error_returns_error_message(self):
        """When model call fails, route should propagate the exception."""
        with patch("app.router_core.check_cache", new_callable=AsyncMock) as mock_cache, \
             patch("app.router_core.call_model", new_callable=AsyncMock) as mock_call, \
             patch("app.router_core.choose_top2_models") as mock_choose, \
             patch("app.router_core.select_model") as mock_select, \
             patch("app.router_core.get_uncertainty_score") as mock_uq, \
             patch("app.router_core._ensure_ollama_model"):

            mock_cache.return_value = None
            mock_call.side_effect = Exception("Connection timeout")
            mock_choose.return_value = ["ollama/phi4:latest"]
            mock_select.return_value = "ollama/phi4:latest"
            mock_uq.return_value = 0.5

            from app.router_core import route_and_answer
            with pytest.raises(Exception, match="Connection timeout"):
                await route_and_answer(query="Test query")


class TestDynamicWeights:
    """Tests for dynamic strategy weights."""

    def test_get_dynamic_strategy_weights_returns_dict(self):
        """Should return a dictionary with quality, latency, and cost weights."""
        with patch("app.router_core.settings") as mock_settings:
            mock_settings.NSGA_W_QUALITY = 1.0
            mock_settings.NSGA_W_LATENCY = 0.5
            mock_settings.NSGA_W_COST = 50.0

            from app.router_core import get_dynamic_strategy_weights
            weights = get_dynamic_strategy_weights("text")

            assert "w_quality" in weights
            assert "w_latency" in weights
            assert "w_cost" in weights
            assert weights["w_quality"] == 1.0
            assert weights["w_latency"] == 0.5
            assert weights["w_cost"] == 50.0


class TestRAGIntegration:
    """Tests for RAG (Retrieval Augmented Generation) integration."""

    @pytest.mark.asyncio
    async def test_rag_enabled_augments_prompt(self):
        """When use_rag=True, should augment the prompt with RAG context."""
        model_response = ("RAG-augmented answer", {
            "latency": 1.5,
            "cost_per_1k": 0.001,
            "quality": 9.0,
            "raw_payload": "{}",
            "prompt_tokens": 50,
            "completion_tokens": 30,
            "load_time": 0.0
        })

        with patch("app.router_core.check_cache", new_callable=AsyncMock) as mock_cache, \
             patch("app.router_core.call_model", new_callable=AsyncMock) as mock_call, \
             patch("app.router_core.choose_top2_models") as mock_choose, \
             patch("app.router_core.select_model") as mock_select, \
             patch("app.router_core.get_uncertainty_score") as mock_uq, \
             patch("app.router_core.build_augmented_prompt", new_callable=AsyncMock) as mock_rag, \
             patch("app.router_core._ensure_ollama_model"):

            mock_cache.return_value = None
            mock_call.return_value = model_response
            mock_choose.return_value = ["ollama/phi4:latest"]
            mock_select.return_value = "ollama/phi4:latest"
            mock_uq.return_value = 0.4
            mock_rag.return_value = "Augmented: What is NSGA-II? Context: NSGA-II is..."

            from app.router_core import route_and_answer
            result = await route_and_answer(
                query="What is NSGA-II?",
                use_rag=True
            )

            mock_rag.assert_called_once()
            assert result["answer"] == "RAG-augmented answer"
