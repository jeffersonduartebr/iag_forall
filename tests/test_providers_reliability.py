# -*- coding: utf-8 -*-
# Objective: Test coverage for providers reliability behavior and regressions.
"""
test_providers_reliability.py — Tests for provider reliability patterns
-----------------------------------------------------------------------
Tests circuit breaker, retry logic, and error handling in providers_async.py
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pybreaker
import pytest


class TestCircuitBreaker:
    """Tests for circuit breaker functionality."""

    def test_cloud_breaker_configuration(self):
        """Cloud breaker should have correct configuration."""
        from app.providers_async import cloud_breaker

        assert cloud_breaker.fail_max == 5
        assert cloud_breaker.reset_timeout == 60
        assert cloud_breaker.name == "cloud_breaker"

    def test_local_breaker_configuration(self):
        """Local breaker should have correct configuration."""
        from app.providers_async import local_breaker

        assert local_breaker.fail_max == 3
        assert local_breaker.reset_timeout == 30
        assert local_breaker.name == "local_breaker"

    def test_breaker_states(self):
        """Circuit breaker should have correct state transitions."""
        # Create a fresh breaker for testing
        test_breaker = pybreaker.CircuitBreaker(
            fail_max=2,
            reset_timeout=1,
            name="test_breaker"
        )

        # Initially closed
        assert test_breaker.current_state == "closed"

        # Simulate failures
        @test_breaker
        def failing_func():
            """Execute the failing func routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            raise Exception("Simulated failure")

        for _ in range(2):
            try:
                failing_func()
            except Exception:
                pass

        # Should be open after fail_max failures
        assert test_breaker.current_state == "open"


class TestRetryLogic:
    """Tests for retry configuration."""

    def test_retryable_errors_includes_timeout(self):
        """RETRYABLE_ERRORS should include timeout errors."""
        from app.providers_async import RETRYABLE_ERRORS

        assert httpx.ConnectTimeout in RETRYABLE_ERRORS
        assert httpx.ReadTimeout in RETRYABLE_ERRORS

    def test_retry_strategy_configuration(self):
        """COMMON_RETRY_STRATEGY should have correct configuration."""
        from app.providers_async import COMMON_RETRY_STRATEGY

        # The retry decorator should exist and be callable
        assert COMMON_RETRY_STRATEGY is not None
        assert hasattr(COMMON_RETRY_STRATEGY, '__call__')


class TestProviderFactory:
    """Tests for ProviderFactory."""

    def test_factory_returns_correct_provider_for_openai(self, monkeypatch):
        """Factory should return OpenAIProvider for openai/ prefix."""
        from app.providers_async import OpenAIProvider, ProviderFactory

        from app import providers_async as pa

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        pa.reset_provider_runtime_state()
        with patch.object(pa, "OPENAI_API_KEY", "test-key"):
            provider = ProviderFactory.get_provider("openai/gpt-4o")
            assert isinstance(provider, OpenAIProvider)

    def test_factory_returns_correct_provider_for_ollama(self):
        """Factory should return OllamaProvider for ollama/ prefix."""
        from app.providers_async import OllamaProvider, ProviderFactory

        provider = ProviderFactory.get_provider("ollama/phi4:latest")
        assert isinstance(provider, OllamaProvider)

    def test_factory_returns_ollama_for_no_prefix(self):
        """Factory should default to OllamaProvider for models without prefix."""
        from app.providers_async import OllamaProvider, ProviderFactory

        provider = ProviderFactory.get_provider("phi4:latest")
        assert isinstance(provider, OllamaProvider)

    def test_factory_raises_for_unknown_namespace(self):
        """Factory should raise ValueError for unknown namespace."""
        from app.providers_async import ProviderFactory

        with pytest.raises(RuntimeError, match="not configured in the environment"):
            ProviderFactory.get_provider("unknown_provider/model")


class TestOllamaProvider:
    """Tests for OllamaProvider."""

    @pytest.mark.asyncio
    async def test_ollama_provider_generates_response(self):
        """OllamaProvider should generate a response."""
        mock_response = {
            "response": "Test response",
            "prompt_eval_count": 10,
            "eval_count": 20,
            "load_duration": 500000000  # 0.5s in nanoseconds
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response_obj
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            from app.providers_async import OllamaProvider

            provider = OllamaProvider()
            # Reset the breaker state for testing
            from app.providers_async import local_breaker
            local_breaker._state = pybreaker.CircuitClosedState(local_breaker)

            result = await provider.generate(
                prompt="Test prompt",
                model="phi4:latest"
            )

            assert result.text == "Test response"
            assert result.prompt_tokens == 10
            assert result.completion_tokens == 20

    @pytest.mark.asyncio
    async def test_ollama_provider_extracts_reasoning(self):
        """OllamaProvider should extract <think> tags from response."""
        mock_response = {
            "response": "<think>This is my reasoning process</think>Final answer here",
            "prompt_eval_count": 10,
            "eval_count": 30,
            "load_duration": 0
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response_obj
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            from app.providers_async import OllamaProvider, local_breaker

            provider = OllamaProvider()
            local_breaker._state = pybreaker.CircuitClosedState(local_breaker)

            result = await provider.generate(
                prompt="Solve this math problem",
                model="phi4:latest"
            )

            assert result.reasoning == "This is my reasoning process"
            assert result.text == "Final answer here"

    @pytest.mark.asyncio
    async def test_ollama_provider_disables_thinking_for_non_reasoning_models(self):
        """OllamaProvider should send think=false for non-reasoning models."""
        mock_response = {
            "response": "4",
            "prompt_eval_count": 10,
            "eval_count": 2,
            "load_duration": 0,
        }

        mock_http_client = AsyncMock()
        mock_response_obj = MagicMock()
        mock_response_obj.json.return_value = mock_response
        mock_response_obj.raise_for_status = MagicMock()
        mock_http_client.post.return_value = mock_response_obj

        with patch("app.providers_async.get_http_client", AsyncMock(return_value=mock_http_client)):
            from app.providers_async import OllamaProvider, local_breaker

            provider = OllamaProvider()
            local_breaker._state = pybreaker.CircuitClosedState(local_breaker)

            result = await provider.generate(
                prompt="Quanto e 2+2?",
                model="qwen3.5:2b",
                max_tokens=64,
            )

            sent_payload = mock_http_client.post.await_args.kwargs["json"]
            assert sent_payload["think"] is False
            assert result.text == "4"
            assert result.reasoning is None

    @pytest.mark.asyncio
    async def test_ollama_provider_captures_separate_thinking_field(self):
        """OllamaProvider should keep separate thinking in reasoning metadata only."""
        mock_response = {
            "response": "Resposta final",
            "thinking": "Etapas internas de raciocinio",
            "prompt_eval_count": 10,
            "eval_count": 8,
            "load_duration": 0,
        }

        mock_http_client = AsyncMock()
        mock_response_obj = MagicMock()
        mock_response_obj.json.return_value = mock_response
        mock_response_obj.raise_for_status = MagicMock()
        mock_http_client.post.return_value = mock_response_obj

        with patch("app.providers_async.get_http_client", AsyncMock(return_value=mock_http_client)):
            from app.providers_async import OllamaProvider, local_breaker

            provider = OllamaProvider()
            local_breaker._state = pybreaker.CircuitClosedState(local_breaker)

            result = await provider.generate(
                prompt="Explique em uma frase.",
                model="qwen3.5:2b",
                max_tokens=64,
            )

            assert result.text == "Resposta final"
            assert result.reasoning == "Etapas internas de raciocinio"

    def test_apply_ollama_performance_preferences_prefers_interactive_warm_models(self):
        """Interactive traffic should prefer warm, resident local models."""
        from app import providers_async as pa

        pa._ollama_runtime_state.clear()
        pa._mark_ollama_model_state("ollama/qwen3.5:4b", loaded=True, load_seconds=0.2, inflight_delta=0, queue_wait_seconds=0.0)
        pa._mark_ollama_model_state("ollama/ministral-3:3b", loaded=False, load_seconds=8.0, inflight_delta=0, queue_wait_seconds=0.8)

        with patch("app.providers_async._runtime_provider_settings", return_value={
            "ollama_load_penalty_enabled": True,
            "ollama_route_candidate_limit": 3,
            "ollama_concurrency_limit": 5,
            "ollama_dynamic_concurrency_enabled": False,
            "ollama_interactive_warm_models": ["ollama/qwen3.5:4b"],
            "ollama_warm_models": ["ollama/qwen3.5:4b"],
        }):
            ordered = pa.apply_ollama_performance_preferences(
                ["ollama/ministral-3:3b", "ollama/qwen3.5:4b"],
                runtime_hints={"workload_class": "simple_text"},
            )

        assert ordered[0] == "ollama/qwen3.5:4b"

    def test_provider_unavailable_negative_cache_roundtrip(self):
        """Temporary unavailability cache should block only inside its TTL window."""
        from app import providers_async as pa

        pa._provider_unavailable_until.clear()
        with patch("app.providers_async._runtime_provider_settings", return_value={"provider_unavailable_negative_cache_ttl_seconds": 30}):
            pa.mark_provider_unavailable("ollama/qwen3.5:4b")
        assert pa.is_provider_temporarily_unavailable("ollama/qwen3.5:4b") is True
        pa.clear_provider_unavailable("ollama/qwen3.5:4b")
        assert pa.is_provider_temporarily_unavailable("ollama/qwen3.5:4b") is False

    def test_background_judge_throttle_activates_under_local_pressure(self):
        """Background judge load shedding should activate when local inflight reaches the effective limit."""
        from app import providers_async as pa

        pa._ollama_runtime_state.clear()
        pa._mark_ollama_model_state("ollama/qwen3.5:4b", loaded=True, inflight_delta=3)
        pa._ollama_concurrency_controller._effective_limit = 3
        pa._ollama_concurrency_controller._latest_ratio = 0.5

        with patch("app.providers_async._runtime_provider_settings", return_value={
            "ollama_background_load_shedding_enabled": True,
            "ollama_concurrency_limit": 3,
            "ollama_dynamic_concurrency_enabled": False,
            "ollama_vram_high_watermark": 0.82,
            "background_throttle_inflight_threshold": 3,
            "background_throttle_queue_wait_p95_ms": 750,
        }):
            assert pa.should_throttle_background_judge() is True

    def test_background_judge_throttle_activates_on_queue_wait_pressure(self):
        """Background shedding should also activate when queue wait grows too high."""
        from app import providers_async as pa

        pa._ollama_runtime_state.clear()
        pa._mark_ollama_model_state("ollama/qwen3.5:4b", loaded=True, queue_wait_seconds=1.2)
        pa._ollama_concurrency_controller._effective_limit = 5
        pa._ollama_concurrency_controller._latest_ratio = 0.4

        with patch("app.providers_async._runtime_provider_settings", return_value={
            "ollama_background_load_shedding_enabled": True,
            "ollama_concurrency_limit": 5,
            "ollama_dynamic_concurrency_enabled": False,
            "ollama_vram_high_watermark": 0.82,
            "background_throttle_inflight_threshold": 5,
            "background_throttle_queue_wait_p95_ms": 700,
        }):
            assert pa.should_throttle_background_judge() is True

    @pytest.mark.asyncio
    async def test_ollama_provider_emits_queue_wait_and_load_metrics(self):
        """OllamaProvider should emit queue wait, inflight, load, and throughput metrics."""
        metric_inflight = MagicMock()
        metric_queue = MagicMock()
        metric_load = MagicMock()
        metric_loaded = MagicMock()
        metric_tps = MagicMock()
        for metric in (metric_inflight, metric_queue, metric_load, metric_loaded, metric_tps):
            metric.labels.return_value = metric

        mock_response = {
            "response": "Resposta final",
            "prompt_eval_count": 10,
            "eval_count": 20,
            "load_duration": 1_000_000_000,
        }

        mock_http_client = AsyncMock()
        mock_response_obj = MagicMock()
        mock_response_obj.json.return_value = mock_response
        mock_response_obj.raise_for_status = MagicMock()
        mock_http_client.post.return_value = mock_response_obj

        with patch("app.providers_async.get_http_client", AsyncMock(return_value=mock_http_client)):
            from app.providers_async import OllamaProvider, local_breaker

            from app import providers_async as pa

            local_breaker._state = pybreaker.CircuitClosedState(local_breaker)
            with patch.object(pa, "PROVIDER_INFLIGHT_REQUESTS", metric_inflight), \
                 patch.object(pa, "PROVIDER_QUEUE_WAIT", metric_queue), \
                 patch.object(pa, "OLLAMA_MODEL_LOAD_SECONDS", metric_load), \
                 patch.object(pa, "OLLAMA_MODEL_LOADED", metric_loaded), \
                 patch.object(pa, "GENERATION_TOKENS_PER_SECOND", metric_tps):
                provider = OllamaProvider()
                await provider.generate(prompt="Explique.", model="qwen3.5:2b", max_tokens=64)

        assert metric_queue.observe.called
        assert metric_inflight.inc.called
        assert metric_inflight.dec.called
        assert metric_load.observe.called
        assert metric_loaded.set.called
        assert metric_tps.observe.called

    @pytest.mark.asyncio
    async def test_dynamic_controller_reduces_and_recovers_limit(self):
        """The dynamic controller should reduce on high VRAM and recover conservatively."""
        from app import providers_async as pa

        controller = pa.OllamaConcurrencyController()
        cfg = {
            "ollama_concurrency_limit": 5,
            "ollama_dynamic_concurrency_enabled": True,
            "ollama_concurrency_min": 1,
            "ollama_concurrency_max": 5,
            "ollama_vram_target_utilization": 0.72,
            "ollama_vram_high_watermark": 0.82,
            "ollama_vram_low_watermark": 0.55,
            "ollama_vram_poll_interval_seconds": 5,
            "ollama_concurrency_step_up": 1,
            "ollama_concurrency_step_down": 1,
            "ollama_concurrency_stable_windows": 2,
            "ollama_gpu_index": 0,
            "adaptive_timeout_enabled": True,
            "base_timeout": 60,
            "max_timeout": 1200,
            "timeout_multiplier": 2.0,
            "reasoning_multiplier": 3.0,
        }

        with patch.object(
            pa,
            "_runtime_provider_settings",
            return_value=cfg,
        ):
            with patch.object(controller, "_read_vram_snapshot", side_effect=[
                (9 * 1024**3, 10 * 1024**3),
                (4 * 1024**3, 10 * 1024**3),
                (4 * 1024**3, 10 * 1024**3),
            ]):
                await controller.force_refresh()
                assert controller.get_effective_limit() == 4
                await controller.force_refresh()
                assert controller.get_effective_limit() == 4
                await controller.force_refresh()
        assert controller.get_effective_limit() == 5

    def test_apply_ollama_performance_preferences_prefers_loaded_models(self):
        """Routing preferences should trim cold local candidates when warm alternatives exist."""
        from app import providers_async as pa

        pa._ollama_runtime_state.clear()
        pa._mark_ollama_model_state("ollama/gemma3:4b", loaded=True, load_seconds=0.2)
        pa._mark_ollama_model_state("ollama/qwen3.5:4b", loaded=False, load_seconds=12.0)
        pa._mark_ollama_model_state("ollama/ministral-3:3b", loaded=True, load_seconds=0.4, inflight_delta=2)

        with patch.object(
            pa,
            "_runtime_provider_settings",
            return_value={
                "ollama_concurrency_limit": 5,
                "ollama_dynamic_concurrency_enabled": False,
                "ollama_concurrency_min": 1,
                "ollama_concurrency_max": 5,
                "ollama_vram_target_utilization": 0.72,
                "ollama_vram_high_watermark": 0.82,
                "ollama_vram_low_watermark": 0.55,
                "ollama_vram_poll_interval_seconds": 5,
                "ollama_concurrency_step_up": 1,
                "ollama_concurrency_step_down": 1,
                "ollama_concurrency_stable_windows": 3,
                "ollama_gpu_index": 0,
                "ollama_warm_models": ["ollama/gemma3:4b"],
                "ollama_warmup_generate_enabled": True,
                "ollama_load_penalty_enabled": True,
                "ollama_route_candidate_limit": 2,
                "adaptive_timeout_enabled": True,
                "base_timeout": 60,
                "max_timeout": 1200,
                "timeout_multiplier": 2.0,
                "reasoning_multiplier": 3.0,
            },
        ):
            preferred = pa.apply_ollama_performance_preferences(
                ["ollama/qwen3.5:4b", "ollama/gemma3:4b", "ollama/ministral-3:3b", "openai/gpt-4o"]
            )

        assert preferred[0] == "ollama/gemma3:4b"
        assert "ollama/qwen3.5:4b" not in preferred[:2]
        assert "openai/gpt-4o" in preferred

    @pytest.mark.asyncio
    async def test_warm_ollama_model_runtime_marks_loaded_state(self):
        """Warmup requests should mark the local model as resident after a successful ping generation."""
        mock_http_client = AsyncMock()
        mock_response_obj = MagicMock()
        mock_response_obj.json.return_value = {"response": "OK", "load_duration": 500_000_000}
        mock_response_obj.raise_for_status = MagicMock()
        mock_http_client.post.return_value = mock_response_obj

        with patch("app.providers_async.get_http_client", AsyncMock(return_value=mock_http_client)):
            from app import providers_async as pa

            pa._ollama_runtime_state.clear()
            warmed = await pa.warm_ollama_model_runtime("ollama/gemma3:4b")

        assert warmed is True
        assert pa._ollama_runtime_state["ollama/gemma3:4b"]["loaded"] is True

    @pytest.mark.asyncio
    async def test_dynamic_controller_falls_back_when_telemetry_fails(self):
        """The dynamic controller should keep a static fallback on telemetry failure."""
        from app import providers_async as pa

        controller = pa.OllamaConcurrencyController()
        cfg = {
            "ollama_concurrency_limit": 5,
            "ollama_dynamic_concurrency_enabled": True,
            "ollama_concurrency_min": 1,
            "ollama_concurrency_max": 5,
            "ollama_vram_target_utilization": 0.72,
            "ollama_vram_high_watermark": 0.82,
            "ollama_vram_low_watermark": 0.55,
            "ollama_vram_poll_interval_seconds": 5,
            "ollama_concurrency_step_up": 1,
            "ollama_concurrency_step_down": 1,
            "ollama_concurrency_stable_windows": 2,
            "ollama_gpu_index": 0,
            "adaptive_timeout_enabled": True,
            "base_timeout": 60,
            "max_timeout": 1200,
            "timeout_multiplier": 2.0,
            "reasoning_multiplier": 3.0,
        }
        with patch.object(
            pa,
            "_runtime_provider_settings",
            return_value=cfg,
        ):
            with patch.object(controller, "_read_vram_snapshot", return_value=None):
                await controller.force_refresh()
                assert controller.get_effective_limit() == 5


class TestCallModelWrapper:
    """Tests for the call_model wrapper function."""

    @pytest.mark.asyncio
    async def test_call_model_returns_text_and_meta(self):
        """call_model should return (text, metadata) tuple."""
        mock_llm_response = MagicMock()
        mock_llm_response.text = "Response text"
        mock_llm_response.latency = 1.0
        mock_llm_response.load_time = 0.5
        mock_llm_response.cost = 0.001
        mock_llm_response.prompt_tokens = 10
        mock_llm_response.completion_tokens = 20
        mock_llm_response.raw_payload = "{}"
        mock_llm_response.reasoning = None

        with patch("app.providers_async.ProviderFactory.get_provider") as mock_factory:
            mock_provider = AsyncMock()
            mock_provider.generate.return_value = mock_llm_response
            mock_factory.return_value = mock_provider

            from app.providers_async import call_model

            text, meta = await call_model(
                model="ollama/phi4:latest",
                prompt="Test prompt"
            )

            assert text == "Response text"
            assert "latency" in meta
            assert "cost_per_1k" in meta

    @pytest.mark.asyncio
    async def test_call_model_handles_circuit_breaker_open(self):
        """call_model should raise ProviderCircuitOpenError for open breaker."""
        with patch("app.providers_async.ProviderFactory.get_provider") as mock_factory:
            mock_provider = AsyncMock()
            mock_provider.generate.side_effect = pybreaker.CircuitBreakerError(
                pybreaker.CircuitBreaker(fail_max=1, reset_timeout=1)
            )
            mock_factory.return_value = mock_provider

            from app.providers_async import ProviderCircuitOpenError, call_model

            with pytest.raises(ProviderCircuitOpenError):
                await call_model(
                    model="ollama/phi4:latest",
                    prompt="Test prompt"
                )

    @pytest.mark.asyncio
    async def test_call_model_handles_general_exception(self):
        """call_model should raise ProviderCallError for generic failures."""
        with patch("app.providers_async.ProviderFactory.get_provider") as mock_factory:
            mock_provider = AsyncMock()
            mock_provider.generate.side_effect = Exception("Network error")
            mock_factory.return_value = mock_provider

            from app.providers_async import ProviderCallError, call_model

            with pytest.raises(ProviderCallError) as exc:
                await call_model(
                    model="ollama/phi4:latest",
                    prompt="Test prompt"
                )
            assert "Network error" in str(exc.value)


class TestHeuristicQuality:
    """Tests for heuristic_quality_estimate function."""

    def test_empty_text_returns_zero(self):
        """Empty text should return 0.0 quality."""
        from app.providers_async import heuristic_quality_estimate

        assert heuristic_quality_estimate("") == 0.0
        assert heuristic_quality_estimate("   ") == 0.0

    def test_longer_text_has_higher_quality(self):
        """Longer text should have higher quality score."""
        from app.providers_async import heuristic_quality_estimate

        short_quality = heuristic_quality_estimate("Hi.")
        long_quality = heuristic_quality_estimate("This is a much longer response with proper punctuation.")

        assert long_quality > short_quality

    def test_punctuation_adds_bonus(self):
        """Text with punctuation should have higher quality."""
        from app.providers_async import heuristic_quality_estimate

        no_punct = heuristic_quality_estimate("This is a test")
        with_punct = heuristic_quality_estimate("This is a test.")

        assert with_punct > no_punct

    def test_quality_is_bounded(self):
        """Quality score should be between 0 and 10."""
        from app.providers_async import heuristic_quality_estimate

        quality = heuristic_quality_estimate("x" * 10000)
        assert 0.0 <= quality <= 10.0


class TestReasoningModelDetection:
    """Tests for reasoning model detection."""

    def test_phi4_is_reasoning_model(self):
        """phi4 should be detected as a reasoning model."""
        from app.providers_async import REASONING_MODEL_KEYWORDS

        model_name = "phi4:latest"
        is_reasoning = any(k in model_name.lower() for k in REASONING_MODEL_KEYWORDS)
        assert is_reasoning

    def test_deepseek_r1_is_reasoning_model(self):
        """deepseek-r1 should be detected as a reasoning model."""
        from app.providers_async import REASONING_MODEL_KEYWORDS

        model_name = "deepseek-r1:latest"
        is_reasoning = any(k in model_name.lower() for k in REASONING_MODEL_KEYWORDS)
        assert is_reasoning

    def test_gpt4_is_not_reasoning_model(self):
        """gpt-4 should not be detected as a reasoning model."""
        from app.providers_async import REASONING_MODEL_KEYWORDS

        model_name = "gpt-4o"
        is_reasoning = any(k in model_name.lower() for k in REASONING_MODEL_KEYWORDS)
        assert not is_reasoning
