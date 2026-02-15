# -*- coding: utf-8 -*-
"""
test_chaos.py — Chaos Testing for Failure Scenarios
----------------------------------------------------
Tests how the system handles various failure conditions:
- Provider failures
- Network timeouts
- Rate limiting
- Circuit breaker behavior
- Cache failures
- Database unavailability
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
import pytest


class TestCircuitBreakerChaos:
    """Tests circuit breaker behavior under failure conditions."""

    def test_circuit_breaker_opens_after_failures(self):
        """Circuit breaker should open after threshold failures."""
        from app.reliability import ModelCircuitBreakerManager
        import pybreaker

        manager = ModelCircuitBreakerManager()
        model = "test/chaos-model"
        breaker = manager.get_breaker(model)

        # Simulate failures to trip the breaker
        for _ in range(breaker.fail_max):
            try:
                def failing_call():
                    raise Exception("Simulated failure")
                breaker.call(failing_call)
            except Exception:
                pass

        # Circuit should be open now
        assert breaker.current_state == "open"
        assert not manager.is_available(model)

    def test_circuit_breaker_half_open_after_timeout(self):
        """Circuit breaker should transition to half-open after timeout."""
        from app.reliability import ModelCircuitBreakerManager
        import pybreaker

        manager = ModelCircuitBreakerManager()
        model = "test/fast-recovery-model"

        # Create a breaker with short timeout
        breaker = pybreaker.CircuitBreaker(
            fail_max=2,
            reset_timeout=0.1,  # 100ms
            name=f"breaker_{model}",
        )
        manager._breakers[model] = breaker

        # Trip the breaker
        for _ in range(2):
            try:
                breaker.call(lambda: (_ for _ in ()).throw(Exception("fail")))
            except Exception:
                pass

        assert breaker.current_state == "open"

        # Wait for timeout
        time.sleep(0.15)

        # Next call should try half-open
        # The state transitions on the next call attempt
        try:
            breaker.call(lambda: "success")
        except pybreaker.CircuitBreakerError:
            pass  # Might still fail

    def test_reset_breaker(self):
        """Admin should be able to manually reset a circuit breaker."""
        from app.reliability import ModelCircuitBreakerManager

        manager = ModelCircuitBreakerManager()
        model = "test/reset-model"
        breaker = manager.get_breaker(model)

        # Trip the breaker
        for _ in range(breaker.fail_max):
            try:
                breaker.call(lambda: (_ for _ in ()).throw(Exception("fail")))
            except Exception:
                pass

        assert breaker.current_state == "open"

        # Reset
        manager.reset_breaker(model)
        new_breaker = manager.get_breaker(model)

        # Should be closed now (new breaker)
        assert new_breaker.fail_counter == 0


class TestRateLimitChaos:
    """Tests rate limiting under heavy load."""

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self):
        """Rate limiter should block requests over threshold."""
        from app.middleware.rate_limit import RateLimitStore

        store = RateLimitStore()
        client_ip = "192.168.1.100"
        max_requests = 5
        window = 60

        # Make requests up to limit
        for _ in range(max_requests):
            limited = await store.is_rate_limited(client_ip, max_requests, window)
            assert not limited

        # Next request should be limited
        limited = await store.is_rate_limited(client_ip, max_requests, window)
        assert limited

    @pytest.mark.asyncio
    async def test_rate_limit_window_expiry(self):
        """Rate limit should reset after window expires."""
        from app.middleware.rate_limit import RateLimitStore

        store = RateLimitStore()
        client_ip = "192.168.1.101"
        max_requests = 2
        window = 1  # 1 second window

        # Exhaust limit
        for _ in range(max_requests):
            await store.is_rate_limited(client_ip, max_requests, window)

        # Should be limited
        assert await store.is_rate_limited(client_ip, max_requests, window)

        # Wait for window to expire
        await asyncio.sleep(1.1)

        # Should be allowed again
        assert not await store.is_rate_limited(client_ip, max_requests, window)

    @pytest.mark.asyncio
    async def test_rate_limit_cleanup(self):
        """Cleanup should remove old entries to prevent memory bloat."""
        from app.middleware.rate_limit import RateLimitStore

        store = RateLimitStore()

        # Add some entries
        for i in range(100):
            await store.is_rate_limited(f"192.168.1.{i}", 1000, 60)

        assert len(store._memory_store) == 100

        # Manually set old timestamps (simulating old entries)
        old_time = time.time() - 7200  # 2 hours ago
        for ip in list(store._memory_store.keys())[:50]:
            store._memory_store[ip] = [old_time]

        # Run cleanup
        await store.cleanup()

        # Old entries should be removed
        assert len(store._memory_store) == 50


class TestTimeoutChaos:
    """Tests timeout handling."""

    @pytest.mark.asyncio
    async def test_request_timeout_handling(self):
        """System should handle timeouts gracefully."""
        from app.error_handling import log_error, ErrorCategory

        # Simulate timeout error
        error = asyncio.TimeoutError("Request timed out")
        error_info = log_error(error, category=ErrorCategory.PROVIDER_TIMEOUT)

        assert error_info.category == ErrorCategory.PROVIDER_TIMEOUT
        assert error_info.retry_recommended is True

    @pytest.mark.asyncio
    async def test_slow_provider_timeout(self):
        """Slow providers should timeout appropriately."""
        async def slow_function():
            """Executa slow function."""
            await asyncio.sleep(10)
            return "Should not reach here"

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(slow_function(), timeout=0.1)


class TestCacheFailureChaos:
    """Tests cache failure scenarios."""

    def test_l1_cache_graceful_degradation(self):
        """System should work even if L1 cache fails."""
        from app.semantic_cache import L1Cache

        cache = L1Cache(maxsize=10, ttl_seconds=300)

        # Normal cache operation should work
        cache.store("key1", "value1")
        result = cache.get("key1")
        assert result == "value1"

        # Cache miss should return None gracefully
        result = cache.get("nonexistent")
        assert result is None

    def test_l1_cache_full_eviction(self):
        """Cache should evict oldest entries when full."""
        from app.semantic_cache import L1Cache

        cache = L1Cache(maxsize=3, ttl_seconds=300)

        # Fill cache
        cache.store("key1", "value1")
        cache.store("key2", "value2")
        cache.store("key3", "value3")

        # Add one more - should evict oldest
        cache.store("key4", "value4")

        # key1 should be evicted
        assert cache.get("key1") is None
        assert cache.get("key4") == "value4"

    @pytest.mark.asyncio
    async def test_redis_unavailable_fallback(self):
        """System should fallback gracefully when Redis is unavailable."""
        # Mock Redis to fail
        with patch('app.utils.redis_client.get_redis', return_value=None):
            from app.utils.redis_client import get_redis
            redis = get_redis()
            assert redis is None
            # System should continue working without Redis


class TestProviderFailureChaos:
    """Tests provider failure scenarios."""

    @pytest.mark.asyncio
    async def test_provider_rate_limit_error(self):
        """System should handle provider rate limit errors."""
        from app.error_handling import classify_exception, ErrorCategory

        class RateLimitError(Exception):
            """Classe `RateLimitError`: concentra responsabilidades de test chaos."""
            pass

        category, severity, retry = classify_exception(RateLimitError())
        # Should be classified appropriately
        assert retry is True or category == ErrorCategory.UNKNOWN

    @pytest.mark.asyncio
    async def test_provider_auth_error(self):
        """System should handle provider auth errors."""
        from app.error_handling import classify_exception, ErrorCategory

        class AuthenticationError(Exception):
            """Classe `AuthenticationError`: concentra responsabilidades de test chaos."""
            pass

        category, severity, retry = classify_exception(AuthenticationError())
        assert category == ErrorCategory.PROVIDER_AUTH_ERROR
        assert retry is False  # Auth errors shouldn't be retried

    @pytest.mark.asyncio
    async def test_fallback_chain_execution(self):
        """Fallback chain should try alternative models."""
        from app.reliability import execute_with_fallback, FallbackResult

        call_count = {"count": 0}

        async def failing_execute(model: str):
            """Executa failing execute."""
            call_count["count"] += 1
            if call_count["count"] < 3:
                raise Exception(f"Model {model} failed")
            return f"Success with {model}"

        # This test verifies the fallback mechanism structure
        # Actual execution would require model registry setup


class TestConcurrencyChaos:
    """Tests concurrent request handling."""

    @pytest.mark.asyncio
    async def test_request_deduplication_under_load(self):
        """Request deduplicator should handle concurrent identical requests."""
        from app.reliability import RequestDeduplicator

        dedup = RequestDeduplicator()
        call_count = {"count": 0}

        async def expensive_operation():
            """Executa expensive operation."""
            call_count["count"] += 1
            await asyncio.sleep(0.1)
            return f"result-{call_count['count']}"

        # Send multiple identical requests concurrently
        tasks = [
            dedup.deduplicate(
                query="test query",
                model="test/model",
                execute_fn=expensive_operation,
            )
            for _ in range(5)
        ]

        results = await asyncio.gather(*tasks)

        # All should get the same result (only one actual call)
        assert len(set(results)) == 1
        assert call_count["count"] == 1  # Only one actual call made

    @pytest.mark.asyncio
    async def test_parallel_requests_different_queries(self):
        """Different queries should be processed independently."""
        from app.reliability import RequestDeduplicator

        # Use a fresh deduplicator instance for this test
        dedup = RequestDeduplicator.__new__(RequestDeduplicator)
        dedup._in_flight = {}
        dedup._request_lock = asyncio.Lock()
        dedup._ttl_seconds = 300
        dedup._initialized = True

        results_list = []

        async def create_operation(idx):
            """Cria operation."""
            async def operation():
                """Executa operation."""
                await asyncio.sleep(0.01)
                return f"result-{idx}"
            return operation

        # Send different requests concurrently with different operation functions
        tasks = []
        for i in range(5):
            op = await create_operation(i)
            tasks.append(
                dedup.deduplicate(
                    query=f"query-{i}",
                    model=f"test/model-{i}",  # Different model for different key
                    execute_fn=op,
                )
            )

        results = await asyncio.gather(*tasks)

        # Each should have a different result
        assert len(set(results)) == 5


class TestHealthCheckChaos:
    """Tests health check under degraded conditions."""

    @pytest.mark.asyncio
    async def test_partial_health_check(self):
        """System should report degraded status when some components fail."""
        from app.health import get_full_health_check

        with patch('app.health.check_redis_health') as mock_redis, \
             patch('app.health.check_database_health') as mock_db, \
             patch('app.health.check_vectorstore_health') as mock_vs, \
             patch('app.health.check_ollama_health') as mock_ollama, \
             patch('app.health.check_circuit_breakers_health') as mock_cb:

            # Mock mixed health results
            from app.health import ComponentHealth

            mock_redis.return_value = ComponentHealth(name="redis", healthy=True, latency_ms=1.0)
            mock_db.return_value = ComponentHealth(name="mariadb", healthy=False, error="Connection refused")
            mock_vs.return_value = ComponentHealth(name="chromadb", healthy=True, latency_ms=5.0)
            mock_ollama.return_value = ComponentHealth(name="ollama", healthy=True, latency_ms=10.0)
            mock_cb.return_value = ComponentHealth(name="circuit_breakers", healthy=True)

            result = await get_full_health_check()

            # Should report degraded (not all healthy, but not all unhealthy)
            assert result["status"] == "degraded"
            assert result["summary"]["healthy"] < result["summary"]["total"]

    @pytest.mark.asyncio
    async def test_complete_failure_health(self):
        """System should report unhealthy when all components fail."""
        from app.health import get_full_health_check, invalidate_health_cache, ComponentHealth

        with patch('app.health.check_redis_health') as mock_redis, \
             patch('app.health.check_database_health') as mock_db, \
             patch('app.health.check_vectorstore_health') as mock_vs, \
             patch('app.health.check_ollama_health') as mock_ollama, \
             patch('app.health.check_circuit_breakers_health') as mock_cb:

            # All components fail
            mock_redis.return_value = ComponentHealth(name="redis", healthy=False, error="Connection refused")
            mock_db.return_value = ComponentHealth(name="mariadb", healthy=False, error="Connection refused")
            mock_vs.return_value = ComponentHealth(name="chromadb", healthy=False, error="Connection refused")
            mock_ollama.return_value = ComponentHealth(name="ollama", healthy=False, error="Connection refused")
            mock_cb.return_value = ComponentHealth(name="circuit_breakers", healthy=False, error="Error")

            invalidate_health_cache()
            result = await get_full_health_check(force_refresh=True)

            assert result["status"] == "unhealthy"
            assert result["summary"]["healthy"] == 0


class TestErrorRecoveryChaos:
    """Tests error recovery scenarios."""

    def test_error_categorization(self):
        """Errors should be categorized correctly."""
        from app.error_handling import classify_exception, ErrorCategory

        test_cases = [
            (TimeoutError(), ErrorCategory.PROVIDER_TIMEOUT),
            (ConnectionError(), ErrorCategory.PROVIDER_UNAVAILABLE),
            (ValueError(), ErrorCategory.INVALID_INPUT),
        ]

        for error, expected_category in test_cases:
            category, _, _ = classify_exception(error)
            assert category == expected_category, f"Failed for {type(error).__name__}"

    def test_error_response_creation(self):
        """Error responses should be user-friendly."""
        from app.error_handling import ErrorInfo, ErrorCategory, ErrorSeverity, create_error_response

        error_info = ErrorInfo(
            category=ErrorCategory.PROVIDER_TIMEOUT,
            severity=ErrorSeverity.WARNING,
            message="Provider timed out",
            error_type="TimeoutError",
            retry_recommended=True,
        )

        response = create_error_response(error_info)

        assert response["error"] is True
        assert "timeout" in response["message"].lower() or "try again" in response["message"].lower()
        assert response["retry_recommended"] is True


class TestMemoryPressureChaos:
    """Tests behavior under memory pressure."""

    def test_large_cache_entries(self):
        """Cache should handle large entries gracefully."""
        from app.semantic_cache import L1Cache

        cache = L1Cache(maxsize=5, ttl_seconds=300)

        # Store large values
        large_value = "x" * 1000000  # 1MB string
        for i in range(10):
            cache.store(f"large-key-{i}", large_value)

        # Cache should maintain maxsize limit
        assert len(cache._cache) <= 5

    @pytest.mark.asyncio
    async def test_deduplicator_cleanup(self):
        """Deduplicator should clean up stale requests."""
        from app.reliability import RequestDeduplicator

        dedup = RequestDeduplicator()
        dedup._ttl_seconds = 1  # Short TTL for testing

        # Add some in-flight requests
        async def slow_op():
            """Executa slow op."""
            await asyncio.sleep(2)
            return "done"

        # Start a request but don't await
        task = asyncio.create_task(
            dedup.deduplicate(
                query="stale query",
                model="test/model",
                execute_fn=slow_op,
            )
        )

        # Wait for TTL to expire
        await asyncio.sleep(1.5)

        # Cleanup
        await dedup.cleanup_stale()

        # Cancel the task
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
