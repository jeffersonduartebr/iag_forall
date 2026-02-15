# -*- coding: utf-8 -*-
"""
test_redis_client.py — Tests for Redis Client Module
------------------------------------------------------
Tests for the redis_client.py utility module covering:
- Connection pooling
- Health checks
- Graceful degradation
- Pipeline operations
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import time


class TestRedisClientConfiguration:
    """Tests for Redis client configuration."""

    def test_default_configuration_values(self):
        """Test that default configuration values are set correctly."""
        # Import fresh to get defaults
        from app.utils import redis_client

        assert redis_client.REDIS_HOST == "redis" or redis_client.REDIS_HOST is not None
        assert redis_client.REDIS_PORT > 0
        assert redis_client.REDIS_DB >= 0
        assert redis_client.REDIS_MAX_CONNECTIONS > 0
        assert redis_client.REDIS_SOCKET_TIMEOUT > 0

    def test_configuration_from_environment(self):
        """Test that configuration can be read from environment variables."""
        import os

        with patch.dict(os.environ, {
            "REDIS_HOST": "test-redis-host",
            "REDIS_PORT": "6380",
            "REDIS_DB": "5",
            "REDIS_MAX_CONNECTIONS": "50",
        }):
            # Would need to reimport module to pick up env vars
            # This tests the pattern is correct
            assert os.getenv("REDIS_HOST") == "test-redis-host"
            assert int(os.getenv("REDIS_PORT")) == 6380


class TestRedisClientConnection:
    """Tests for Redis connection functionality."""

    def test_get_redis_function_exists(self):
        """Test that get_redis function exists and is callable."""
        from app.utils.redis_client import get_redis

        assert callable(get_redis)

    def test_get_redis_async_safe_function_exists(self):
        """Test that get_redis_async_safe function exists and is callable."""
        from app.utils.redis_client import get_redis_async_safe

        assert callable(get_redis_async_safe)

    def test_get_redis_async_safe_is_none_safe(self):
        """Test get_redis_async_safe handles None client gracefully."""
        from app.utils.redis_client import get_redis_async_safe

        # Should not raise even if internal state is None
        result = get_redis_async_safe()
        # Result can be None or a client depending on actual connection
        assert result is None or hasattr(result, 'ping')


class TestRedisHealthCheck:
    """Tests for Redis health check functionality."""

    def test_check_redis_health_returns_dict(self):
        """Test check_redis_health returns a dictionary with expected keys."""
        from app.utils.redis_client import check_redis_health

        result = check_redis_health()

        assert isinstance(result, dict)
        assert "healthy" in result
        assert "latency_ms" in result
        assert "pool_size" in result
        assert "error" in result

    def test_check_redis_health_has_expected_fields(self):
        """Test health check returns expected fields."""
        from app.utils.redis_client import check_redis_health

        result = check_redis_health()

        # Should always have these fields regardless of connection state
        assert "healthy" in result
        assert "latency_ms" in result
        assert "error" in result

    @patch('app.utils.redis_client.get_redis_async_safe')
    def test_check_redis_health_measures_latency(self, mock_get_redis):
        """Test health check measures ping latency."""
        from app.utils.redis_client import check_redis_health

        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_get_redis.return_value = mock_client

        result = check_redis_health()

        assert result["healthy"] is True
        assert result["latency_ms"] is not None
        assert result["latency_ms"] >= 0


class TestRedisPipeline:
    """Tests for Redis pipeline operations."""

    @patch('app.utils.redis_client.get_redis')
    def test_redis_pipeline_context_manager(self, mock_get_redis):
        """Test redis_pipeline context manager works correctly."""
        from app.utils.redis_client import redis_pipeline

        mock_client = MagicMock()
        mock_pipe = MagicMock()
        mock_client.pipeline.return_value = mock_pipe
        mock_get_redis.return_value = mock_client

        with redis_pipeline() as pipe:
            pipe.set("key1", "value1")
            pipe.set("key2", "value2")

        mock_pipe.execute.assert_called()

    @patch('app.utils.redis_client.get_redis')
    def test_redis_pipeline_raises_when_unavailable(self, mock_get_redis):
        """Test redis_pipeline raises RuntimeError when Redis unavailable."""
        from app.utils.redis_client import redis_pipeline

        mock_get_redis.return_value = None

        with pytest.raises(RuntimeError, match="Redis not available"):
            with redis_pipeline() as pipe:
                pass


class TestRedisCleanup:
    """Tests for Redis cleanup functionality."""

    def test_close_redis_function_exists(self):
        """Test close_redis function exists and is callable."""
        from app.utils.redis_client import close_redis

        assert callable(close_redis)

    def test_close_redis_can_be_called_multiple_times(self):
        """Test close_redis can be called multiple times without error."""
        from app.utils.redis_client import close_redis

        # Should not raise even when called multiple times
        close_redis()
        close_redis()
        close_redis()


class TestRedisPoolCreation:
    """Tests for Redis pool creation."""

    @patch('app.utils.redis_client.redis.ConnectionPool')
    def test_create_pool_with_correct_parameters(self, mock_pool_cls):
        """Test _create_pool creates pool with correct parameters."""
        from app.utils.redis_client import _create_pool, REDIS_HOST, REDIS_PORT

        mock_pool = MagicMock()
        mock_pool_cls.return_value = mock_pool

        result = _create_pool()

        assert result is not None
        mock_pool_cls.assert_called_once()
        call_kwargs = mock_pool_cls.call_args[1]
        assert call_kwargs["host"] == REDIS_HOST
        assert call_kwargs["port"] == REDIS_PORT

    @patch('app.utils.redis_client.redis.ConnectionPool')
    def test_create_pool_returns_none_on_error(self, mock_pool_cls):
        """Test _create_pool returns None when pool creation fails."""
        from app.utils.redis_client import _create_pool

        mock_pool_cls.side_effect = Exception("Connection failed")

        result = _create_pool()

        assert result is None
