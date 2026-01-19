import pytest
from unittest.mock import patch, MagicMock
from app.settings_dynamic import LRUCache, DynamicSettings


class TestLRUCache:
    """Test the LRU cache implementation."""

    def test_cache_basic_operations(self):
        """Test basic get/set operations."""
        cache = LRUCache(maxsize=10, ttl_s=300)

        # Set a value
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        # Non-existent key returns None
        assert cache.get("nonexistent") is None

    def test_cache_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = LRUCache(maxsize=3, ttl_s=300)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Access key1 to make it recently used
        cache.get("key1")

        # Add key4 - should evict key2 (least recently used)
        cache.set("key4", "value4")

        assert cache.get("key1") == "value1"  # Still there
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") == "value3"  # Still there
        assert cache.get("key4") == "value4"  # Newly added

    def test_cache_ttl_expiry(self):
        """Test TTL expiry."""
        import time

        cache = LRUCache(maxsize=10, ttl_s=1)  # 1 second TTL

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        # Wait for TTL to expire
        time.sleep(1.5)

        assert cache.get("key1") is None

    def test_cache_clear(self):
        """Test cache clear operation."""
        cache = LRUCache(maxsize=10, ttl_s=300)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None


class TestDynamicSettings:
    """Test the DynamicSettings class."""

    def test_settings_defaults(self):
        """Test that default values are accessible."""
        from app.settings_dynamic import settings

        # These should have default values
        assert hasattr(settings, "MAX_TOKENS_DEFAULT")
        assert hasattr(settings, "TEMPERATURE_DEFAULT")

    def test_settings_get_method(self):
        """Test the get method with default values."""
        from app.settings_dynamic import settings

        # Get with default
        result = settings.get("NONEXISTENT_KEY", "default_value")
        assert result == "default_value"

    def test_settings_snapshot(self):
        """Test snapshot method returns dict."""
        from app.settings_dynamic import settings

        snapshot = settings.snapshot()
        assert isinstance(snapshot, dict)
