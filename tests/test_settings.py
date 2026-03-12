"""Tests for dynamic settings cache, snapshot, and catalog helpers."""

from app.settings_dynamic import DynamicSettings, LRUCache


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

    def test_snapshot_only_known_and_keys_metadata(self):
        """Catalog helpers should expose known keys and metadata."""
        settings = DynamicSettings()
        keys = settings.keys()
        assert "MAX_TOKENS_DEFAULT" in keys
        assert "DB_HOST" not in settings.snapshot(only_known=True)
        meta = settings.metadata("OLLAMA_HOST")
        assert meta["domain"] == "providers"
        assert meta["mutability"] == "requires_restart"

    def test_typed_getters_cover_defaults(self):
        """Typed getters should coerce fallback values consistently."""
        settings = DynamicSettings()
        assert settings._get_int("MISSING_INT", 7) == 7
        assert settings._get_float("MISSING_FLOAT", 1.5) == 1.5
        assert settings._get_bool("MISSING_BOOL", True) is True
        assert settings._get_list("MISSING_LIST") == []

    def test_many_properties_are_accessible(self, monkeypatch):
        """Access a broad set of typed properties to cover the property surface."""
        settings = DynamicSettings()
        monkeypatch.setattr(settings, "get", lambda key, fallback=None: settings.DEFAULTS.get(key, fallback))

        props = [
            "CANDIDATE_MODELS_LIST", "CANDIDATE_VISION_MODELS_LIST", "CANDIDATE_MULTIMODAL_MODELS_LIST",
            "VLM_OLLAMA_MODELS", "JUDGE_MODELS", "EMBED_TEXT_MODEL", "TEXT_EMBEDDING_MODEL",
            "IMAGE_EMBEDDING_MODEL", "MULTIMODAL_EMBEDDING_MODEL", "EMBED_MODEL", "EMBED_PROVIDER",
            "EMBED_DEVICE", "MAX_TOKENS_DEFAULT", "TEMPERATURE_DEFAULT", "BANDIT_EPSILON",
            "QUERY_LOG_RETENTION_DAYS", "REDIS_HOST", "REDIS_PORT", "REDIS_DB", "REDIS_PASSWORD",
            "DB_HOST", "DB_PORT", "DB_USER", "DB_PASS", "DB_NAME", "ADMIN_TOKEN", "ADMIN_TOKEN_PREVIOUS",
            "JUDGES_ENABLED", "JUDGES_MODE", "JUDGES_LOCAL_MODEL", "JUDGES_REMOTE_MODEL", "JUDGES_TIMEOUT_S",
            "JUDGE_MIN_SAMPLE_RATE", "OLLAMA_HOST", "OLLAMA_BASE_URL", "CENTROIDS_DIM", "CENTROIDS_K",
            "CENTROIDS_MIN_SIM_CREATE", "CENTROIDS_ENABLE_ONLINE", "CENTROIDS_UPDATE_INTERVAL_S",
            "CENTROIDS_MIN_RECORDS_FOR_TRAIN", "CENTROIDS_MAX_HISTORY", "CENTROIDS_HOURLY_REFRESH_ENABLED",
            "CENTROIDS_MIN_LOG_ROWS_FOR_REFRESH", "NSGA_UPDATE_INTERVAL_S", "NSGA_LOOKBACK_MINUTES",
            "NSGA_LOOKBACK_MAXROWS", "METAOPT_REPS", "METAOPT_TRIALS", "NSGA_W_QUALITY", "NSGA_W_LATENCY",
            "NSGA_W_COST", "NSGA_W_ALIGNMENT", "NSGA_CONVERGENCE_HISTORY_SIZE", "CASCADE_WARNING_THRESHOLD",
            "CASCADE_CRITICAL_THRESHOLD", "RISK_FACTOR_SOTA_HIGH_UQ", "RISK_FACTOR_LOCAL_HIGH_UQ",
            "RISK_FACTOR_LOCAL_LOW_UQ", "RISK_FACTOR_ADAPT_ENABLED", "RISK_FACTOR_ADAPT_RATE",
            "ADAPTIVE_TIMEOUT_ENABLED", "ADAPTIVE_TIMEOUT_MULTIPLIER", "ADAPTIVE_TIMEOUT_REASONING_MULTIPLIER",
            "MIN_TIMEOUT", "MAX_TIMEOUT", "META_OPT_ENABLED", "META_OPT_SCHEDULE_HOUR",
            "META_OPT_SCHEDULED_TRIALS", "DRIFT_THRESHOLD", "DRIFT_WINDOW_SIZE", "USER_FEEDBACK_WEIGHT",
            "AB_TESTING_ENABLED", "CACHE_THRESHOLD_MIN", "CACHE_THRESHOLD_MAX", "CACHE_HIT_RATE_TARGET",
            "CACHE_THRESHOLD_ADAPT_ENABLED", "PREDICTOR_VALIDATION_ENABLED", "PREDICTOR_BRIER_SCORE_THRESHOLD",
            "PREDICTOR_CALIBRATION_WINDOW", "UQ_CALIBRATION_ENABLED", "UQ_QUALITY_GAP_RELAX",
            "UQ_QUALITY_GAP_TIGHTEN", "JUDGE_CALIBRATION_ENABLED", "JUDGE_CACHE_AGREEMENT_TARGET",
            "CIRCUIT_BREAKER_FAIL_MAX", "CIRCUIT_BREAKER_RESET_TIMEOUT", "CIRCUIT_BREAKER_LOCAL_FAIL_MAX",
            "CIRCUIT_BREAKER_LOCAL_RESET_TIMEOUT", "MAX_CONCURRENT_REQUESTS", "BACKPRESSURE_ENABLED",
            "EMERGENCY_FALLBACK_MODELS",
        ]
        for prop in props:
            getattr(settings, prop)
