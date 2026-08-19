# Objective: Test coverage for semantic cache extra behavior and regressions.
"""Test coverage for semantic cache extra behavior and regressions.

This test module verifies expected behavior, regression boundaries, and failure
handling for the corresponding runtime component.
"""


import threading
from types import SimpleNamespace

import pytest

from app import semantic_cache as sc


class _Metric:
    def __init__(self):
        self.values = []

    def labels(self, **kwargs):
        self.values.append(("labels", kwargs))
        return self

    def inc(self, value=1):
        self.values.append(("inc", value))

    def set(self, value):
        self.values.append(("set", value))

    def observe(self, value):
        self.values.append(("observe", value))


def test_normalize_query_casefolds_and_collapses_whitespace():
    """Enabled normalization canonicalizes case and whitespace (perf #24)."""
    assert sc._normalize_query("What is  2+2?") == sc._normalize_query("what is 2+2?")
    assert sc._normalize_query("  Hello   WORLD  ") == "hello world"


def test_normalize_query_disabled_is_identity(monkeypatch):
    """With the flag off, the query is returned unchanged."""
    monkeypatch.setattr(sc.settings, "get", lambda key, default=None: "0")
    assert sc._normalize_query("What is  2+2?") == "What is  2+2?"


@pytest.mark.asyncio
async def test_query_normalization_yields_l1_hit_on_surface_variant(monkeypatch):
    """A stored answer is served from L1 for a case/whitespace variant (perf #24)."""
    async def _noop_add(**kwargs):
        return None

    monkeypatch.setattr(sc, "add_document", _noop_add)
    sc._l1_cache._cache.clear()

    await sc.store_cache("What is  the Capital of France?", "Paris", modality="text")
    hit = await sc.check_cache("what is the capital of france?", modality="text")
    assert hit is not None
    assert hit["text"] == "Paris"


@pytest.mark.asyncio
async def test_make_embedding_and_normalize_paths(monkeypatch):
    """Testa make embedding and normalize paths."""
    async def _atext(q):
        return [len(q)]

    async def _aimage(b64):
        return [len(b64)]

    async def _amultimodal(q, b):
        return {"multimodal": [9], "text": [8]}

    monkeypatch.setattr(sc, "aembed_text", _atext)
    monkeypatch.setattr(sc, "aembed_image", _aimage)
    monkeypatch.setattr(sc, "aembed_multimodal", _amultimodal)

    assert sc._normalize_modality("image") == "vision"
    assert sc._normalize_modality("multimodal") == "multimodal"
    assert sc._normalize_modality("any") == "text"

    assert await sc._make_embedding("abc", "text", None) == [3]
    assert await sc._make_embedding("q", "vision", "img") == [3]
    assert await sc._make_embedding("q", "vision", None) == [1]
    assert await sc._make_embedding("q", "multimodal", "img") == [9]

    async def _athrow(q):
        raise RuntimeError("x")

    monkeypatch.setattr(sc, "aembed_text", _athrow)
    assert await sc._make_embedding("abc", "text", None) is None


@pytest.mark.asyncio
async def test_check_cache_branches(monkeypatch):
    """Testa check cache branches."""
    metric_lookup = _Metric()
    metric_latency = _Metric()
    metric_l1_hits = _Metric()
    metric_l1_misses = _Metric()
    metric_l1_size = _Metric()
    monkeypatch.setattr(sc, "SEMANTIC_CACHE_LOOKUP_TOTAL", metric_lookup)
    monkeypatch.setattr(sc, "SEMANTIC_CACHE_LATENCY", metric_latency)
    monkeypatch.setattr(sc, "L1_CACHE_HITS", metric_l1_hits)
    monkeypatch.setattr(sc, "L1_CACHE_MISSES", metric_l1_misses)
    monkeypatch.setattr(sc, "L1_CACHE_SIZE", metric_l1_size)

    class _L1:
        """Represent `_L1` within this module.

The class groups the state and behavior required for L1."""
        def __init__(self):
            """Initialize the internal state required by this instance.

The constructor keeps setup local to the object so callers can use it without additional bootstrapping."""
            self.saved = []

        def get(self, key):
            """Execute the get routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            return None

        def store(self, key, value):
            """Execute the store routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            self.saved.append((key, value))

    l1 = _L1()
    monkeypatch.setattr(sc, "_l1_cache", l1)
    async def _emb(*args, **kwargs):
        """Execute the emb routine.

This helper encapsulates one focused step used by the surrounding workflow."""
        return [0.1, 0.2]

    monkeypatch.setattr(sc, "_make_embedding", _emb)

    # no docs
    async def _nodoc(**kwargs):
        """Execute the nodoc routine.

This helper encapsulates one focused step used by the surrounding workflow."""
        return {"documents": []}

    monkeypatch.setattr(sc, "query_embedding", _nodoc)
    assert await sc.check_cache("q") is None

    async def _empty_nested(**kwargs):
        """Execute the empty nested routine.

This helper encapsulates one focused step used by the surrounding workflow."""
        return {"documents": [[]], "distances": [[]], "metadatas": [[]]}

    monkeypatch.setattr(sc, "query_embedding", _empty_nested)
    assert await sc.check_cache("q") is None

    # similarity below threshold
    fake_settings = SimpleNamespace(get=lambda k, d=None: 0.95 if k == "CACHE_THRESHOLD" else d)
    monkeypatch.setattr(sc, "settings", fake_settings)

    async def _low(**kwargs):
        """Execute the low routine.

This helper encapsulates one focused step used by the surrounding workflow."""
        return {"documents": [["q"]], "distances": [[0.4]], "metadatas": [[{"answer_payload": "a"}]]}

    monkeypatch.setattr(sc, "query_embedding", _low)
    assert await sc.check_cache("q") is None

    # no answer payload
    async def _nopayload(**kwargs):
        """Execute the nopayload routine.

This helper encapsulates one focused step used by the surrounding workflow."""
        return {"documents": [["q"]], "distances": [[0.01]], "metadatas": [[{"model_used": "m"}]]}

    monkeypatch.setattr(sc, "query_embedding", _nopayload)
    assert await sc.check_cache("q") is None

    # success
    async def _ok(**kwargs):
        """Execute the ok routine.

This helper encapsulates one focused step used by the surrounding workflow."""
        return {
            "documents": [["q"]],
            "distances": [[0.01]],
            "metadatas": [[{"model_used": "m", "answer_payload": "resp", "image_output_b64": "img"}]],
        }

    monkeypatch.setattr(sc, "query_embedding", _ok)
    out = await sc.check_cache("q")
    assert out["text"] == "resp"
    assert out["model_used"] == "m"
    assert l1.saved
    assert any(item[0] == "labels" and item[1]["result"] == "below_threshold" for item in metric_lookup.values)
    assert any(item[0] == "labels" and item[1]["result"] == "empty_result" for item in metric_lookup.values)
    assert any(item[0] == "observe" for item in metric_latency.values)


@pytest.mark.asyncio
async def test_check_cache_handles_l1_and_metric_failures(monkeypatch):
    """Cache lookup should survive L1 hits and instrumentation failures without breaking the result."""
    class _BrokenMetric:
        def inc(self, *args, **kwargs):
            raise RuntimeError("metric fail")

        def set(self, *args, **kwargs):
            raise RuntimeError("metric fail")

        def labels(self, **kwargs):
            raise RuntimeError("metric fail")

        def observe(self, *args, **kwargs):
            raise RuntimeError("metric fail")

    class _L1:
        def get(self, key):
            return {"text": "cached", "model_used": "m"}

        def stats(self):
            return {"size": 1}

    monkeypatch.setattr(sc, "_l1_cache", _L1())
    monkeypatch.setattr(sc, "L1_CACHE_HITS", _BrokenMetric())
    monkeypatch.setattr(sc, "L1_CACHE_SIZE", _BrokenMetric())
    monkeypatch.setattr(sc, "SEMANTIC_CACHE_LOOKUP_TOTAL", _BrokenMetric())
    monkeypatch.setattr(sc, "SEMANTIC_CACHE_LATENCY", _BrokenMetric())

    out = await sc.check_cache("q")
    assert out["text"] == "cached"


def test_extract_first_result_guards_partial_payloads():
    """Testa extração defensiva do primeiro resultado do Chroma."""
    assert sc._extract_first_result({}) == (None, None)
    assert sc._extract_first_result({"documents": [[]], "distances": [[0.1]], "metadatas": [[{}]]}) == (None, None)
    assert sc._extract_first_result({"documents": [["q"]], "distances": [[]], "metadatas": [[{}]]}) == (None, None)
    assert sc._extract_first_result({"documents": [["q"]], "distances": [[0.1]], "metadatas": [[]]}) == (None, None)
    assert sc._extract_first_result({"documents": [["q"]], "distances": [[0.1]], "metadatas": [[{"a": 1}]]}) == (0.1, {"a": 1})


@pytest.mark.asyncio
async def test_store_cache_and_hit_rate_tuning(monkeypatch):
    """Testa store cache and hit rate tuning."""
    stored = []

    class _L1:
        """Represent `_L1` within this module.

The class groups the state and behavior required for L1."""
        def __init__(self):
            """Initialize the internal state required by this instance.

The constructor keeps setup local to the object so callers can use it without additional bootstrapping."""
            self._hits = 0
            self._misses = 0
            self._lock = threading.Lock()

        def store(self, key, value):
            """Execute the store routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            stored.append((key, value))

        def stats(self):
            """Execute the stats routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            return {"hits": self._hits, "misses": self._misses, "size": 0, "maxsize": 10}

    l1 = _L1()
    monkeypatch.setattr(sc, "_l1_cache", l1)

    async def _add_document(**kwargs):
        """Execute the add document routine.

This helper encapsulates one focused step used by the surrounding workflow."""
        return True

    monkeypatch.setattr(sc, "add_document", _add_document)
    monkeypatch.setattr("app.judges.update_calibration_cache_status", lambda q: None)

    await sc.store_cache("q1", "a1", modality="text", model_used="m1")
    assert stored

    # hit rate
    l1._hits = 3
    l1._misses = 4
    assert sc.get_cache_hit_rate() == -1.0
    l1._hits = 60
    l1._misses = 40
    assert sc.get_cache_hit_rate() == pytest.approx(0.6)

    fake_settings = SimpleNamespace(
        CACHE_THRESHOLD_ADAPT_ENABLED=False,
        get=lambda k, d=None: 0.92 if k == "CACHE_THRESHOLD" else d,
    )
    monkeypatch.setattr(sc, "settings", fake_settings)
    assert await sc.tune_cache_threshold() is None

    calls = []
    fake_settings2 = SimpleNamespace(
        CACHE_THRESHOLD_ADAPT_ENABLED=True,
        CACHE_HIT_RATE_TARGET=0.1,
        CACHE_THRESHOLD_MIN=0.7,
        CACHE_THRESHOLD_MAX=0.99,
        get=lambda k, d=None: 0.7 if k == "CACHE_THRESHOLD" else d,
        set=lambda *a, **k: calls.append((a, k)),
    )
    monkeypatch.setattr(sc, "settings", fake_settings2)
    l1._hits = 90
    l1._misses = 10
    new_val = await sc.tune_cache_threshold()
    assert new_val is not None
    assert calls

    # reset stats
    sc.reset_cache_stats()
    assert l1._hits == 0
    assert l1._misses == 0


@pytest.mark.asyncio
async def test_store_cache_handles_metric_and_judge_failures(monkeypatch):
    """Cache storage should keep working when metrics and judge calibration updates fail."""
    stored = []

    class _L1:
        def __init__(self):
            self.saved = []

        def store(self, key, value):
            self.saved.append((key, value))

        def stats(self):
            return {"hits": 0, "misses": 0, "size": 1, "maxsize": 10}

    class _BrokenMetric:
        def set(self, value):
            raise RuntimeError("metric down")

    async def _add_document(**kwargs):
        stored.append(kwargs)
        return True

    monkeypatch.setattr(sc, "_l1_cache", _L1())
    monkeypatch.setattr(sc, "L1_CACHE_SIZE", _BrokenMetric())
    monkeypatch.setattr(sc, "add_document", _add_document)

    import builtins
    real_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name.endswith(".judges") or name == "app.judges":
            raise RuntimeError("judge import fail")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)
    await sc.store_cache("q2", "a2", model_used="m2")
    assert stored and stored[0]["modality"] == "cache"


@pytest.mark.asyncio
async def test_store_cache_and_tuning_failure_paths(monkeypatch):
    """Storage and threshold tuning should degrade gracefully on downstream failures."""
    class _L1:
        def __init__(self):
            self._hits = 55
            self._misses = 45
            self._lock = threading.Lock()

        def store(self, key, value):
            return None

        def stats(self):
            return {"hits": self._hits, "misses": self._misses, "size": 0, "maxsize": 10}

    async def _broken_add_document(**kwargs):
        raise RuntimeError("persist fail")

    monkeypatch.setattr(sc, "_l1_cache", _L1())
    monkeypatch.setattr(sc, "add_document", _broken_add_document)
    await sc.store_cache("q3", "a3")

    fake_settings = SimpleNamespace(
        CACHE_THRESHOLD_ADAPT_ENABLED=True,
        CACHE_HIT_RATE_TARGET=0.5,
        CACHE_THRESHOLD_MIN=0.7,
        CACHE_THRESHOLD_MAX=0.99,
        get=lambda k, d=None: 0.7 if k == "CACHE_THRESHOLD" else d,
        set=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("set fail")),
    )
    monkeypatch.setattr(sc, "settings", fake_settings)
    assert await sc.tune_cache_threshold() is None

    fake_settings2 = SimpleNamespace(
        CACHE_THRESHOLD_ADAPT_ENABLED=True,
        CACHE_HIT_RATE_TARGET=0.5,
        CACHE_THRESHOLD_MIN=0.7,
        CACHE_THRESHOLD_MAX=0.99,
        get=lambda k, d=None: 0.8 if k == "CACHE_THRESHOLD" else d,
        set=lambda *a, **k: None,
    )
    monkeypatch.setattr(sc, "settings", fake_settings2)
    monkeypatch.setattr(sc, "get_cache_threshold", lambda: 0.8)
    monkeypatch.setattr(sc.logger, "debug", lambda *a, **k: None)
    sc._l1_cache._hits = 1
    sc._l1_cache._misses = 1
    assert await sc.tune_cache_threshold() is None
