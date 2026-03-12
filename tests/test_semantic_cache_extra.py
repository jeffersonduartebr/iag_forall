# Objective: Test coverage for semantic cache extra behavior and regressions.
"""Test coverage for semantic cache extra behavior and regressions.

This test module verifies expected behavior, regression boundaries, and failure
handling for the corresponding runtime component.
"""


from types import SimpleNamespace
import threading

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


@pytest.mark.asyncio
async def test_make_embedding_and_normalize_paths(monkeypatch):
    """Testa make embedding and normalize paths."""
    monkeypatch.setattr(sc, "embed_text", lambda q: [len(q)])
    monkeypatch.setattr(sc, "embed_image", lambda b64: [len(b64)])
    monkeypatch.setattr(sc, "embed_multimodal", lambda q, b: {"multimodal": [9], "text": [8]})

    assert sc._normalize_modality("image") == "vision"
    assert sc._normalize_modality("multimodal") == "multimodal"
    assert sc._normalize_modality("any") == "text"

    assert await sc._make_embedding("abc", "text", None) == [3]
    assert await sc._make_embedding("q", "vision", "img") == [3]
    assert await sc._make_embedding("q", "vision", None) == [1]
    assert await sc._make_embedding("q", "multimodal", "img") == [9]

    monkeypatch.setattr(sc, "embed_text", lambda q: (_ for _ in ()).throw(RuntimeError("x")))
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
