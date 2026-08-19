# Objective: Test coverage for reranker module behavior and regressions.
"""Test coverage for reranker module behavior and regressions.

This test module verifies expected behavior, regression boundaries, and failure
handling for the corresponding runtime component.
"""


from app import reranker as rr


def test_get_reranker_model_and_rerank_paths(monkeypatch):
    """Testa get reranker model and rerank paths."""
    rr._RERANKER_INSTANCE = None
    monkeypatch.setattr(rr, "CE_AVAILABLE", False)
    assert rr.get_reranker_model() is None
    assert rr.rerank_documents("q", ["a", "b"], top_k=1) == ["a"]
    assert rr.rerank_documents("", ["a", "b"], top_k=1) == ["a"]

    rr._RERANKER_INSTANCE = None
    monkeypatch.setattr(rr, "CE_AVAILABLE", True)

    class _CE:
        """Represent `_CE` within this module.

The class groups the state and behavior required for CE."""
        def __init__(self, name, device="cpu"):
            """Initialize the internal state required by this instance.

The constructor keeps setup local to the object so callers can use it without additional bootstrapping."""
            self.name = name

        def predict(self, pairs):
            """Execute the predict routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            return [0.1, 0.9, 0.3][: len(pairs)]

    monkeypatch.setattr(rr, "CrossEncoder", _CE, raising=False)
    m = rr.get_reranker_model()
    assert m is not None

    docs = rr.rerank_documents("q", ["d1", "d2", "d3"], top_k=2)
    assert docs == ["d2", "d3"]

    class _BrokenCE(_CE):
        """Represent `_BrokenCE` within this module.

The class groups the state and behavior required for BrokenCE."""
        def predict(self, pairs):
            """Execute the predict routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            raise RuntimeError("x")

    rr._RERANKER_INSTANCE = _BrokenCE("x")
    assert rr.rerank_documents("q", ["d1", "d2"], top_k=1) == ["d1"]


def test_get_reranker_model_load_error(monkeypatch):
    """Testa get reranker model load error."""
    rr._RERANKER_INSTANCE = None
    monkeypatch.setattr(rr, "CE_AVAILABLE", True)
    monkeypatch.setattr(rr, "CrossEncoder", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("load fail")), raising=False)
    assert rr.get_reranker_model() is None
