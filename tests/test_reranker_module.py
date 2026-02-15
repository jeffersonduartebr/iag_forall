from app import reranker as rr


def test_get_reranker_model_and_rerank_paths(monkeypatch):
    rr._RERANKER_INSTANCE = None
    monkeypatch.setattr(rr, "CE_AVAILABLE", False)
    assert rr.get_reranker_model() is None
    assert rr.rerank_documents("q", ["a", "b"], top_k=1) == ["a"]
    assert rr.rerank_documents("", ["a", "b"], top_k=1) == ["a"]

    rr._RERANKER_INSTANCE = None
    monkeypatch.setattr(rr, "CE_AVAILABLE", True)

    class _CE:
        def __init__(self, name, device="cpu"):
            self.name = name

        def predict(self, pairs):
            return [0.1, 0.9, 0.3][: len(pairs)]

    monkeypatch.setattr(rr, "CrossEncoder", _CE, raising=False)
    m = rr.get_reranker_model()
    assert m is not None

    docs = rr.rerank_documents("q", ["d1", "d2", "d3"], top_k=2)
    assert docs == ["d2", "d3"]

    class _BrokenCE(_CE):
        def predict(self, pairs):
            raise RuntimeError("x")

    rr._RERANKER_INSTANCE = _BrokenCE("x")
    assert rr.rerank_documents("q", ["d1", "d2"], top_k=1) == ["d1"]


def test_get_reranker_model_load_error(monkeypatch):
    rr._RERANKER_INSTANCE = None
    monkeypatch.setattr(rr, "CE_AVAILABLE", True)
    monkeypatch.setattr(rr, "CrossEncoder", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("load fail")), raising=False)
    assert rr.get_reranker_model() is None
