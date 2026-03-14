# Objective: Test coverage for sparse index more behavior and regressions.
"""Additional branch coverage for sparse_index."""

from app import sparse_index as si


def test_sparse_index_commit_search_save_and_load_paths(monkeypatch, tmp_path):
    """SparseIndex should cover save/load success and empty/error branches."""
    monkeypatch.setattr(si, "INDEX_PATH", str(tmp_path / "bm25.pkl"))
    monkeypatch.setattr(si.os.path, "exists", lambda path: False)
    idx = si.SparseIndex()

    idx.add_document("d1", "texto um")
    idx.add_document("d2", "texto dois")
    idx.commit()
    assert idx.bm25 is not None
    assert idx.is_dirty is False
    idx.bm25 = type("_BM25", (), {"get_scores": lambda self, query: [1.0, 0.0]})()
    assert idx.search("um", top_k=1) == [("d1", 1.0)]

    idx.documents = []
    idx.is_dirty = True
    idx.commit()
    assert idx.is_dirty is True

def test_sparse_index_handles_save_load_and_search_failures(monkeypatch):
    """SparseIndex should swallow disk and BM25 failures without crashing callers."""
    monkeypatch.setattr(si.os.path, "exists", lambda path: False)
    idx = si.SparseIndex()

    monkeypatch.setattr(si.os, "makedirs", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("mkdir fail")))
    idx._save()

    class _BrokenBm25:
        def get_scores(self, _query):
            raise RuntimeError("bm25 fail")

    idx.documents = ["doc"]
    idx.doc_ids = ["d1"]
    idx.bm25 = _BrokenBm25()
    assert idx.search("query") == []

    monkeypatch.setattr(si.os.path, "exists", lambda path: True)

    class _Ctx:
        def __enter__(self):
            raise RuntimeError("bad file")

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(si, "open", lambda *a, **k: _Ctx(), raising=False)
    idx._load()


def test_sparse_index_load_success(monkeypatch):
    """SparseIndex load should restore documents when pickle payload is valid."""
    monkeypatch.setattr(si.os.path, "exists", lambda path: True)

    class _Ctx:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(si, "open", lambda *a, **k: _Ctx(), raising=False)
    monkeypatch.setattr(si.pickle, "load", lambda handle: {"docs": ["doc"], "ids": ["d1"], "bm25": "bm25"})
    idx = si.SparseIndex()
    assert idx.documents == ["doc"]
    assert idx.doc_ids == ["d1"]
    assert idx.bm25 == "bm25"
