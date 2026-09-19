# Objective: Test coverage for the ChromaDB incompatible-collection cleanup used at prestart.
"""reset_incompatible_collections removes only collections whose embedding dimension differs."""

from types import SimpleNamespace

from app import reset_chroma_collections as rcc


class _Collection(SimpleNamespace):
    def peek(self):
        if isinstance(self.sample, Exception):
            raise self.sample
        return self.sample


class _Client:
    def __init__(self, collections):
        self.collections = collections
        self.deleted = []

    def list_collections(self):
        return self.collections

    def delete_collection(self, name):
        if name == "protegida":
            raise PermissionError("read-only")
        self.deleted.append(name)


def test_removes_only_incompatible_collections(monkeypatch, capsys):
    client = _Client(
        [
            _Collection(name="meta_ok", metadata={"dimension": 1024}, sample=None),
            _Collection(name="meta_velha", metadata={"embedding_dimension": 768}, sample=None),
            _Collection(name="inferida", metadata=None, sample={"embeddings": [[0.1] * 384]}),
            _Collection(name="vazia", metadata={}, sample={"embeddings": []}),
            _Collection(name="peek_falha", metadata={}, sample=RuntimeError("boom")),
            _Collection(name="protegida", metadata={"dimension": 3}, sample=None),
        ]
    )
    monkeypatch.setattr(rcc, "PersistentClient", lambda path: client)

    out = rcc.reset_incompatible_collections("/tmp/chroma", expected_dim=1024)

    assert client.deleted == ["meta_velha", "inferida"]
    assert out["removed"] == [("meta_velha", 768), ("inferida", 384)]
    assert out["kept"] == [("meta_ok", 1024), ("vazia", None), ("peek_falha", None)]
    assert (out["total_removed"], out["total_kept"]) == (2, 3)
    printed = capsys.readouterr().out
    assert "[WARN] Falha ao inspecionar protegida" in printed and "Total removido: 2" in printed


def test_nothing_to_remove(monkeypatch, capsys):
    monkeypatch.setattr(rcc, "PersistentClient", lambda path: _Client([]))
    assert rcc.reset_incompatible_collections()["total_removed"] == 0
    assert "Nenhuma coleção incompatível" in capsys.readouterr().out
