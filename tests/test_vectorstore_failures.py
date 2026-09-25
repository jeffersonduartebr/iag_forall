# Objective: The RAG store must not claim success it did not have, nor destroy data on a read.
"""Failure paths of the vector store.

Two defects lived here. ``add_document`` returned ``True`` unconditionally, so
an ingest against a dead ChromaDB answered 200 and indexed nothing. And the
"auto-healing" for a dimension mismatch ran on the *query* path, so one user
question deleted the whole collection.
"""

import pytest

from app import vectorstore as vs


@pytest.mark.asyncio
async def test_add_document_reports_a_failed_insertion(monkeypatch):
    """A failed Chroma insert must not be reported as a successful ingest.

    ``add_document`` returned ``True`` unconditionally, which made the
    ``raise HTTPException(500)`` in the ingest endpoint dead code: an upload
    against a dead ChromaDB answered 200 and indexed nothing.
    """
    sparse_added = []
    monkeypatch.setattr(vs, "embed_text", lambda txt, *tarefa: [0.1, 0.2])
    monkeypatch.setattr(vs, "_insert_embedding_sync", lambda *a, **k: False)
    monkeypatch.setattr(vs.sparse_index, "add_document", lambda did, txt: sparse_added.append(did))

    assert await vs.add_document("text", "d9", text="abc") is False
    # O espelho BM25 é deliberadamente saltado: uma entrada encontrável por
    # palavra-chave sem vetor por trás parece um ingest bem-sucedido até
    # alguém procurar semanticamente.
    assert sparse_added == []


def test_insert_embedding_sync_returns_false_on_a_non_dimension_error(monkeypatch):
    """Any other Chroma error is a failure, not a silent no-op."""

    class _Boom:
        def get_or_create_collection(self, **kwargs):
            raise RuntimeError("chroma indisponível")

    monkeypatch.setattr(vs, "get_chroma_client", lambda: _Boom())
    assert vs._insert_embedding_sync("col", "d1", "t", [0.1], None) is False


def test_insert_embedding_sync_reports_a_failed_heal(monkeypatch):
    """When the dimension heal itself fails, the document is not stored."""

    class _AlwaysDimensionError:
        def get_or_create_collection(self, **kwargs):
            raise RuntimeError("dimension does not match")

        def delete_collection(self, name):
            raise RuntimeError("delete falhou")

    monkeypatch.setattr(vs, "get_chroma_client", lambda: _AlwaysDimensionError())
    assert vs._insert_embedding_sync("col", "d1", "t", [0.1], None) is False


def test_a_query_never_deletes_the_collection(monkeypatch):
    """A read must not destroy data.

    The dimension mismatch that triggered this used to call
    ``delete_collection`` from the *query* path, and its usual cause is the
    local embedding model failing to load, so ``embed_text`` returns a
    zero vector of the wrong size. One user question therefore wiped the whole
    RAG corpus, and the only warning came after the damage.
    """
    deleted = []

    class _DimensionMismatch:
        def get_or_create_collection(self, **kwargs):
            raise RuntimeError("Collection expecting embedding with dimension of 768, got 384")

        def delete_collection(self, name):
            deleted.append(name)

    monkeypatch.setattr(vs, "get_chroma_client", lambda: _DimensionMismatch())

    assert vs._query_embedding_sync("meu_corpus", [0.1, 0.2], 3) == {}
    assert deleted == [], "uma consulta apagou a coleção"
