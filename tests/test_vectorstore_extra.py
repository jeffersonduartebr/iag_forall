"""Módulo `tests/test_vectorstore_extra.py`: descreve responsabilidades e integrações deste arquivo."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from app import vectorstore as vs


def test_vectorstore_helpers_and_collection_name(monkeypatch):
    """Testa vectorstore helpers and collection name."""
    assert vs._sanitize_model_name("nomic-ai/nomic-embed-text-v1.5") == "nomic_ai_nomic_embed_text_v1_5"
    assert vs._sanitize_model_name("") == "default"

    assert vs._normalize_modality(None) == "text"
    assert vs._normalize_modality("image") == "vision"
    assert vs._normalize_modality("unknown") == "text"
    assert vs._safe_metadata(None) == {"source": "router"}
    assert vs._safe_metadata({"a": 1}) == {"a": 1}

    assert vs._collection_for_modality("cache").startswith(vs.BASE_CACHE_COLLECTION)
    assert vs._collection_for_modality("text").startswith(vs.BASE_TEXT_COLLECTION)
    assert vs._collection_for_modality("vision").startswith(vs.BASE_IMAGE_COLLECTION)
    assert vs._collection_for_modality("multimodal").startswith(vs.BASE_MULTIMODAL_COLLECTION)

    assert vs._ensure_list_of_floats(np.array([1, 2])) == [1.0, 2.0]
    assert vs._ensure_list_of_floats([1, 2, 3]) == [1.0, 2.0, 3.0]
    assert vs._ensure_list_of_floats("x") == [0.0]


@pytest.mark.asyncio
async def test_get_or_create_collection_async_with_versioning(monkeypatch):
    """Testa get or create collection async with versioning."""
    called = []

    def _fake(name, metadata=None):
        """Executa fake."""
        called.append((name, metadata))
        return {"name": name}

    monkeypatch.setattr(vs, "_get_or_create_sync", _fake)

    out1 = await vs.get_or_create_collection_async(vs.BASE_TEXT_COLLECTION, {"m": 1})
    out2 = await vs.get_or_create_collection_async("custom_collection", None)

    assert out1["name"].startswith(vs.BASE_TEXT_COLLECTION)
    assert out2["name"] == "custom_collection"
    assert called


def test_insert_embedding_sync_and_query_sync_paths(monkeypatch):
    """Testa insert embedding sync and query sync paths."""
    class _ColOK:
        """Classe `_ColOK`: concentra responsabilidades de test vectorstore extra."""
        def __init__(self):
            """Inicializa estado interno necessário para uso da classe."""
            self.add_calls = 0

        def add(self, **kwargs):
            """Executa add."""
            self.add_calls += 1

        def query(self, **kwargs):
            """Executa query."""
            return {"ids": [["1"]], "documents": [["doc"]], "distances": [[0.1]]}

    class _ColDimFail(_ColOK):
        """Classe `_ColDimFail`: concentra responsabilidades de test vectorstore extra."""
        def add(self, **kwargs):
            """Executa add."""
            raise RuntimeError("dimension does not match")

    class _ColOtherFail(_ColOK):
        """Classe `_ColOtherFail`: concentra responsabilidades de test vectorstore extra."""
        def add(self, **kwargs):
            """Executa add."""
            raise RuntimeError("any other error")

    class _Client:
        """Classe `_Client`: concentra responsabilidades de test vectorstore extra."""
        def __init__(self, col):
            """Inicializa estado interno necessário para uso da classe."""
            self.col = col
            self.deleted = []
            self.created = []

        def get_or_create_collection(self, name, metadata=None):
            """Obtém or create collection."""
            return self.col

        def delete_collection(self, name):
            """Remove collection."""
            self.deleted.append(name)

        def create_collection(self, name):
            """Cria collection."""
            self.created.append(name)
            return _ColOK()

    ok_client = _Client(_ColOK())
    monkeypatch.setattr(vs, "chroma_client", ok_client)
    vs._insert_embedding_sync("c1", "d1", "txt", [1, 2], {"a": 1})
    assert ok_client.col.add_calls == 1

    dim_client = _Client(_ColDimFail())
    monkeypatch.setattr(vs, "chroma_client", dim_client)
    vs._insert_embedding_sync("c2", "d2", "txt", [1, 2], {"a": 1})
    assert dim_client.deleted == ["c2"]
    assert dim_client.created == ["c2"]

    other_client = _Client(_ColOtherFail())
    monkeypatch.setattr(vs, "chroma_client", other_client)
    vs._insert_embedding_sync("c3", "d3", "txt", [1, 2], {"a": 1})

    class _QDimFail(_ColOK):
        """Classe `_QDimFail`: concentra responsabilidades de test vectorstore extra."""
        def query(self, **kwargs):
            """Executa query."""
            raise RuntimeError("dimension does not match")

    qdim_client = _Client(_QDimFail())
    monkeypatch.setattr(vs, "chroma_client", qdim_client)
    assert vs._query_embedding_sync("cq", [1, 2], 3) == {}
    assert qdim_client.deleted == ["cq"]

    class _QOtherFail(_ColOK):
        """Classe `_QOtherFail`: concentra responsabilidades de test vectorstore extra."""
        def query(self, **kwargs):
            """Executa query."""
            raise RuntimeError("boom")

    qother_client = _Client(_QOtherFail())
    monkeypatch.setattr(vs, "chroma_client", qother_client)
    assert vs._query_embedding_sync("cq2", [1, 2], 3) == {}


@pytest.mark.asyncio
async def test_add_query_reset_and_health(monkeypatch):
    """Testa add query reset and health."""
    inserts = []
    queries = []
    sparse_added = []
    committed = {"n": 0}

    monkeypatch.setattr(vs, "embed_text", lambda txt: [0.1, 0.2])
    monkeypatch.setattr(vs, "embed_image", lambda img: [0.3, 0.4])
    monkeypatch.setattr(vs, "embed_multimodal", lambda txt, img: {"multimodal": [0.5, 0.6]})
    monkeypatch.setattr(vs, "_insert_embedding_sync", lambda *a, **k: inserts.append((a, k)))
    monkeypatch.setattr(vs.sparse_index, "add_document", lambda did, txt: sparse_added.append((did, txt)))
    monkeypatch.setattr(vs.sparse_index, "commit", lambda: committed.__setitem__("n", committed["n"] + 1))

    assert await vs.add_document("text", "d1", text="abc", metadata={"x": 1}) is True
    assert await vs.add_document("vision", "d2", text="abc", image_b64="img") is True
    assert await vs.add_document("multimodal", "d3", text="abc", image_b64="img") is True
    assert len(inserts) == 3
    assert len(sparse_added) == 2  # text + multimodal
    assert committed["n"] == 2

    monkeypatch.setattr(vs, "_query_embedding_sync", lambda *a, **k: {"ok": True, "name": a[0]})
    res_known = await vs.query_embedding("text", [0.1], 2)
    res_custom = await vs.query_embedding("knowledge_base", [0.1], 2)
    assert res_known["ok"] is True
    assert res_custom["name"] == "knowledge_base"

    class _Client:
        """Classe `_Client`: concentra responsabilidades de test vectorstore extra."""
        def reset(self):
            """Executa reset."""
            return None

        def heartbeat(self):
            """Executa heartbeat."""
            return "ok"

    monkeypatch.setattr(vs, "chroma_client", _Client())
    await vs.reset_collections()
    assert await vs.health_async() is True

    class _BadClient:
        """Classe `_BadClient`: concentra responsabilidades de test vectorstore extra."""
        def heartbeat(self):
            """Executa heartbeat."""
            raise RuntimeError("down")

        def reset(self):
            """Executa reset."""
            raise RuntimeError("nope")

    monkeypatch.setattr(vs, "chroma_client", _BadClient())
    await vs.reset_collections()
    assert await vs.health_async() is False
