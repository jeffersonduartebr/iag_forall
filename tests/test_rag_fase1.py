# Objective: RAG phase 1 (2026-09-25): the corpus survives embedding failures, ingest is idempotent, the
# healthcheck stays out of the corpus, and model loads are serialized with backoff.
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app import embeddings, reranker
from app import vectorstore as vs


class _Col:
    def __init__(self, erro=None):
        self.upserts, self.erro = [], erro

    def upsert(self, **kwargs):
        if self.erro:
            raise RuntimeError(self.erro)
        self.upserts.append(kwargs)


def _cliente(monkeypatch, col):
    cliente = SimpleNamespace(
        criadas=[], apagadas=[], get_or_create_collection=None, delete_collection=lambda n: cliente.apagadas.append(n)
    )

    def _goc(name, metadata=None):
        cliente.criadas.append((name, metadata))
        return col

    cliente.get_or_create_collection = _goc
    monkeypatch.setattr(vs, "chroma_client", cliente)
    return cliente


@pytest.mark.parametrize("vetor", [[0.0], [0.0] * 768, []])
def test_failed_embeddings_never_reach_chroma(monkeypatch, vetor):
    col = _Col()
    _cliente(monkeypatch, col)
    assert vs._insert_embedding_sync("text_embeddings_x_d2", "d", "t", vetor, {}) is False
    assert col.upserts == []


def test_a_dimension_mismatch_reports_and_never_deletes_the_collection(monkeypatch):
    cliente = _cliente(monkeypatch, _Col(erro="Embedding dimension 1536 does not match collection dimensionality 768"))
    assert vs._insert_embedding_sync("text_embeddings_x_d2", "d", "t", [0.1, 0.2], {}) is False
    assert cliente.apagadas == []


def test_ingest_is_an_upsert_into_a_cosine_collection(monkeypatch):
    col = _Col()
    cliente = _cliente(monkeypatch, col)
    nome = vs._get_versioned_collection_name(vs.BASE_TEXT_COLLECTION, "text")
    assert nome.endswith("_d2") and not vs._get_versioned_collection_name(vs.BASE_CACHE_COLLECTION, "cache").endswith(
        "_d2"
    )
    assert vs._insert_embedding_sync(nome, "tenant:doc:0", "t", [0.1, 0.2], {"k": 1}) is True
    assert col.upserts[0]["ids"] == ["tenant:doc:0"] and cliente.criadas[0] == (nome, {"hnsw:space": "cosine"})


@pytest.mark.asyncio
async def test_documents_are_embedded_as_documents_and_the_healthcheck_stays_out_of_the_corpus(monkeypatch):
    tarefas, gravados, bm25 = [], [], []
    monkeypatch.setattr(vs, "embed_text", lambda texto, tarefa="consulta": tarefas.append(tarefa) or [0.1, 0.2])
    monkeypatch.setattr(vs, "_insert_embedding_sync", lambda nome, *a: gravados.append(nome) or True)
    monkeypatch.setattr(vs.sparse_index, "add_document", lambda *a: bm25.append(a))
    monkeypatch.setattr(vs.sparse_index, "commit", lambda: None)
    assert await vs.add_document("text", "d1", text="capítulo 1")
    assert await vs.add_document("cache", "q1", text="pergunta")
    assert await vs.add_document("text", "hc", text="verificação", colecao="rag_healthcheck")
    assert tarefas == ["documento", "consulta", "documento"]
    assert gravados[-1] == "rag_healthcheck" and len(bm25) == 1  # só o documento do corpus entra no BM25


@pytest.mark.parametrize("modulo", [embeddings, reranker])
def test_a_failed_model_load_is_not_retried_on_every_request(monkeypatch, modulo):
    chamadas = []

    def _falha(*a, **k):
        chamadas.append(1)
        raise OSError("sem rede para baixar o modelo")

    relogio = [1000.0]
    monkeypatch.setattr(modulo.time, "monotonic", lambda: relogio[0])
    if modulo is embeddings:
        monkeypatch.setattr(modulo, "ST_AVAILABLE", True)
        monkeypatch.setattr(modulo, "SentenceTransformer", _falha, raising=False)
        monkeypatch.setattr(modulo, "_LOCAL_MODEL_INSTANCE", None)
        carregar = modulo.get_local_model
    else:
        monkeypatch.setattr(modulo, "CE_AVAILABLE", True)
        monkeypatch.setattr(modulo, "CrossEncoder", _falha, raising=False)
        monkeypatch.setattr(modulo, "_RERANKER_INSTANCE", None)
        carregar = modulo.get_reranker_model
    monkeypatch.setattr(modulo, "_LOAD_FAILED_AT", 0.0)
    assert carregar() is None and carregar() is None and len(chamadas) == 1
    relogio[0] += modulo.LOAD_RETRY_S
    assert carregar() is None and len(chamadas) == 2


def test_the_default_reranker_is_multilingual():
    assert "mmarco" in reranker.DEFAULT_RERANK_MODEL  # o ms-marco inglês reordenava mal o acervo em português
