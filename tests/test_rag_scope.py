# Objective: Test coverage for scoped RAG (rag_filter, tenant stamping, scoped deletion).
"""Scoped RAG: one discipline's corpus never answers another's query, and a scope is honoured exactly."""

from types import SimpleNamespace

import pytest
from app.api.auth import AuthContext
from app.routers import rag_router as rr
from app.schemas import QueryRequest
from app.services import query_profile as qp
from app.services import rag_scope
from fastapi import HTTPException
from pydantic import ValidationError


class _FakeCollection:
    """Minimal Chroma collection: flat equality / ``$and`` filtering over stored metadata."""

    def __init__(self, docs):
        self.docs = dict(docs)
        self.deleted = []

    @staticmethod
    def _match(meta, where):
        clauses = where.get("$and", [where])
        return all(meta.get(k) == v for clause in clauses for k, v in clause.items())

    def get(self, ids=None, where=None, include=None):
        pool = ids if ids is not None else list(self.docs)
        return {"ids": [i for i in pool if i in self.docs and self._match(self.docs[i], where or {})]}

    def delete(self, ids):
        self.deleted.extend(ids)
        for i in ids:
            self.docs.pop(i, None)


@pytest.fixture
def collection(monkeypatch):
    col = _FakeCollection({
        "t:bd-1": {"tenant_id": "t", "disciplina": "bd"},
        "t:mat-1": {"tenant_id": "t", "disciplina": "mat"},
        "u:bd-1": {"tenant_id": "u", "disciplina": "bd"},
    })
    client = SimpleNamespace(get_or_create_collection=lambda name: col)
    monkeypatch.setattr(rag_scope, "get_chroma_client", lambda: client)
    return col


def test_scope_where_is_none_without_filter_and_tenant_always_wins():
    assert rag_scope.scope_where(None, "t") is None
    assert rag_scope.scope_where({"disciplina": "bd"}, None) == {"disciplina": "bd"}
    assert rag_scope.scope_where({"disciplina": "bd", "tenant_id": "forged"}, "t") == {
        "$and": [{"disciplina": "bd"}, {"tenant_id": "t"}]
    }


@pytest.mark.asyncio
async def test_restrict_to_scope_drops_bm25_hits_from_other_scopes(collection):
    hits = (["t:mat-1", "t:bd-1", "u:bd-1"], {"t:bd-1": "x", "t:mat-1": "y"}, {"t:bd-1": {}})
    where = rag_scope.scope_where({"disciplina": "bd"}, "t")
    ids, docs, metas = await rag_scope.restrict_to_scope(hits, where, "text")
    assert (ids, docs, metas) == (["t:bd-1"], {"t:bd-1": "x"}, {"t:bd-1": {}})
    assert await rag_scope.restrict_to_scope(hits, None, "text") is hits


@pytest.mark.asyncio
async def test_restrict_to_scope_fails_closed(monkeypatch):
    def _boom():
        raise RuntimeError("chroma down")

    monkeypatch.setattr(rag_scope, "get_chroma_client", _boom)
    assert await rag_scope.restrict_to_scope((["a"], {"a": "x"}, {}), {"k": 1}, "text") == ([], {}, {})


@pytest.mark.asyncio
async def test_delete_scope_removes_chunks_from_chroma_and_bm25(collection, monkeypatch):
    removed_sparse = []
    monkeypatch.setattr(rag_scope.sparse_index, "remove_documents", removed_sparse.extend)
    removed = await rag_scope.delete_scope(rag_scope.scope_where({"disciplina": "bd"}, "t"))
    assert removed == removed_sparse == ["t:bd-1"]
    assert "u:bd-1" in collection.docs
    assert await rag_scope.delete_scope({"disciplina": "nada"}) == []


def test_rag_filter_rejects_operator_keys_and_oversized_scopes():
    QueryRequest(query="q", rag_filter={"disciplina": "bd", "periodo": "2026.2"})
    with pytest.raises(ValidationError):
        QueryRequest(query="q", rag_filter={"$and": "x"})
    with pytest.raises(ValidationError):
        QueryRequest(query="q", rag_filter={f"k{i}": i for i in range(9)})


def _req(**overrides):
    values = {"rag_filter": None, "enable_rag_for_answer": False, "enable_rag_for_image": False}
    values.update(overrides)
    return SimpleNamespace(**values)


def test_scope_forces_requested_retrieval_even_for_short_queries():
    hints = {"retrieval_mode": "no_retrieval", "needs_retrieval": False}
    assert qp._apply_rag_scope(_req(rag_filter={"d": "bd"}, enable_rag_for_answer=True), hints, False) is True
    assert hints == {"retrieval_mode": "full_retrieval", "needs_retrieval": True, "rag_filter": {"d": "bd"}}


def test_scope_without_rag_request_and_no_scope_leave_profile_alone():
    hints = {"retrieval_mode": "no_retrieval"}
    assert qp._apply_rag_scope(_req(rag_filter={"d": "bd"}), hints, False) is False
    assert hints == {"retrieval_mode": "no_retrieval", "rag_filter": {"d": "bd"}}
    untouched = {"retrieval_mode": "no_retrieval"}
    assert qp._apply_rag_scope(_req(enable_rag_for_answer=True), untouched, False) is False
    assert untouched == {"retrieval_mode": "no_retrieval"}


def test_ingest_stamps_caller_tenant_and_namespaces_the_id():
    req = rr.IngestRequest(text="t", doc_id="m42#0", metadata={"tenant_id": "forged", "material_id": 42})
    scoped = rr._tenant_scoped(req, AuthContext(authenticated=True, tenant_id="t"))
    assert scoped.doc_id == "t:m42#0"
    assert scoped.metadata == {"material_id": 42, "tenant_id": "t"}
    unbound = rr._tenant_scoped(req, AuthContext(authenticated=True, method="api_key"))
    assert unbound.doc_id == "m42#0"
    assert unbound.metadata == {"material_id": 42}


@pytest.mark.asyncio
async def test_delete_endpoint_requires_tenant_or_admin(collection, monkeypatch):
    monkeypatch.setattr(rag_scope.sparse_index, "remove_documents", lambda ids: None)
    body = rr.DeleteScopeRequest(rag_filter={"disciplina": "bd"})
    with pytest.raises(HTTPException) as exc:
        await rr.delete_docs(body, AuthContext(authenticated=True, method="api_key"))
    assert exc.value.status_code == 403
    assert await rr.delete_docs(body, AuthContext(authenticated=True, tenant_id="t")) == {"status": "ok", "removed": 1}
    assert "u:bd-1" in collection.docs
    with pytest.raises(ValidationError):
        rr.DeleteScopeRequest(rag_filter={})


@pytest.mark.asyncio
async def test_route_passes_scope_and_never_uses_semantic_cache(monkeypatch):
    from app.services import query_runtime as qr

    seen = {}

    async def _route(**kwargs):
        seen.update(kwargs)
        return {}

    monkeypatch.setattr(qr, "route_and_answer", _route)
    profile = {"use_rag": True, "max_tokens": 64, "runtime_hints": {}}
    req = QueryRequest(query="q", rag_filter={"disciplina": "bd"}, use_cache=True)
    await qr._route(req, "text", None, profile)
    assert seen["use_cache"] is False
    await qr._route(QueryRequest(query="q", use_cache=True), "text", None, profile)
    assert seen["use_cache"] is True


@pytest.mark.asyncio
async def test_retrieve_stage_builds_where_from_hints_and_tenant():
    from app.services.router_stages import RouteContext, _retrieve

    calls = []

    async def _bundle(query, **kwargs):
        calls.append(kwargs["where"])
        return {"augmented_prompt": "ctx+" + query}

    ctx = RouteContext(
        deps={"build_retrieval_bundle": _bundle}, query="q", system_prompt="", use_rag=True, max_tokens=64,
        temperature=0.1, modality="text", image_b64=None, rag_modality="text", use_cache=False,
        runtime_hints={"rag_filter": {"disciplina": "bd"}}, tenant_id="t",
    )
    assert (await _retrieve(ctx))[0] == "ctx+q"
    ctx.runtime_hints = {}
    await _retrieve(ctx)
    assert calls == [{"$and": [{"disciplina": "bd"}, {"tenant_id": "t"}]}, None]


@pytest.mark.asyncio
async def test_bundle_confines_dense_and_sparse_search_to_the_scope(monkeypatch):
    from app import rag_local

    seen = {}

    async def _dense(query, rag_mode, image_b64, collection_modality, n_results, where=None):
        seen["dense"] = where
        return [], {}, {}

    async def _restrict(hits, where, collection_modality):
        seen["sparse"] = where
        return [], {}, {}

    async def _sparse(query, top_k):
        return ["other"], {"other": "x"}, {}

    monkeypatch.setattr(rag_local, "_dense_search", _dense)
    monkeypatch.setattr(rag_local, "_sparse_search", _sparse)
    monkeypatch.setattr(rag_local, "restrict_to_scope", _restrict)
    bundle = await rag_local.build_retrieval_bundle("normalização", where={"disciplina": "bd"})
    assert seen == {"dense": {"disciplina": "bd"}, "sparse": {"disciplina": "bd"}}
    assert bundle["retrieval_skipped_reason"] == "no_candidates"
