import pytest

from app import rag_local as rl


@pytest.mark.asyncio
async def test_rag_local_helpers_and_visual_query_cache(monkeypatch):
    assert len(rl._hash_image("abc")) == 32
    assert rl._auto_modality("text", "img") == "vision"
    assert rl._auto_modality("multimodal", "img") == "multimodal"
    assert rl._auto_modality("vision", "img") == "vision"
    assert rl._auto_modality("text", None) == "text"

    class _Redis:
        def __init__(self):
            self.store = {}

        def get(self, key):
            return self.store.get(key)

        def setex(self, key, _ttl, val):
            self.store[key] = val

    rds = _Redis()
    monkeypatch.setattr(rl, "_rds", rds)

    async def _call_model(**kwargs):
        return "ferrugem no parafuso", {"ok": 1}

    monkeypatch.setattr(rl, "call_model", _call_model)

    q1 = await rl._generate_visual_search_query("img-1")
    q2 = await rl._generate_visual_search_query("img-1")
    assert q1 == "ferrugem no parafuso"
    assert q2 == "ferrugem no parafuso"


@pytest.mark.asyncio
async def test_compute_embedding_and_fusion_paths(monkeypatch):
    monkeypatch.setattr(rl, "embed_text", lambda txt: [len(txt)])
    monkeypatch.setattr(rl, "embed_multimodal", lambda q, img: {"multimodal": [9.0], "text": [8.0]})
    async def _vq(_img):
        return "descrição visual"

    monkeypatch.setattr(rl, "_generate_visual_search_query", _vq)

    emb_text = await rl._compute_embedding("abc", "text", None)
    emb_vision_no_query = await rl._compute_embedding("", "vision", "img")
    emb_vision_with_query = await rl._compute_embedding("pergunta", "vision", "img")
    emb_mm = await rl._compute_embedding("x", "multimodal", "img")
    emb_other = await rl._compute_embedding("fallback", "other", None)

    assert emb_text == [3]
    assert emb_vision_no_query == [16]  # "descrição visual"
    assert emb_vision_with_query == [25]  # "pergunta descrição visual"
    assert emb_mm == [9.0]
    assert emb_other == [8]

    fused = rl.reciprocal_rank_fusion(["a", "b"], ["b", "c"], k=60)
    assert fused[0] == "b"
    assert set(fused) == {"a", "b", "c"}


@pytest.mark.asyncio
async def test_build_prompt_add_document_and_health(monkeypatch):
    async def _emb(*a, **k):
        return [0.1, 0.2]

    async def _query(**kwargs):
        return {"ids": [["d1", "d2"]], "documents": [["doc1", "doc2"]]}

    monkeypatch.setattr(rl, "_compute_embedding", _emb)
    monkeypatch.setattr(
        rl,
        "query_embedding",
        _query,
    )
    monkeypatch.setattr(rl.sparse_index, "search", lambda q, top_k=20: [("d2", 0.9), ("d3", 0.7)])
    monkeypatch.setattr(rl.sparse_index, "get_text", lambda did: {"d2": "doc2", "d3": "doc3"}.get(did, ""))
    monkeypatch.setattr(rl, "rerank_documents", lambda q, docs, k: docs[:k])
    monkeypatch.setattr(rl.settings, "get", lambda key, default=None: "1" if key == "RERANK_ENABLED" else default, raising=False)

    prompt = await rl.build_augmented_prompt("como arrumar?", modality="text", image_b64=None, k=2)
    assert "CONTEXTO RECUPERADO" in prompt
    assert "doc1" in prompt or "doc2" in prompt

    async def _emb_none(*a, **k):
        return None

    monkeypatch.setattr(rl, "_compute_embedding", _emb_none)
    monkeypatch.setattr(rl.sparse_index, "search", lambda q, top_k=20: [])
    only_query = await rl.build_augmented_prompt("sem contexto", modality="text")
    assert only_query == "sem contexto"
    assert await rl.build_augmented_prompt("", modality="text", image_b64=None) == ""

    async def _add_ok(**kwargs):
        return True

    monkeypatch.setattr(rl, "add_document", _add_ok)
    ok = await rl.add_document_local("d1", text="abc")
    assert ok is True

    async def _add_fail(**kwargs):
        raise RuntimeError("x")

    monkeypatch.setattr(rl, "add_document", _add_fail)
    assert await rl.add_document_local("d2", text="abc") is False

    async def _health_ok():
        return True

    async def _health_fail():
        raise RuntimeError("x")

    monkeypatch.setattr(rl, "health_async", _health_ok)
    assert (await rl.health())["status"] == "ok"
    monkeypatch.setattr(rl, "health_async", _health_fail)
    assert (await rl.health())["status"] == "fail"
