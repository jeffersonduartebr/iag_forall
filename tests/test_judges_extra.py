from types import SimpleNamespace

import pytest

from app import judges


def test_verdict_cache_and_helpers():
    c = judges.VerdictCache(maxsize=2, ttl_s=1)
    c.set("q1", "a1", 0.7)
    assert c.get("q1", "a1") == 0.7
    c._data[next(iter(c._data))] = (0.7, 0.0)
    assert c.get("q1", "a1") is None

    assert judges._adaptive_threshold([], 0.3) == 0.3
    assert judges._adaptive_threshold([0.2, 0.8], 0.3) >= 0.3
    assert judges._image_hash_from_b64(None) is None
    assert judges._image_hash_from_b64("abc") is not None

    s = judges.JudgeStats("m", avg_score=0.9, avg_cost=0.001, fitness=0.8)
    assert judges._score_candidate(s) > 0


def test_choose_two_and_extract_verdict(monkeypatch):
    models = ["m1", "m2", "m3"]
    stats = {"m1": judges.JudgeStats("m1", fitness=0.9), "m2": judges.JudgeStats("m2", fitness=0.8)}

    monkeypatch.setattr(judges.random, "random", lambda: 0.99)
    chosen = judges._choose_two(models, stats)
    assert len(chosen) == 2

    monkeypatch.setattr(judges.random, "random", lambda: 0.0)
    monkeypatch.setattr(judges.random, "sample", lambda v, k=2: v[:2])
    chosen2 = judges._choose_two(models, stats)
    assert len(chosen2) == 2

    assert judges._extract_binary_verdict("<verdict>CORRECT</verdict>") == 10.0
    assert judges._extract_binary_verdict("<verdict>INCORRECT</verdict>") == 0.0
    assert judges._extract_binary_verdict("VERDICT: CORRECT") == 10.0
    assert judges._extract_binary_verdict("VEREDITO: INCORRETO") == 0.0


@pytest.mark.asyncio
async def test_get_rag_context_describe_and_meta(monkeypatch):
    monkeypatch.setattr(judges, "embed_text", lambda q: [0.1])

    async def _query_embedding(coll, vec, n_results=5):
        return {"documents": [["doc1", "doc2"]]}

    monkeypatch.setattr(judges, "query_embedding", _query_embedding)
    monkeypatch.setattr(judges, "settings", SimpleNamespace(get=lambda k, d=None: "knowledge_base"))
    ctx = await judges.get_rag_context("q", n_results=2, max_chars=5)
    assert isinstance(ctx, str)

    async def _call_model(**kwargs):
        return "descrição curta", {}

    monkeypatch.setattr(judges, "call_model", _call_model)
    monkeypatch.setattr(judges, "IMAGE_DESC_MODEL_HINT", "openai/gpt-4o-mini")
    monkeypatch.setattr(judges, "VISION_VLM_CANDIDATES", ["openai/gpt-4o-mini"])
    monkeypatch.setattr(judges, "MULTIMODAL_VLM_CANDIDATES", [])
    desc = await judges._describe_image_if_needed("img", "vision")
    assert desc == "descrição curta"

    async def _meta(**kwargs):
        return "<verdict>CORRECT</verdict>", {}

    monkeypatch.setattr(judges, "call_model", _meta)
    m = await judges._meta_evaluate_binary("q", "a", [("j1", 0.0), ("j2", 10.0)], "prompt", reference=None)
    assert m == 10.0


@pytest.mark.asyncio
async def test_llm_pair_score_and_judge_answer_modes(monkeypatch):
    judges._verdict_cache = judges.VerdictCache()
    monkeypatch.setattr(judges, "_load_judge_stats", lambda w: {})
    monkeypatch.setattr(judges, "_choose_two", lambda models, stats: [judges.SelectedJudge("j1", 1.0), judges.SelectedJudge("j2", 1.0)])
    monkeypatch.setattr(judges, "settings", SimpleNamespace(JUDGE_MODELS=["j1", "j2"], JUDGES_MODE="hybrid"))

    calls = {"n": 0}

    async def _judge_call(**kwargs):
        calls["n"] += 1
        if kwargs["model"] == "j1":
            return "<verdict>CORRECT</verdict>", {"latency": 1.0, "cost_per_1k": 0.01}
        return "<verdict>INCORRECT</verdict>", {"latency": 1.2, "cost_per_1k": 0.02}

    monkeypatch.setattr(judges, "call_model", _judge_call)
    monkeypatch.setattr(judges, "_persist_judge_metrics", lambda **k: None)
    monkeypatch.setattr(judges, "_persist_judge_log", lambda **k: None)
    async def _get_rag_context(q):
        return ""

    async def _describe(*args, **kwargs):
        return ""

    async def _meta(*args, **kwargs):
        return 10.0

    monkeypatch.setattr(judges, "get_rag_context", _get_rag_context)
    monkeypatch.setattr(judges, "_describe_image_if_needed", _describe)
    monkeypatch.setattr(judges, "_meta_evaluate_binary", _meta)

    s1 = await judges.llm_based_score("q", "a", False, "text", None)
    assert 0.0 <= s1 <= 1.0
    s2 = await judges.llm_based_score("q", "a", False, "text", None)
    assert s2 == s1  # cached
    assert calls["n"] == 2

    async def _llm_score(**kwargs):
        return 0.8

    monkeypatch.setattr(judges, "llm_based_score", _llm_score)
    monkeypatch.setattr(judges, "heuristic_score", lambda a: 0.3)
    out = await judges.judge_answer("q", "a")
    assert len(out) == 2

    monkeypatch.setattr(judges, "settings", SimpleNamespace(JUDGES_MODE="heuristic"))
    out2 = await judges.judge_answer("q", "a")
    assert len(out2) == 1 and out2[0]["judge_id"] == "heuristic"


def test_judge_calibration_functions(monkeypatch):
    class _Conn:
        def execute(self, *_a, **_k):
            return None

    class _Ctx:
        def __enter__(self):
            return _Conn()

        def __exit__(self, exc_type, exc, tb):
            return False

    class _Engine:
        def begin(self):
            return _Ctx()

        def connect(self):
            class _C(_Ctx):
                def __enter__(self):
                    class _Result:
                        def fetchall(self):
                            return [("j1", 10, 8.0, 4, 6, 5)]

                    class _Conn2:
                        def execute(self, *_a, **_k):
                            return _Result()

                    return _Conn2()

            return _C()

    monkeypatch.setattr(judges, "engine", _Engine())
    monkeypatch.setattr(judges, "settings", SimpleNamespace(JUDGE_CALIBRATION_ENABLED=False))
    judges.record_judge_calibration("j1", "q", 8.0)
    judges.update_calibration_cache_status("q")
    assert judges.calibrate_judges()["status"] == "disabled"

    monkeypatch.setattr(
        judges,
        "settings",
        SimpleNamespace(JUDGE_CALIBRATION_ENABLED=True, JUDGE_CACHE_AGREEMENT_TARGET=0.9),
    )
    judges.record_judge_calibration("j1", "q", 8.0)
    judges.update_calibration_cache_status("q")
    metrics = judges.get_judge_calibration_metrics()
    assert "j1" in metrics
    res = judges.calibrate_judges()
    assert res["status"] == "ok"
