# Objective: Performance regression benchmarks for hot-path pure computations.
"""Hot-path micro-benchmarks (pytest-benchmark).

In the default suite ``--benchmark-disable`` (pytest.ini) runs each benchmark
once, as a plain test. Measure for real with:

    pytest tests/benchmarks --benchmark-enable --benchmark-only

External I/O is replaced by fakeredis and fixed vectors, so the numbers
reflect the CPU cost of our own code.
"""

import json

import fakeredis
import numpy as np
import pytest

pytestmark = pytest.mark.benchmark

DIM = 768
N_CENTROIDS = 20


def _unit(rng, n=DIM):
    v = rng.standard_normal(n).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def centroid_redis(monkeypatch):
    """fakeredis with the bandit centroids and a fixed query embedding."""
    from app.utils import uncertainty

    rng = np.random.default_rng(7)
    server = fakeredis.FakeRedis()
    cents = [{"id": i, "vec": _unit(rng).tolist(), "count": 5, "last": 0} for i in range(N_CENTROIDS)]
    server.set(uncertainty.R_CENTROIDS_KEY, json.dumps(cents))
    query_vec = _unit(rng).tolist()
    monkeypatch.setattr(uncertainty, "get_redis", lambda *a, **k: server)
    monkeypatch.setattr(uncertainty, "embed_text", lambda text: query_vec)
    return server


def test_bench_uncertainty_score(benchmark, centroid_redis):
    from app.utils.uncertainty import get_uncertainty_score

    score = benchmark(get_uncertainty_score, "Explique o teorema de Pitágoras", "text")
    assert 0.0 <= score <= 1.0


def test_bench_reciprocal_rank_fusion(benchmark):
    from app.rag_local import reciprocal_rank_fusion

    dense = [f"d{i}" for i in range(20)]
    sparse = [f"d{i}" for i in range(10, 30)]
    fused = benchmark(reciprocal_rank_fusion, dense, sparse)
    assert fused[0] in {"d10", "d11"} and len(fused) == 30


def test_bench_trim_context(benchmark):
    from app.rag_local import _trim_context_to_budget

    docs = [("lorem ipsum " * 70)[:800] for _ in range(10)]
    trimmed = benchmark(_trim_context_to_budget, docs, 1200)
    assert sum(len(d) for d in trimmed) <= 1200 * 4


def test_bench_choose_top2_models(benchmark, monkeypatch):
    from app import router_strategy

    models = [f"openrouter/vendor/model-{i}" for i in range(36)] + [
        "ollama/gemma3:4b",
        "ollama/phi4:latest",
        "anthropic/claude-haiku-4-5",
        "openai/gpt-5.5",
    ]
    snapshot = {
        m: {"mean": 0.5 + (i % 7) / 20, "count": 10 + i, "alpha": 3.0, "beta": 2.0} for i, m in enumerate(models)
    }
    monkeypatch.setattr(router_strategy, "get_snapshot", lambda: snapshot)
    weights = {"w_quality": 1.0, "w_latency": 0.5, "w_cost": 50.0}
    top2 = benchmark(router_strategy.choose_top2_models, models, weights, "pergunta", "text", 0.3)
    assert 1 <= len(top2) <= 2


def test_bench_parse_rubric_scores(benchmark):
    from app.services.judge_rubric import parse_rubric_scores

    text = (
        "<reasoning>"
        + "justificativa " * 40
        + '</reasoning><scores>{"clareza": 8, "acuracia": 7, "alinhamento": 9}</scores>'
    )
    assert benchmark(parse_rubric_scores, text) == {"clareza": 8.0, "acuracia": 7.0, "alinhamento": 9.0}


def test_bench_compute_reward(benchmark, monkeypatch):
    from app.services import reward

    server = fakeredis.FakeRedis()
    server.set("nsga:reward_weights:text", json.dumps({"quality": 0.6, "latency": 0.25, "cost": 0.15}))
    monkeypatch.setattr(reward, "_get_rds", lambda: server)
    value = benchmark(reward.compute_reward, "m", 8.0, 3.0, 0.004, "text")
    assert 0.0 <= value <= 1.0
