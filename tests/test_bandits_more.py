import numpy as np
import pytest


class _DummyMetric:
    def labels(self, **_kwargs):
        return self

    def inc(self, *_args, **_kwargs):
        return None

    def observe(self, *_args, **_kwargs):
        return None


class _FakeRedis:
    def __init__(self, value=None):
        self._value = value

    def get(self, _key):
        return self._value


def test_numpy_helpers():
    from app import bandits

    v = np.array([3.0, 4.0], dtype=np.float32)
    u = bandits._unit(v)
    assert np.isclose(np.linalg.norm(u), 1.0)

    assert bandits._cosine(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(1.0)
    assert bandits._cosine(np.array([0.0, 0.0]), np.array([1.0, 0.0])) == 0.0

    short = bandits._ensure_dim(np.array([1.0, 2.0], dtype=np.float32))
    assert len(short) == bandits.CENTROIDS_DIM
    assert np.isclose(np.linalg.norm(short), 1.0)


def test_centroid_matrix_cache_basic():
    from app.bandits import CentroidMatrixCache

    cache = CentroidMatrixCache()
    idx, sim, cid = cache.nearest(np.array([1.0, 0.0], dtype=np.float32))
    assert idx is None and sim == 0.0 and cid == -1

    cache.update(
        [
            {"id": 10, "vec": np.array([1.0, 0.0], dtype=np.float32)},
            {"id": 11, "vec": np.array([0.0, 1.0], dtype=np.float32)},
        ]
    )
    idx, sim, cid = cache.nearest(np.array([0.0, 1.0], dtype=np.float32))
    assert idx == 1
    assert cid == 11
    assert sim == pytest.approx(1.0)


def test_new_centroid_id_and_nearest(monkeypatch):
    from app import bandits

    assert bandits._new_centroid_id([{"id": 0}, {"id": 2}]) == 1

    cents = [
        {"id": 1, "vec": np.array([1.0, 0.0], dtype=np.float32)},
        {"id": 2, "vec": np.array([0.0, 1.0], dtype=np.float32)},
    ]

    monkeypatch.setattr(bandits._centroid_matrix_cache, "is_stale", lambda max_age_s=120.0: True)
    idx, sim = bandits._nearest_centroid_vec(np.array([0.0, 1.0], dtype=np.float32), cents)
    assert idx == 1
    assert sim == pytest.approx(1.0)

    monkeypatch.setattr(bandits._centroid_matrix_cache, "is_stale", lambda max_age_s=120.0: False)
    monkeypatch.setattr(bandits._centroid_matrix_cache, "nearest", lambda v: (0, 0.9, 1))
    idx, sim = bandits._nearest_centroid_vec(np.array([0.0, 1.0], dtype=np.float32), cents)
    assert idx == 0
    assert sim == pytest.approx(0.9)


def test_dynamic_epsilon_and_choosers(monkeypatch):
    from app import bandits

    eps_empty = bandits._dynamic_epsilon({})
    eps_rich = bandits._dynamic_epsilon(
        {
            "a": {"count": 20, "var": 0.01},
            "b": {"count": 20, "var": 0.01},
        }
    )
    assert eps_empty > eps_rich

    stats = {"m1": {"count": 0, "var": 0.5, "mean": 0.1}, "m2": {"count": 10, "var": 0.0, "mean": 0.9}}
    monkeypatch.setattr("app.bandits.random.random", lambda: 0.0)
    assert bandits._choose_epsilon_greedy(["m1", "m2"], stats) == "m1"
    monkeypatch.setattr("app.bandits.random.random", lambda: 1.0)
    assert bandits._choose_epsilon_greedy(["m1", "m2"], stats) == "m2"

    # UCB1 should force unseen model first due +inf bonus.
    assert bandits._choose_ucb1(["m1", "m2"], stats) == "m1"

    vals = iter([0.2, 0.8])
    monkeypatch.setattr("app.bandits.np.random.beta", lambda a, b: next(vals))
    assert bandits._choose_thompson(["m1", "m2"], {"m1": {"alpha": 1, "beta": 1}, "m2": {"alpha": 1, "beta": 1}}) == "m2"


def test_meta_strategy_and_combine(monkeypatch):
    from app import bandits

    monkeypatch.setattr(bandits, "rds", _FakeRedis(b"thompson"))
    assert bandits._meta_choose_strategy() == "thompson"

    monkeypatch.setattr(bandits, "rds", _FakeRedis(b"unknown"))
    assert bandits._meta_choose_strategy() == "ucb1"

    monkeypatch.setattr(bandits, "_choose_epsilon_greedy", lambda *_: "a")
    monkeypatch.setattr(bandits, "_choose_ucb1", lambda *_: "a")
    monkeypatch.setattr(bandits, "_choose_thompson", lambda *_: "b")
    chosen, debug = bandits._meta_combine_choices(["a", "b"], {})
    assert chosen == "a"
    assert set(debug.keys()) == {"epsilon_greedy", "ucb1", "thompson"}

    monkeypatch.setattr(bandits, "_choose_epsilon_greedy", lambda *_: "a")
    monkeypatch.setattr(bandits, "_choose_ucb1", lambda *_: "b")
    monkeypatch.setattr(bandits, "_choose_thompson", lambda *_: "c")
    monkeypatch.setattr(bandits, "_meta_choose_strategy", lambda: "epsilon_greedy")
    chosen, _ = bandits._meta_combine_choices(["a", "b", "c"], {})
    assert chosen == "a"


def test_select_model_and_snapshot(monkeypatch):
    from app import bandits

    monkeypatch.setattr(bandits, "BANDIT_SELECT", _DummyMetric())
    monkeypatch.setattr(bandits, "_auto_context_labels", lambda q, modality="text": ["global"])
    monkeypatch.setattr(bandits, "_get_ctx_stats", lambda ctx: {"m1": {"mean": 0.1}})
    monkeypatch.setattr(
        bandits,
        "_meta_combine_choices",
        lambda models, stats: (
            "m2",
            {"epsilon_greedy": "m1", "ucb1": "m2", "thompson": "m2"},
        ),
    )
    assert bandits.select_model(["m1", "m2"], "hello", modality="text") == "m2"
    assert bandits.select_model([], "hello") == "ollama/gemma3:4b"

    monkeypatch.setattr(bandits, "_get_ctx_stats", lambda ctx: {"m": {"alpha": 2.0, "beta": 2.0}})
    snap = bandits.get_snapshot()
    assert "m" in snap
    out = bandits.sample_metrics_from_snapshot(snap)
    assert "m" in out
    assert "m::text" in out


def test_bandit_update_and_reward(monkeypatch):
    from app import bandits

    monkeypatch.setattr(bandits, "BANDIT_UPDATE", _DummyMetric())
    monkeypatch.setattr(bandits, "BANDIT_REWARD", _DummyMetric())
    monkeypatch.setattr(bandits, "centroids_online_update", lambda query: 1)
    monkeypatch.setattr(bandits, "_auto_context_labels", lambda q, m: ["global"])
    monkeypatch.setattr(bandits, "_upsert_ctx_db", lambda *_args, **_kwargs: None)

    captured = {}

    def _set_ctx(ctx, stats):
        captured[ctx] = stats

    monkeypatch.setattr(bandits, "_set_ctx_stats", _set_ctx)
    monkeypatch.setattr(bandits, "_get_ctx_stats", lambda ctx: {})
    bandits.bandit_update("m1", "q", reward=2.0, modality="text")
    assert "global" in captured
    assert captured["global"]["m1"]["mean"] == pytest.approx(1.0)

    # _load_nsga_weights default path (no redis)
    monkeypatch.setattr(bandits, "rds", None)
    wq, wl, wc = bandits._load_nsga_weights()
    assert (wq + wl + wc) == pytest.approx(1.0)

    monkeypatch.setattr(bandits, "rds", _FakeRedis('{"quality": 2, "latency": 1, "cost": 1}'))
    wq, wl, wc = bandits._load_nsga_weights()
    assert wq == pytest.approx(0.5)
    assert wl == pytest.approx(0.25)
    assert wc == pytest.approx(0.25)

    monkeypatch.setattr(bandits, "_load_nsga_weights", lambda: (0.6, 0.3, 0.1))
    r = bandits.compute_reward("m1", quality=8.0, latency_s=1.0, cost_per_1k=0.1)
    assert 0.0 <= r <= 1.0

    def _boom():
        raise RuntimeError("x")

    monkeypatch.setattr(bandits, "_load_nsga_weights", _boom)
    assert bandits.compute_reward("m1", quality=1.0, latency_s=1.0, cost_per_1k=0.1) == 0.0


def test_centroids_online_update_and_label(monkeypatch):
    from app import bandits

    monkeypatch.setattr(bandits, "_acquire_lock", lambda *args, **kwargs: True)
    monkeypatch.setattr(bandits, "_release_lock", lambda *args, **kwargs: None)
    monkeypatch.setattr(bandits, "embed_text", lambda q: np.ones(4, dtype=np.float32))
    monkeypatch.setattr(bandits, "CENTROIDS_DIM", 4)

    saved = {}

    def _save(c):
        saved["cents"] = c

    monkeypatch.setattr(bandits, "_save_centroids", _save)
    monkeypatch.setattr(bandits, "_load_centroids", lambda update_matrix_cache=True: [])
    cid = bandits.centroids_online_update("hello")
    assert cid == 0
    assert saved["cents"][0]["id"] == 0

    now = int(__import__("time").time())
    cents = [{"id": 7, "vec": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), "count": 1, "last": now}]
    monkeypatch.setattr(bandits, "_load_centroids", lambda update_matrix_cache=True: cents)
    monkeypatch.setattr(bandits, "_nearest_centroid_vec", lambda v, c: (0, 0.99))
    cid = bandits.centroids_online_update("hello")
    assert cid == 7

    monkeypatch.setattr(bandits, "_load_centroids", lambda update_matrix_cache=True: cents)
    monkeypatch.setattr(bandits, "_nearest_centroid_vec", lambda v, c: (0, 0.42))
    label = bandits._nearest_centroid_label("hello")
    assert label == "semctx:7"
