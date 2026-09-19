# Objective: Test coverage for semantic centroid persistence and online clustering (fakeredis).
"""services.centroid_store: load/save round trip, degenerate reset, online update and lock."""

import json

import fakeredis
import numpy as np
import pytest
from app.services import centroid_store as cs


@pytest.fixture
def store(monkeypatch):
    rds = fakeredis.FakeRedis()
    monkeypatch.setattr(cs, "_get_rds", lambda: rds)
    monkeypatch.setattr(cs, "CENTROIDS_DIM", 4)
    monkeypatch.setattr(cs, "CENTROIDS_K", 3)
    monkeypatch.setattr(cs, "_centroid_matrix_cache", cs.CentroidMatrixCache())
    return rds


def _embed(monkeypatch, vec):
    monkeypatch.setattr(cs, "embed_text", lambda text: np.array(vec, dtype=np.float32))


def _unit(vec):
    v = np.array(vec, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_save_and_load_round_trip(store):
    cs._save_centroids([{"id": 3, "vec": [2.0, 0, 0, 0], "count": 5, "last": 10}])

    stored = json.loads(store.get(cs.R_CENTROIDS))
    assert stored == [{"id": 3, "vec": [1.0, 0.0, 0.0, 0.0], "count": 5, "last": 10}]
    meta = store.hgetall(cs.R_CENTROIDS_META)
    assert (meta[b"count"], meta[b"dim"], meta[b"rev"]) == (b"1", b"4", b"1")

    (loaded,) = cs._load_centroids()
    assert (loaded["id"], loaded["count"], loaded["last"]) == (3, 5, 10)
    np.testing.assert_allclose(loaded["vec"], [1.0, 0, 0, 0])
    assert cs._centroid_matrix_cache.nearest(_unit([1, 0, 0, 0]))[2] == 3


def test_save_resets_degenerate_vectors(store):
    cs._save_centroids([{"id": 0, "vec": [0.0, 0, 0, 0], "count": 9}, {"id": 1, "vec": [np.nan, 1, 0, 0], "count": 2}])
    for item in json.loads(store.get(cs.R_CENTROIDS)):
        assert item["count"] == 0
        assert np.linalg.norm(item["vec"]) == pytest.approx(1.0, abs=1e-5)


def test_load_skips_malformed_items_and_handles_bad_json(store):
    store.set(cs.R_CENTROIDS, json.dumps([{"id": 1}, "lixo", {"vec": [1, 0]}, {"id": 2, "vec": [0, 1]}]))
    loaded = cs._load_centroids(update_matrix_cache=False)
    assert [c["id"] for c in loaded] == [2]
    assert loaded[0]["vec"].shape == (4,)  # projetado para CENTROIDS_DIM
    assert cs._centroid_matrix_cache.nearest(_unit([0, 1, 0, 0]))[0] is None  # cache intocado

    store.set(cs.R_CENTROIDS, "{not json")
    assert cs._load_centroids() == []
    store.delete(cs.R_CENTROIDS)
    assert cs._load_centroids() == []


def test_online_update_creates_then_moves_nearest(store, monkeypatch):
    _embed(monkeypatch, [1, 0, 0, 0])
    assert cs.centroids_online_update("q1") == 0  # primeiro centróide

    _embed(monkeypatch, [0, 1, 0, 0])  # ortogonal: sim 0 < 0,35 cria outro
    assert cs.centroids_online_update("q2") == 1

    _embed(monkeypatch, [1, 0.1, 0, 0])  # próximo do 0: puxa o centróide 0
    assert cs.centroids_online_update("q3") == 0
    by_id = {c["id"]: c for c in cs._load_centroids()}
    assert by_id[0]["count"] == 2 and by_id[1]["count"] == 1
    assert 0 < by_id[0]["vec"][1] < 0.1
    assert store.get(cs.R_CENTROIDS_LOCK) is None  # lock liberado


def test_online_update_respects_k_limit(store, monkeypatch):
    for i in range(3):
        _embed(monkeypatch, np.eye(4)[i])
        cs.centroids_online_update(f"q{i}")
    _embed(monkeypatch, [0, 0, 0, 1])  # novo tema, mas K = 3: atualiza o mais próximo
    assert cs.centroids_online_update("q4") in {0, 1, 2}
    assert len(cs._load_centroids()) == 3


def test_online_update_skips_when_locked_or_embedding_fails(store, monkeypatch):
    store.set(cs.R_CENTROIDS_LOCK, "1")
    _embed(monkeypatch, [1, 0, 0, 0])
    assert cs.centroids_online_update("q") is None
    assert store.get(cs.R_CENTROIDS) is None

    def broken(text):
        raise RuntimeError("embedder down")

    monkeypatch.setattr(cs, "embed_text", broken)
    assert cs.centroids_online_update("q") is None


def test_nearest_label_is_read_only(store, monkeypatch):
    assert cs._nearest_centroid_label("q") is None  # sem centróides
    cs._save_centroids([{"id": 7, "vec": [0, 0, 1, 0]}, {"id": 8, "vec": [1, 0, 0, 0]}])
    _embed(monkeypatch, [0.1, 0, 1, 0])
    before = store.get(cs.R_CENTROIDS)
    assert cs._nearest_centroid_label("q") == "semctx:7"
    assert store.get(cs.R_CENTROIDS) == before


def test_without_redis_everything_is_a_noop(monkeypatch):
    monkeypatch.setattr(cs, "_get_rds", lambda: None)
    assert cs._load_centroids() == []
    cs._save_centroids([{"id": 0, "vec": [1, 0]}])
    assert cs._acquire_lock("k") is False
    cs._release_lock("k")
    assert cs._nearest_centroid_label("q") is None
