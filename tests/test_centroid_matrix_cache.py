# Objective: Test coverage for the versioned centroid matrix cache and UQ.
"""Centroid matrix loading (revision-keyed cache) and the uncertainty score built on it."""

import json

import fakeredis
import numpy as np
import pytest
from app.services import bandit_centroids as bc
from app.utils import uncertainty as uq

KEY = uq.R_CENTROIDS_KEY
META = uq.R_CENTROIDS_META_KEY


@pytest.fixture(autouse=True)
def _fresh_cache():
    bc.reset_centroid_matrix_cache()
    yield
    bc.reset_centroid_matrix_cache()


def _store(server, vecs, rev):
    server.set(KEY, json.dumps([{"id": i, "vec": list(map(float, v))} for i, v in enumerate(vecs)]))
    server.hset(META, mapping={"updated_at": "1", "rev": str(rev)})


class _CountingRedis(fakeredis.FakeRedis):
    gets = 0

    def get(self, name):
        _CountingRedis.gets += 1
        return super().get(name)


def test_matrix_is_reused_until_revision_changes():
    server = _CountingRedis()
    _CountingRedis.gets = 0
    _store(server, [[1, 0, 0], [0, 2, 0]], rev=1)

    first = bc.load_centroid_matrix(server, KEY, META)
    second = bc.load_centroid_matrix(server, KEY, META)
    assert first is second and _CountingRedis.gets == 1
    assert np.allclose(np.linalg.norm(first.matrix, axis=1), 1.0)  # linhas unitárias

    _store(server, [[0, 0, 1]], rev=2)
    third = bc.load_centroid_matrix(server, KEY, META)
    assert _CountingRedis.gets == 2 and third.ids == [0] and third.matrix.shape == (1, 3)


def test_without_revision_parses_every_call():
    server = _CountingRedis()
    _CountingRedis.gets = 0
    server.set(KEY, json.dumps([{"vec": [1.0, 0.0]}]))
    bc.load_centroid_matrix(server, KEY, META)
    bc.load_centroid_matrix(server, KEY, META)
    assert _CountingRedis.gets == 2


def test_parse_skips_invalid_items_and_pads_dim():
    parsed = bc.parse_centroid_matrix(json.dumps([{"vec": []}, "x", {"id": 9, "vec": [3.0, 4.0]}]), dim=3)
    assert parsed.ids == [9]
    assert np.allclose(parsed.matrix, [[0.6, 0.8, 0.0]])
    assert bc.parse_centroid_matrix("[]") is None


def test_uncertainty_matches_reference_cosine(monkeypatch):
    rng = np.random.default_rng(3)
    cents = rng.standard_normal((20, 16))
    query = rng.standard_normal(16)
    server = fakeredis.FakeRedis()
    _store(server, cents, rev=1)
    monkeypatch.setattr(uq, "get_redis", lambda *a, **k: server)
    monkeypatch.setattr(uq, "embed_text", lambda text: query.tolist())

    reference = max(uq._cosine_similarity(query, c) for c in cents)
    expected = 1.0 - max(0.0, min(1.0, reference))
    assert uq.get_uncertainty_score("pergunta", "text") == pytest.approx(expected, abs=1e-5)


def test_uncertainty_cold_start_skips_embedding(monkeypatch):
    server = fakeredis.FakeRedis()
    monkeypatch.setattr(uq, "get_redis", lambda *a, **k: server)
    monkeypatch.setattr(uq, "embed_text", lambda text: (_ for _ in ()).throw(AssertionError("não deveria embedar")))
    assert uq.get_uncertainty_score("pergunta", "text") == 1.0


def test_save_centroids_bumps_revision(monkeypatch):
    from app.services import centroid_store as cs

    server = fakeredis.FakeRedis()
    monkeypatch.setattr(cs, "_get_rds", lambda: server)
    vec = np.ones(cs.CENTROIDS_DIM, dtype=np.float32)
    cs._save_centroids([{"id": 0, "vec": vec, "count": 1, "last": 0}])
    cs._save_centroids([{"id": 0, "vec": vec, "count": 2, "last": 0}])
    assert server.hget(cs.R_CENTROIDS_META, "rev") == b"2"
