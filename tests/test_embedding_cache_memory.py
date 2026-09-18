# Objective: Test coverage for embedding cache memory footprint and L2 encoding.
"""Embedding caches: float32 storage in L1 and binary/legacy decoding in L2."""

import json
import tracemalloc

import fakeredis
import numpy as np
import pytest

from app import embeddings

DIM = 768


def test_l1_entry_costs_about_float32_bytes():
    cache = embeddings.EmbeddingL1Cache(maxsize=1000, ttl_s=0)
    rng = np.random.default_rng(1)
    vectors = [rng.standard_normal(DIM).tolist() for _ in range(200)]

    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    for i, vec in enumerate(vectors):
        cache.set(f"k{i}", vec)
    after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    per_entry = sum(stat.size_diff for stat in after.compare_to(before, "filename")) / len(vectors)
    # float32 (768 x 4 = 3072 B) + overhead; como list[float] seriam ~24 KB.
    assert per_entry < 4096


def test_l1_returns_lists_and_copies_input():
    cache = embeddings.EmbeddingL1Cache(maxsize=10, ttl_s=0)
    src = np.ones(4, dtype=np.float32)
    cache.set("k", src)
    src[0] = 9.0  # mutar a origem não altera o cache
    out = cache.get("k")
    assert isinstance(out, list) and out == [1.0, 1.0, 1.0, 1.0]


def test_l2_roundtrip_binary_and_legacy_json(monkeypatch):
    server = fakeredis.FakeRedis()
    monkeypatch.setattr(embeddings, "_rds", server)
    vec = np.linspace(-1, 1, DIM).tolist()

    embeddings._save_cache("emb:test", vec)
    raw = server.get("emb:test")
    assert raw.startswith(b"f32:") and len(raw) == 4 + DIM * 4
    assert embeddings._load_cache("emb:test") == pytest.approx(vec, abs=1e-6)

    server.set("emb:legacy", json.dumps({"v": [0.25, 0.5]}))
    assert embeddings._load_cache("emb:legacy") == [0.25, 0.5]

    server.set("emb:broken", b"f32:abc")  # tamanho não múltiplo de 4
    assert embeddings._load_cache("emb:broken") is None
