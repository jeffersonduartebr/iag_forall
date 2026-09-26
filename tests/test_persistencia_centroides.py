# Objective: Semantic centroids survive a Redis wipe, and a fresh start never reuses a bandit cluster id.
from __future__ import annotations

import json

import fakeredis
from app.services import centroid_store, centroides_duraveis, estado_duravel


def test_an_empty_redis_is_refilled_from_the_durable_copy(monkeypatch):
    rds = fakeredis.FakeRedis()
    doc = [{"id": 3, "vec": [1.0] + [0.0] * 767, "count": 9, "last": 1}]
    monkeypatch.setattr(estado_duravel, "ler", lambda chave: doc)
    monkeypatch.setattr(centroid_store, "_get_rds", lambda: rds)
    cents = centroid_store._load_centroids(update_matrix_cache=False)
    assert [(c["id"], c["count"]) for c in cents] == [(3, 9)]
    assert json.loads(rds.get(centroid_store.R_CENTROIDS))[0]["id"] == 3


def test_saves_are_immediate_for_a_new_centroid_and_throttled_otherwise(monkeypatch):
    gravados = []
    monkeypatch.setattr(estado_duravel, "gravar", lambda chave, valor: gravados.append(len(valor)) or True)
    monkeypatch.setattr(centroides_duraveis, "_estado", {"ultimo": 0.0, "n": -1})
    centroides_duraveis.salvar([{"id": 0}])
    centroides_duraveis.salvar([{"id": 0}])  # só moveu: dentro do intervalo, não grava
    centroides_duraveis.salvar([{"id": 0}, {"id": 1}])  # centróide novo: grava na hora
    assert gravados == [1, 2]


def test_new_ids_never_reuse_a_known_cluster():
    assert centroid_store._new_centroid_id([{"id": 7}, {"id": 8}]) == 9
    assert centroid_store._new_centroid_id([]) == 0


def test_first_id_starts_after_the_bandit_clusters(monkeypatch):
    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, *a, **k):
            rotulos = ["cluster:4", "formative_v2:cluster:11", "global", "cluster:x"]
            return type("R", (), {"scalars": lambda s: type("S", (), {"all": lambda s2: rotulos})()})()

    monkeypatch.setattr("app.db.get_engine", lambda: type("E", (), {"connect": lambda s: _Conn()})())
    assert centroides_duraveis.primeiro_id() == 12


def test_first_id_is_zero_when_the_database_is_unreadable(monkeypatch):
    monkeypatch.setattr("app.db.get_engine", lambda: (_ for _ in ()).throw(RuntimeError("db")))
    assert centroides_duraveis.primeiro_id() == 0
