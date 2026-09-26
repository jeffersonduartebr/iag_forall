# Objective: A participant's exploration window lost with Redis is replayed from query_log (field phase only), once.
from __future__ import annotations

import json

from app.services.regime import janela, reconstrucao


def _linhas(*decisoes):
    return [json.dumps({"regime": d}) for d in decisoes]


def test_a_wiped_window_is_replayed_before_counting(monkeypatch, fake_redis):
    registros = [("E1", False), ("E2", True), ("E2", False)]
    monkeypatch.setattr(reconstrucao, "_decisoes", lambda p: registros)
    assert janela.contagem(fake_redis, "P1", "E3", 20) == (3, 1)
    monkeypatch.setattr(reconstrucao, "_decisoes", lambda p: (_ for _ in ()).throw(AssertionError("de novo")))
    assert janela.contagem(fake_redis, "P1", "E3", 20) == (3, 1)  # uma vez só por participante


def test_a_participant_without_history_queries_once(monkeypatch, fake_redis):
    chamadas = []
    monkeypatch.setattr(reconstrucao, "_decisoes", lambda p: chamadas.append(p) or [])
    for _ in range(3):
        assert janela.contagem(fake_redis, "novo", "E1", 20) == (0, 0)
    assert chamadas == ["novo"]


def test_only_field_decisions_with_an_episode_are_replayed(monkeypatch):
    brutos = _linhas(
        {"fase": "campo", "episodio": "E1", "explorou": True},
        {"fase": "aquecimento", "episodio": "E0", "explorou": True},
        {"fase": "campo", "explorou": False},
    )

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, *a, **k):
            return type("R", (), {"scalars": lambda s: type("S", (), {"all": lambda s2: brutos})()})()

    monkeypatch.setattr("app.db.get_engine", lambda: type("E", (), {"connect": lambda s: _Conn()})())
    assert reconstrucao._decisoes("P1") == [("E1", True)]


def test_an_unreadable_log_leaves_the_window_as_is(monkeypatch, fake_redis):
    monkeypatch.setattr(reconstrucao, "_decisoes", lambda p: (_ for _ in ()).throw(RuntimeError("db")))
    assert janela.contagem(fake_redis, "P9", "E1", 20) == (0, 0)
