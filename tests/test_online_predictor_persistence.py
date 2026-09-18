# Objective: Test coverage for online predictor persistence (debounce + atomic write).
"""OnlineErrorPredictor.maybe_save debounce and atomic state files."""

import pickle

import pytest

from app import online_predictor as op

pytestmark = pytest.mark.skipif(not op.RIVER_AVAILABLE, reason="river não instalado")


@pytest.fixture
def predictor(monkeypatch, tmp_path):
    monkeypatch.setattr(op, "_resolve_state_dir", lambda: tmp_path)
    p = op.OnlineErrorPredictor("ollama/test:1b")
    p.persistence_enabled = True
    if p.pipeline is None:
        p.pipeline = {"weights": [0.1, 0.2]}  # qualquer objeto picklável
    return p


def test_maybe_save_debounces_by_count(monkeypatch, predictor):
    saves = []
    monkeypatch.setattr(op, "SAVE_EVERY_N_UPDATES", 3)
    monkeypatch.setattr(op, "SAVE_EVERY_S", 3600)
    monkeypatch.setattr(predictor, "save", lambda: saves.append(1) or op.OnlineErrorPredictor.save(predictor))

    results = [predictor.maybe_save() for _ in range(7)]
    # a primeira grava (nunca gravou); depois, a cada 3 atualizações
    assert results == [True, False, False, True, False, False, True]
    assert len(saves) == 3


def test_maybe_save_debounces_by_time(monkeypatch, predictor):
    now = [100.0]
    monkeypatch.setattr(op.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(op, "SAVE_EVERY_N_UPDATES", 1000)
    monkeypatch.setattr(op, "SAVE_EVERY_S", 60)
    assert predictor.maybe_save() is True
    now[0] += 30
    assert predictor.maybe_save() is False
    now[0] += 31
    assert predictor.maybe_save() is True


def test_save_is_atomic_and_readable(predictor, tmp_path):
    predictor.save()
    with open(predictor.save_path, "rb") as f:
        assert pickle.load(f) is not None
    assert not list(tmp_path.glob("*.tmp.*"))  # nenhum temporário deixado para trás
