# Objective: Two processes updating the same predictor never discard each other's learning.
from __future__ import annotations

import os
import pickle

from app.services.preditor_compartilhado import aprendizado


class _Preditor:
    """Minimal stand-in: the state is a list of learned samples, persisted as a pickle."""

    def __init__(self, caminho):
        self.save_path, self.persistence_enabled, self.estado, self.cargas = caminho, True, [], 0

    def _load(self):
        with open(self.save_path, "rb") as f:
            self.estado = pickle.load(f)
        self.cargas += 1

    def _load_validation(self):
        pass

    def save(self):
        tmp = f"{self.save_path}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self.estado, f)
        os.replace(tmp, self.save_path)


def test_each_update_starts_from_the_other_process_state(tmp_path):
    caminho = str(tmp_path / "predictor_m_logistic.pkl")
    a, b = _Preditor(caminho), _Preditor(caminho)
    for i in range(3):
        with aprendizado(a):
            a.estado.append(f"a{i}")
        with aprendizado(b):
            b.estado.append(f"b{i}")
    with open(caminho, "rb") as f:
        assert pickle.load(f) == ["a0", "b0", "a1", "b1", "a2", "b2"]  # nada foi descartado


def test_an_unchanged_file_is_not_reloaded(tmp_path):
    p = _Preditor(str(tmp_path / "x.pkl"))
    for _ in range(3):
        with aprendizado(p):
            p.estado.append(1)
    assert p.cargas == 0


def test_without_persistence_it_is_a_plain_block():
    p = _Preditor("/nao/existe.pkl")
    p.persistence_enabled = False
    with aprendizado(p):
        p.estado.append(1)
    assert p.estado == [1]
