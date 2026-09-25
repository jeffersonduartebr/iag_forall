# Objective: Coverage for publishing the best meta-optimization trial (params + per-model weights).
"""update_nsga_best_params: table DDL, best-trial lookup, params upsert and EMA-derived weights."""

from contextlib import contextmanager
from types import SimpleNamespace

import fakeredis
import pytest
from sqlalchemy.exc import SQLAlchemyError

from app import update_nsga_best_params as unb

TRIAL = {
    "trial_id": 4,
    "N_pop": 16,
    "N_gen": 10,
    "cxpb": 0.8,
    "mutpb": 0.1,
    "eta_c": 15.0,
    "eta_m": 20.0,
    "eff_mean": 2.5,
    "eff_std": 0.1,
}


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def first(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return self._rows


@pytest.fixture
def db(monkeypatch):
    """Engine stub: ``db.rows`` feeds reads, ``db.calls`` records every statement's params."""
    state = SimpleNamespace(rows=[], calls=[], sql=[], error=None)

    class _Conn:
        def execute(self, stmt, params=None):
            state.calls.append(params)
            state.sql.append(str(stmt))
            return _Result(state.rows)

    @contextmanager
    def ctx():
        if state.error:
            raise state.error
        yield _Conn()

    monkeypatch.setattr(unb, "engine", SimpleNamespace(connect=ctx, begin=ctx))
    return state


def test_init_tables_creates_three_tables_and_tolerates_db_errors(db):
    unb.init_tables()
    assert db.calls == [None, None, None]
    db.error = SQLAlchemyError("db down")
    unb.init_tables()  # só registra


def test_load_best_trial(db):
    db.rows = [TRIAL]
    assert unb.load_best_trial("text") == TRIAL
    assert db.calls == [{"m": "text"}]
    db.rows = []
    assert unb.load_best_trial("vision") is None
    db.error = SQLAlchemyError("db down")
    assert unb.load_best_trial("text") is None


@pytest.mark.parametrize("modality, row_id", [("text", 1), ("vision", 2), ("multimodal", 3)])
def test_update_best_params_upserts_one_row_per_modality(db, modality, row_id):
    unb.update_best_params(modality, TRIAL)
    assert db.calls == [dict(id=row_id, mod=modality, np=16, ng=10, cx=0.8, mu=0.1, ec=15.0, em=20.0)]


def test_update_best_params_rejects_unknown_modality_and_db_errors(db):
    unb.update_best_params("audio", TRIAL)
    assert db.calls == []
    db.error = SQLAlchemyError("db down")
    unb.update_best_params("text", TRIAL)  # erro só registrado


def test_compute_model_weights_normalizes_quality_per_latency_and_cost(db):
    db.rows = [
        {"model": "fast", "ema_latency": 1.0, "ema_quality": 8.0, "ema_cost": 0.01},
        {"model": "slow", "ema_latency": 4.0, "ema_quality": 8.0, "ema_cost": 0.01},
    ]
    weights = unb.compute_model_weights("text")
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["fast"] == pytest.approx(0.8, rel=1e-4)  # 4x mais rápido, mesmo custo e qualidade
    assert db.calls == [{"m": "text"}]


def test_compute_model_weights_free_model_is_floored_at_cheapest_paid_cost(db):
    # Com o piso 1e-6 o grátis tinha score 0.8/1e-6 e ficava com ~100% do peso.
    db.rows = [
        {"model": "free", "ema_latency": 1.0, "ema_quality": 8.0, "ema_cost": 0.0},
        {"model": "cheap", "ema_latency": 1.0, "ema_quality": 8.0, "ema_cost": 0.01},
        {"model": "pricey", "ema_latency": 1.0, "ema_quality": 8.0, "ema_cost": 0.02},
    ]
    weights = unb.compute_model_weights("text")
    assert weights["free"] == pytest.approx(weights["cheap"]) == pytest.approx(0.4)
    assert weights["pricey"] == pytest.approx(0.2)


def test_compute_model_weights_all_free_and_zero_latency_do_not_discriminate(db):
    db.rows = [
        {"model": "a", "ema_latency": 0.0, "ema_quality": 6.0, "ema_cost": 0.0},
        {"model": "b", "ema_latency": 2.0, "ema_quality": 6.0, "ema_cost": 0.0},
    ]
    # custo todo zero → 1.0 para ambos; latência 0 conta como a menor positiva (2.0).
    assert unb.compute_model_weights("text") == {"a": pytest.approx(0.5), "b": pytest.approx(0.5)}


def test_compute_model_weights_clamps_negative_scores_and_handles_all_zero(db):
    db.rows = [{"model": "bad", "ema_latency": 1.0, "ema_quality": -3.0, "ema_cost": 0.01}]
    assert unb.compute_model_weights("text") == {"bad": 0.0}


def test_compute_model_weights_empty_or_broken_history(db):
    assert unb.compute_model_weights("vision") == {}
    db.rows = [{"model": "m", "ema_latency": "n/a", "ema_quality": 8.0, "ema_cost": 0.01}]
    assert unb.compute_model_weights("vision") == {}
    db.error = SQLAlchemyError("db down")
    assert unb.compute_model_weights("vision") == {}


def test_persist_weights_uses_one_executemany_upsert_then_redis(monkeypatch, db):
    rds = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(unb, "rds", rds)
    unb.persist_weights("multimodal", {"a": 0.25, "b": 0.75})
    assert db.calls == [
        [{"mod": "multimodal", "model": "a", "w": 0.25}, {"mod": "multimodal", "model": "b", "w": 0.75}]
    ]
    # Em executemany o PyMySQL não substitui :w depois de VALUES(...): o UPDATE precisa de VALUES(weight).
    assert "weight = VALUES(weight)" in db.sql[0] and ":w" not in db.sql[0].split("UPDATE")[1]
    assert rds.get("nsga:weights:multimodal") == '{"a": 0.25, "b": 0.75}'


def test_persist_weights_skips_empty_and_survives_store_failures(monkeypatch, db):
    unb.persist_weights("text", {})
    assert db.calls == []
    db.error = SQLAlchemyError("db down")
    monkeypatch.setattr(unb, "rds", SimpleNamespace(set=_raise_redis))
    unb.persist_weights("text", {"a": 1.0})  # ambas as falhas são só registradas
    monkeypatch.setattr(unb, "rds", None)
    unb.persist_weights("text", {"a": 1.0})


def _raise_redis(*_a):
    raise ConnectionError("redis down")
