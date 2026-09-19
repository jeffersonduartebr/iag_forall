# Objective: Test coverage for the multi-objective correlation service (compute, publish, persist).
"""correlation_metrics: per-model correlations, Prometheus gauges and history rows."""

import numpy as np
import pandas as pd
import pytest

from app import correlation_metrics as cm


def _frame():
    return pd.DataFrame(
        {
            "model": ["a"] * 4 + ["b"] * 2 + ["c"] * 3,
            "latency_ms": [100, 200, 300, 400, 1, 2, "x", 5, 6],
            "cost_usd": [1, 1, 1, 1, 1, 1, 1, 1, 1],
            "quality_score": [9, 7, 5, 3, 1, 1, 1, 1, 1],
            "fitness": [1, 2, 3, 4, 1, 1, 1, 1, 1],
            "generation": [1, 2, 3, 7, 1, 1, 1, 1, 1],
        }
    )


def test_safe_corr_edge_cases():
    assert cm._safe_corr(np.array([1.0]), np.array([2.0])) == 0.0
    assert cm._safe_corr(np.array([1.0, 1.0]), np.array([1.0, 2.0])) == 0.0  # variância zero
    assert cm._safe_corr(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0])) == pytest.approx(1.0)
    assert cm._safe_corr(np.array([1.0, 2.0]), np.array([1.0])) == 0.0  # tamanhos incompatíveis


def test_compute_correlations_per_model():
    df = _frame()
    out = cm.compute_correlations(df)
    # "b" tem 2 amostras e "c" fica com 2 depois de descartar a latência inválida.
    assert list(out) == ["a"]
    a = out["a"]
    assert a["corr_lq"] == pytest.approx(-1.0)
    assert a["corr_cq"] == 0.0  # custo constante
    assert a["generation"] == 7
    assert a["matrix"].shape == (3, 3)
    assert df["latency_ms"].tolist()[6] == "x"  # não altera o DataFrame do chamador
    assert cm.compute_correlations(pd.DataFrame()) == {}


def test_publish_and_persist(monkeypatch):
    data = cm.compute_correlations(_frame())
    cm.publish_metrics(data)
    assert cm.correlation_latency_quality.labels(model="a")._value.get() == pytest.approx(-1.0)
    assert cm.model_correlation_matrix.labels(model="a", metric_x="cost", metric_y="cost")._value.get() == 0.0

    saved = {}
    monkeypatch.setattr(
        pd.DataFrame,
        "to_sql",
        lambda self, table, engine, **kw: saved.update(table=table, rows=self.to_dict("records")),
    )
    cm.persist_correlations(data)
    assert saved["table"] == "correlation_history"
    (row,) = saved["rows"]
    assert row["model"] == "a" and row["generation"] == 7
    assert row["r2_mean"] == pytest.approx((1.0 + 0.0 + row["corr_fitness_weights"] ** 2) / 3)

    saved.clear()
    cm.persist_correlations({})
    assert saved == {}


def test_persist_swallows_db_errors(monkeypatch):
    def broken(self, *a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(pd.DataFrame, "to_sql", broken)
    cm.persist_correlations(cm.compute_correlations(_frame()))
