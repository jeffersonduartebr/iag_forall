# Objective: Test coverage for coverage boosters behavior and regressions.
"""Test coverage for coverage boosters behavior and regressions.

This test module verifies expected behavior, regression boundaries, and failure
handling for the corresponding runtime component.
"""


import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest


def test_uncertainty_helpers_and_paths(monkeypatch):
    """Testa uncertainty helpers and paths."""
    from app.utils import uncertainty as uq

    assert uq._cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(1.0)
    assert uq._cosine_similarity(np.array([0.0, 0.0]), np.array([1.0, 0.0])) == 0.0

    assert uq.get_uncertainty_score("img", modality="vision") == 0.7

    monkeypatch.setattr(uq, "get_redis", lambda: None)
    assert uq.get_uncertainty_score("hello", modality="text") == 1.0

    class _R:
        """Represent `_R` within this module.

The class groups the state and behavior required for R."""
        def get(self, _):
            """Execute the get routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            return None

    monkeypatch.setattr(uq, "get_redis", lambda: _R())
    assert uq.get_uncertainty_score("hello", modality="text") == 1.0

    class _R2:
        """Represent `_R2` within this module.

The class groups the state and behavior required for R2."""
        def get(self, _):
            """Execute the get routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            return '[{"vec":[1.0,0.0,0.0]}]'

    monkeypatch.setattr(uq, "get_redis", lambda: _R2())
    monkeypatch.setattr(uq, "embed_text", lambda t: [1.0, 0.0, 0.0])
    score = uq.get_uncertainty_score("hello", modality="text")
    assert 0.0 <= score <= 1.0

    monkeypatch.setattr(uq, "embed_text", lambda t: [0.0, 0.0, 0.0])
    assert uq.get_uncertainty_score("hello", modality="text") == 1.0


def test_query_service_helpers_and_insert(monkeypatch):
    """Testa query service helpers and insert."""
    from app import query_service as qs

    assert qs._to_blob([1.0, 2.0]) is not None
    assert qs._to_blob(None) is None
    assert qs._safe_json({"a": 1}) == '{"a": 1}'

    class _Conn:
        """Represent `_Conn` within this module.

The class groups the state and behavior required for Conn."""
        def __init__(self):
            """Initialize the internal state required by this instance.

The constructor keeps setup local to the object so callers can use it without additional bootstrapping."""
            self.executed = []

        def execute(self, stmt, params=None):
            """Execute the execute routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            self.executed.append((str(stmt), params))
            return SimpleNamespace(rowcount=1)

    class _Ctx:
        """Represent `_Ctx` within this module.

The class groups the state and behavior required for Ctx."""
        def __init__(self, conn):
            """Initialize the internal state required by this instance.

The constructor keeps setup local to the object so callers can use it without additional bootstrapping."""
            self.conn = conn

        def __enter__(self):
            """Execute the enter routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            return self.conn

        def __exit__(self, exc_type, exc, tb):
            """Execute the exit routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            return False

    conn = _Conn()
    monkeypatch.setattr(qs, "engine", SimpleNamespace(begin=lambda: _Ctx(conn)))

    qs.ensure_query_log()
    assert conn.executed

    qs.insert_query_log(
        query_text="q",
        model="m",
        modality="text",
        image_provided=False,
        answer="a",
        image_output_b64=None,
        latency_s=0.1,
        estimated_cost_usd=0.0,
        quality=8.0,
        reward=0.8,
        context_label="ctx",
        raw_payload={"x": 1},
        query_embedding=[0.1, 0.2],
        answer_embedding=[0.3, 0.4],
    )
    assert len(conn.executed) >= 2


def test_sparse_index_core(monkeypatch):
    """Testa sparse index core."""
    from app import sparse_index as si

    # Avoid touching real disk paths.
    monkeypatch.setattr(si.os.path, "exists", lambda p: False)
    idx = si.SparseIndex()
    assert idx._tokenize("Hello, World!") == ["hello", "world"]

    idx.add_document("d1", "texto um")
    idx.add_document("d2", "texto dois")
    idx.add_document("d1", "texto atualizado")
    assert idx.doc_ids.count("d1") == 1

    monkeypatch.setattr(idx, "_save", lambda: None)
    idx.commit()
    assert idx.bm25 is not None

    results = idx.search("texto", top_k=5)
    assert isinstance(results, list)
    assert idx.get_text("d1") != ""
    assert idx.get_text("missing") == ""

    idx.is_dirty = False
    idx.commit()  # no-op


def test_tasks_event_loop_and_task(monkeypatch):
    """Testa tasks event loop and task."""
    from app import tasks

    tasks._worker_loop = None
    loop = tasks._get_or_create_event_loop()
    assert loop is not None
    assert tasks._get_or_create_event_loop() is loop

    assert tasks.run_async(asyncio.sleep(0, result=123)) == 123

    # Signal handlers and loop lifecycle.
    tasks.on_worker_process_init()
    assert tasks._worker_loop is not None
    tasks.on_worker_process_shutdown()
    assert tasks._worker_loop is None


def test_online_predictor_metrics_and_calibration(monkeypatch, tmp_path):
    """Testa online predictor metrics and calibration."""
    from app import online_predictor as op

    if not op.RIVER_AVAILABLE:
        pytest.skip("river unavailable")

    p = op.OnlineErrorPredictor("test/model")
    p.save_path = str(tmp_path / "pred.pkl")
    p.validation_path = str(tmp_path / "val.pkl")

    emb = [0.1] * 8
    prob = p.predict_error_probability(emb)
    assert 0.0 <= prob <= 1.0
    p.learn(emb, is_correct=True)
    p.record_outcome(0.7, actual_error=True)
    p.record_outcome(0.1, actual_error=False)
    assert p.compute_accuracy() >= 0.0
    assert p.compute_brier_score() >= 0.0
    m = p.get_calibration_metrics()
    assert "brier_score" in m and "accuracy" in m

    # force enough records for calibration path
    p.prediction_log = [(0.6, True, 0.0)] * 120
    p.auto_calibrate_temperature()
    assert p.calibration_temp > 0

    p.save()
    # Loaders should not crash
    p._load()
    p._load_validation()

    # singleton helpers
    op._predictors.clear()
    p1 = op.get_predictor("m1")
    p2 = op.get_predictor("m1")
    assert p1 is p2
    assert "m1" in op.get_all_predictor_metrics()
    op.calibrate_all_predictors()
