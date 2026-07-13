# Objective: Test coverage for observability extra behavior and regressions.
"""Extra tests for observability helpers and logging setup."""

import logging

from app import observability as ob


def test_prometheus_dir_and_registry_fallbacks(monkeypatch, tmp_path):
    """Prometheus directory and registry helpers should handle missing env and collector failures."""
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    assert ob._ensure_prometheus_dir() is None

    prom_dir = tmp_path / "prom"
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(prom_dir))
    assert ob._ensure_prometheus_dir() == str(prom_dir)

    monkeypatch.setattr(ob, "_prom_dir", str(prom_dir))
    monkeypatch.setattr(ob.multiprocess, "MultiProcessCollector", lambda reg: (_ for _ in ()).throw(RuntimeError("collector fail")))
    reg = ob._build_registry()
    assert reg is not None


def test_json_renderer_and_manual_json_log(monkeypatch):
    """Structured logging helpers should serialize both normal and fallback payloads."""
    renderer = ob.JsonUTF8Renderer()
    rendered = renderer(None, None, {"event": "ok", "value": 1})
    assert '"event": "ok"' in rendered

    calls = {"n": 0}
    real_dumps = ob.json.dumps

    def _dumps(payload, ensure_ascii=False):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TypeError("boom")
        return real_dumps(payload, ensure_ascii=ensure_ascii, default=str)

    monkeypatch.setattr(ob.json, "dumps", _dumps)
    fallback = renderer(None, None, {"event": "bad"})
    assert "Falha ao serializar log" in fallback

    records = []

    logger = logging.getLogger("observability")
    handler = logging.Handler()
    handler.emit = lambda record: records.append(record.getMessage())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    ob.json_log("info", "evt", value=1)
    logger.removeHandler(handler)
    assert records and '"event": "evt"' in records[0]


def test_add_correlation_and_setup_logging_idempotent(monkeypatch):
    """Correlation injection and setup should be safe and idempotent."""
    monkeypatch.setattr("app.correlation.get_correlation_id", lambda: "cid-1")
    event = ob._add_correlation_id(None, None, {"event": "x"})
    assert event["correlation_id"] == "cid-1"

    monkeypatch.setattr(ob.structlog, "configure", lambda **kwargs: None)
    monkeypatch.setattr(ob.logging, "basicConfig", lambda **kwargs: None)
    monkeypatch.setattr(ob.logger, "info", lambda *a, **k: None)
    monkeypatch.setattr(ob, "_logger_configured", False)
    ob.setup_logging(level=logging.DEBUG)
    assert ob._logger_configured is True
    ob.setup_logging(level=logging.INFO)
