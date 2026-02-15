from types import SimpleNamespace

from app import prometheus_setup as ps


def test_setup_prometheus_and_registry(monkeypatch, tmp_path):
    pdir = tmp_path / "prom"
    pdir.mkdir()
    f1 = pdir / "a.db"
    f1.write_text("x")

    monkeypatch.setattr(ps, "PROM_DIR", str(pdir))
    ps.setup_prometheus()
    assert list(pdir.iterdir()) == []

    called = {"n": 0}
    monkeypatch.setattr(ps.multiprocess, "MultiProcessCollector", lambda registry: called.__setitem__("n", called["n"] + 1))
    reg = ps.prometheus_registry()
    assert reg is not None
    assert called["n"] == 1


def test_prometheus_metrics(monkeypatch):
    monkeypatch.setattr(ps, "prometheus_registry", lambda: SimpleNamespace())
    monkeypatch.setattr(ps, "generate_latest", lambda r: b"metrics")
    data, ctype = ps.prometheus_metrics()
    assert data == b"metrics"
    assert ctype == ps.CONTENT_TYPE_LATEST
