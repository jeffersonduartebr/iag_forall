"""Módulo `tests/test_metrics_collector.py`: descreve responsabilidades e integrações deste arquivo."""

from types import SimpleNamespace

from app import metrics_collector as mc


class _Conn:
    """Classe `_Conn`: concentra responsabilidades de test metrics collector."""
    def __init__(self):
        """Inicializa estado interno necessário para uso da classe."""
        self.calls = []

    def execute(self, stmt, params=None):
        """Executa execute."""
        self.calls.append((str(stmt), params))
        return SimpleNamespace(rowcount=1)


class _Ctx:
    """Classe `_Ctx`: concentra responsabilidades de test metrics collector."""
    def __init__(self, conn):
        """Inicializa estado interno necessário para uso da classe."""
        self.conn = conn

    def __enter__(self):
        """Executa enter."""
        return self.conn

    def __exit__(self, exc_type, exc, tb):
        """Executa exit."""
        return False


def test_ensure_table_and_persist_sample_success(monkeypatch):
    """Testa ensure table and persist sample success."""
    conn = _Conn()
    monkeypatch.setattr(mc, "engine", SimpleNamespace(begin=lambda: _Ctx(conn)))

    mc._persist_sample(
        model_name="gpt-4o",
        modality="vision",
        latency_s=1.5,
        quality_0_10=8.0,
        cost_usd=0.02,
        cost_per_1k=0.03,
        tokens_in=11,
        tokens_out=22,
        embedding_dim=512,
        vision_usage=True,
        generation=3,
    )

    assert len(conn.calls) == 2
    _, params = conn.calls[1]
    assert params["model"] == "gpt-4o"
    assert params["modality"] == "vision"
    assert params["latms"] == 1500.0
    assert params["vis"] == 1
    assert params["gen"] == 3
    assert 0.0 <= params["fit"] <= 1.0


def test_persist_sample_swallows_exceptions(monkeypatch):
    """Testa persist sample swallows exceptions."""
    def _boom():
        """Executa boom."""
        raise RuntimeError("db down")

    monkeypatch.setattr(mc, "_ensure_model_metrics_table", _boom)
    mc._persist_sample("m", "text", 0.1, 5.0, 0.0)


def test_update_model_metrics_ema_and_snapshot_copy(monkeypatch):
    """Testa update model metrics ema and snapshot copy."""
    mc._MODEL_METRICS.clear()
    persisted = []
    monkeypatch.setattr(mc, "_persist_sample", lambda **kw: persisted.append(kw))

    mc.update_model_metrics("m1", latency=1.0, quality=8.0, cost=0.010, modality="text")
    snap1 = mc.get_snapshot()
    assert snap1[("m1", "text")] == {"quality": 8.0, "latency": 1.0, "cost": 0.01}

    mc.update_model_metrics("m1", latency=2.0, quality=6.0, cost=0.020, modality="text")
    snap2 = mc.get_snapshot()
    assert snap2[("m1", "text")]["quality"] == 7.4
    assert snap2[("m1", "text")]["latency"] == 1.3
    assert snap2[("m1", "text")]["cost"] == 0.013
    assert len(persisted) == 2

    # copy isolation
    snap2[("m1", "text")]["quality"] = -1
    assert mc.get_snapshot()[("m1", "text")]["quality"] == 7.4
