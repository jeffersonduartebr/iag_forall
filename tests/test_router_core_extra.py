"""Módulo `tests/test_router_core_extra.py`: descreve responsabilidades e integrações deste arquivo."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app import router_core as rc


def test_ema_history_cache_and_batch_queue(monkeypatch):
    """Testa ema history cache and batch queue."""
    h = rc.EMAHistoryCache(maxsize=2, ttl_s=1)
    h.set(("m", "a"), {"ema_latency": 1})
    assert ("m", "a") in h
    assert h.get(("m", "a")) is not None

    # force expiry
    h._data[("m", "a")]["_last_update"] = 0
    assert h.get(("m", "a")) is None

    h.set(("m", "1"), {"ema_latency": 1})
    h.set(("m", "2"), {"ema_latency": 2})
    h.set(("m", "3"), {"ema_latency": 3})
    assert h.size() == 2
    removed = h.cleanup_expired()
    assert removed >= 0

    persisted = []
    q = rc.EMABatchQueue(max_size=2, flush_interval=0)
    monkeypatch.setattr(q, "_persist_batch", lambda items: persisted.append(items) or len(items))
    q.add("text", "m1", {"ema_latency": 1, "ema_quality": 1, "ema_cost": 1, "updates": 1})
    q.add("text", "m2", {"ema_latency": 1, "ema_quality": 1, "ema_cost": 1, "updates": 10})
    assert persisted
    assert q.should_flush() is True
    assert q.flush() == 0


def test_load_ema_from_db_and_start_stop_services(monkeypatch):
    """Testa load ema from db and start stop services."""
    class _Rows:
        """Classe `_Rows`: concentra responsabilidades de test router core extra."""
        def mappings(self):
            """Executa mappings."""
            return self

        def all(self):
            """Executa all."""
            return [{"modality": "text", "model": "m", "ema_latency": 1, "ema_quality": 2, "ema_cost": 3, "ema_alignment": 1}]

    class _Conn:
        """Classe `_Conn`: concentra responsabilidades de test router core extra."""
        def __enter__(self):
            """Executa enter."""
            return self

        def __exit__(self, exc_type, exc, tb):
            """Executa exit."""
            return False

        def execute(self, *_a, **_k):
            """Executa execute."""
            return _Rows()

    class _Engine:
        """Classe `_Engine`: concentra responsabilidades de test router core extra."""
        def connect(self):
            """Executa connect."""
            return _Conn()

    monkeypatch.setattr(rc, "_get_db_engine", lambda: _Engine())
    rc.EMA_HISTORY = rc.EMAHistoryCache()
    rc._load_ema_from_db()
    assert rc.EMA_HISTORY.size() == 1

    started = []

    class _T:
        """Classe `_T`: concentra responsabilidades de test router core extra."""
        def __init__(self, target=None, daemon=None, name=None):
            """Inicializa estado interno necessário para uso da classe."""
            self._name = name

        def start(self):
            """Executa start."""
            started.append(self._name)

        def join(self, timeout=None):
            """Executa join."""
            return None

    monkeypatch.setattr(rc.threading, "Thread", _T)
    monkeypatch.setattr(rc, "_load_ema_from_db", lambda: None)
    monkeypatch.setattr(rc.EMA_BATCH, "flush", lambda: 0)
    rc._bg_threads.clear()
    rc._bg_started = False
    rc.start_background_services()
    assert started
    rc.stop_background_services()
    assert rc._bg_started is False


@pytest.mark.asyncio
async def test_route_and_answer_dedup_and_timeout(monkeypatch):
    """Testa route and answer dedup and timeout."""
    monkeypatch.setattr(rc, "_route_and_answer_internal", AsyncMock(return_value={"ok": True}))
    monkeypatch.setattr(rc, "settings", SimpleNamespace(get=lambda k, d=None: "1" if k == "REQUEST_DEDUP_ENABLED" else 1))

    class _Dedup:
        """Classe `_Dedup`: concentra responsabilidades de test router core extra."""
        async def deduplicate(self, **kwargs):
            """Executa deduplicate."""
            return await kwargs["execute_fn"]()

    monkeypatch.setattr(rc, "get_request_deduplicator", lambda: _Dedup())
    out = await rc.route_and_answer("q", deduplicate=True)
    assert out["ok"] is True

    monkeypatch.setattr(
        rc,
        "settings",
        SimpleNamespace(get=lambda k, d=None: "0" if k == "REQUEST_DEDUP_ENABLED" else 1),
    )
    out2 = await rc.route_and_answer("q", deduplicate=True)
    assert out2["ok"] is True

    async def _slow(*args, **kwargs):
        """Executa slow."""
        await rc.asyncio.sleep(0.05)
        return {"ok": False}

    monkeypatch.setattr(rc, "_route_and_answer_internal", _slow)
    with pytest.raises(rc.asyncio.TimeoutError):
        await rc.route_and_answer("q", timeout_seconds=0.001, deduplicate=False)


@pytest.mark.asyncio
async def test_process_background_feedback_branches(monkeypatch):
    """Testa process background feedback branches."""
    monkeypatch.setattr(rc, "_get_ctx_stats", lambda _k: {"m1": {"count": 10, "mean": 0.8}})
    monkeypatch.setattr(rc, "embed_text", lambda q: [0.1, 0.2])

    class _Pred:
        """Classe `_Pred`: concentra responsabilidades de test router core extra."""
        def predict_error_probability(self, emb):
            """Executa predict error probability."""
            return 0.9

        def learn(self, emb, is_correct):
            """Executa learn."""
            return None

        def record_outcome(self, p, e):
            """Executa record outcome."""
            return None

        def save(self):
            """Executa save."""
            return None

    monkeypatch.setattr(rc, "get_predictor", lambda model: _Pred())
    monkeypatch.setattr(rc, "judge_answer", AsyncMock(return_value=[{"score": 0.8}, {"score": 1.0}]))
    monkeypatch.setattr(rc, "compute_reward", lambda *a, **k: 0.7)
    monkeypatch.setattr(rc, "bandit_update", lambda **k: None)
    monkeypatch.setattr(rc, "_persist_ema", lambda *a, **k: None)
    monkeypatch.setattr(rc, "store_cache", AsyncMock(return_value=None))
    monkeypatch.setattr(rc, "insert_query_log", lambda **k: None)
    monkeypatch.setattr(rc.random, "random", lambda: 0.0)  # force judge
    monkeypatch.setattr("app.router_core.asyncio.create_task", lambda coro: coro.close())
    monkeypatch.setattr(rc, "settings", SimpleNamespace(JUDGE_MIN_SAMPLE_RATE=0.0))

    await rc.process_background_feedback(
        query="q",
        answer="a",
        chosen_model="m1",
        modality="text",
        latency_s=0.2,
        cost_val=0.01,
    )

    # no judge branch
    monkeypatch.setattr(rc.random, "random", lambda: 1.0)
    await rc.process_background_feedback(
        query="q2",
        answer="a2",
        chosen_model="m1",
        modality="text",
        latency_s=0.2,
        cost_val=0.01,
    )
