# Objective: The durable bandit posterior never regresses: a DB read failure skips the update instead of writing
# the prior over the history, and concurrent feedback processes serialize on the context hash.
from __future__ import annotations

import pytest
from app.services import bandit_stats_store as store


def test_welford_beta_matches_the_closed_form():
    s = {}
    for r in (1.0, 0.0, 0.5):
        s = store.welford_beta(s, r)
    assert (s["count"], s["mean"], s["var"]) == (3, pytest.approx(0.5), pytest.approx(0.25))
    assert (s["alpha"], s["beta"]) == (pytest.approx(2.5), pytest.approx(2.5))


def test_writers_see_an_unreadable_database_and_readers_do_not():
    store.cold_contexts.clear()

    def fora(ctx):
        raise store.BancoIndisponivelError("fora")

    assert store.db_fallback("c", fora, lambda c, s: None) == {}
    with pytest.raises(store.BancoIndisponivelError):
        store.db_fallback("c", fora, lambda c, s: None, estrito=True)
    assert not store.cold_contexts.is_cold("c")  # falha não vira "contexto sem dados"


def test_bandit_update_skips_a_context_it_cannot_read(monkeypatch):
    from app import bandits

    escritos = []
    monkeypatch.setattr(bandits, "centroids_online_update", lambda q: None)
    monkeypatch.setattr(bandits, "_auto_context_labels", lambda q, m: ["global"])
    monkeypatch.setattr(bandits, "_get_rds", lambda: None)

    def _ler(ctx, estrito=False):
        raise store.BancoIndisponivelError("fora")

    monkeypatch.setattr(bandits, "_get_ctx_stats", _ler)
    monkeypatch.setattr(bandits, "_set_ctx_stats", lambda ctx, s: escritos.append(ctx))
    monkeypatch.setattr(bandits, "_batch_upsert_ctx_db", lambda u: escritos.append(u))
    bandits.bandit_update("m1", "q", reward=1.0, modality="text")
    assert escritos == []


def test_the_upsert_keeps_the_larger_history():
    """No PostgreSQL o SET lê sempre a linha antiga; count NULL conta como 0 (a verificação real está em tests/pg)."""
    sql = str(store._UPSERT_SQL)
    assert "count = GREATEST(COALESCE(bandit_context_stats.count, 0), EXCLUDED.count)" in sql
    assert "CASE WHEN EXCLUDED.count >= COALESCE(bandit_context_stats.count, 0)" in sql


def test_the_context_lock_is_a_redis_lock_and_a_noop_without_redis():
    chamadas = []

    class _Rds:
        def lock(self, nome, **kw):
            chamadas.append((nome, kw))
            return "trava"

    assert store.trava_contexto(_Rds(), "meta:bandit:ctx:global") == "trava"
    assert chamadas == [("meta:bandit:ctx:global:lock", {"timeout": 10, "blocking_timeout": 5})]
    with store.trava_contexto(None, "x"):
        pass
