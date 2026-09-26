# Objective: OpenRouter exploration stats and blocklist survive a Redis wipe, and the durable row never regresses.
from __future__ import annotations

import inspect
import json

import pytest
from app.services import estado_duravel

from app import openrouter_exploration_state as st
from app import openrouter_explorer


@pytest.fixture
def rds(fake_aioredis):
    return fake_aioredis


@pytest.mark.asyncio
async def test_missing_stats_are_restored_from_the_table(monkeypatch, rds):
    monkeypatch.setattr(st, "_stats_do_banco", lambda nomes: {n: {"count": 40, "mean_reward": 0.7} for n in nomes})
    stats = await st._load_model_stats(rds, "openrouter/x/y")
    assert stats["count"] == 40
    assert json.loads(await rds.get(f"{st.REDIS_MODEL_STATS_PREFIX}openrouter/x/y"))["count"] == 40


@pytest.mark.asyncio
async def test_batch_reads_fill_only_the_gaps(monkeypatch, rds):
    await rds.set(f"{st.REDIS_MODEL_STATS_PREFIX}a", json.dumps({"count": 3}))
    pedidos = []
    monkeypatch.setattr(st, "_stats_do_banco", lambda nomes: pedidos.append(list(nomes)) or {"b": {"count": 9}})
    stats = await st._load_many_model_stats(rds, ["a", "b"])
    assert (stats["a"]["count"], stats["b"]["count"], pedidos) == (3, 9, [["b"]])


@pytest.mark.asyncio
async def test_the_blocklist_is_durable(monkeypatch, rds):
    gravado = {}
    monkeypatch.setattr(estado_duravel, "gravar", lambda chave, valor: gravado.update({chave: valor}) or True)
    monkeypatch.setattr(estado_duravel, "ler", lambda chave: gravado.get(chave))
    await st._save_blocklist(rds, {"openrouter/m/ruim"})
    await rds.flushall()
    assert await st._get_blocklist(rds) == {"openrouter/m/ruim"}


def test_the_durable_row_never_regresses():
    fonte = inspect.getsource(openrouter_explorer._persist_stats_to_db)
    assert "count = GREATEST(count, VALUES(count))" in fonte
    assert "stats_json = IF(VALUES(count) >= count, VALUES(stats_json), stats_json)" in fonte
    assert fonte.index("count = GREATEST") > fonte.index("stats_json = IF")


def test_unreadable_table_restores_nothing(monkeypatch):
    monkeypatch.setattr("app.db.get_engine", lambda: (_ for _ in ()).throw(RuntimeError("db")))
    assert st._stats_do_banco(["a"]) == {} and st._stats_do_banco([]) == {}
