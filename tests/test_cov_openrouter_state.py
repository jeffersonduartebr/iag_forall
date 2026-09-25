# Objective: Redis state of OpenRouter exploration (blocklist, daily budget counters, per-model stats).
"""Behaviour of ``app.openrouter_exploration_state`` against a real (fake) Redis."""

from __future__ import annotations

import json
import types

import pytest

from app import openrouter_exploration_state as st


class _Broken:
    """Redis client whose every command fails (connection dropped)."""

    def __getattr__(self, name):
        async def _fail(*args, **kwargs):
            raise ConnectionError("redis down")

        return _fail


@pytest.fixture
def day(monkeypatch):
    """Controls the calendar date seen by the state module (``time.strftime``)."""
    clock = types.SimpleNamespace(today="2026-09-24")
    monkeypatch.setattr(st, "time", types.SimpleNamespace(strftime=lambda fmt: clock.today))
    return clock


@pytest.mark.asyncio
async def test_blocklist_roundtrip_is_sorted_json(fake_aioredis):
    await st._save_blocklist(fake_aioredis, {"openrouter/b/m", "openrouter/a/m"})
    assert json.loads(await fake_aioredis.get(st.REDIS_BLOCKLIST_KEY)) == ["openrouter/a/m", "openrouter/b/m"]
    assert await st._get_blocklist(fake_aioredis) == {"openrouter/a/m", "openrouter/b/m"}


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", ["{not json", json.dumps({"a": 1}), json.dumps("x")])
async def test_blocklist_ignores_corrupt_or_non_list_payloads(fake_aioredis, raw):
    await fake_aioredis.set(st.REDIS_BLOCKLIST_KEY, raw)
    assert await st._get_blocklist(fake_aioredis) == set()


@pytest.mark.asyncio
async def test_blocklist_drops_empty_entries(fake_aioredis):
    await fake_aioredis.set(st.REDIS_BLOCKLIST_KEY, json.dumps(["x/y", "", None]))
    assert await st._get_blocklist(fake_aioredis) == {"x/y"}


@pytest.mark.asyncio
async def test_no_redis_means_empty_state_and_noop_writes():
    assert await st._get_blocklist(None) == set()
    assert await st._get_daily_count(None) == 0
    assert await st._get_daily_usd(None) == 0.0
    assert await st._load_model_stats(None, "m") == {}
    for write in (st._save_blocklist(None, {"m"}), st._track_model_key(None, "m"), st._increment_daily(None, 1.0)):
        assert await write is None
    assert await st._save_model_stats(None, "m", {"count": 1}) is None


@pytest.mark.asyncio
async def test_redis_failures_degrade_to_empty_state_without_raising():
    rds = _Broken()
    assert await st._get_blocklist(rds) == set()
    assert await st._get_daily_count(rds) == 0
    assert await st._get_daily_usd(rds) == 0.0
    assert await st._load_model_stats(rds, "m") == {}
    await st._save_blocklist(rds, {"m"})
    await st._track_model_key(rds, "m")
    await st._increment_daily(rds, 0.5)
    await st._save_model_stats(rds, "m", {"count": 1})


@pytest.mark.asyncio
async def test_daily_count_resets_all_counters_on_a_new_day(fake_aioredis, day):
    await st._increment_daily(fake_aioredis, 0.25)
    await st._increment_daily(fake_aioredis, 0.0)
    assert await st._get_daily_count(fake_aioredis) == 2
    assert await st._get_daily_usd(fake_aioredis) == pytest.approx(0.25)

    day.today = "2026-09-25"
    assert await st._get_daily_usd(fake_aioredis) == 0.0  # outro dia: gasto de ontem não conta
    assert await st._get_daily_count(fake_aioredis) == 0
    assert await fake_aioredis.get(st.REDIS_DAILY_USD_KEY) == b"0"


@pytest.mark.asyncio
async def test_increment_after_midnight_does_not_inherit_yesterdays_budget(fake_aioredis, day):
    """Regression: a pick at 23:59 and its outcome at 00:00 used to carry yesterday's count and USD over."""
    for _ in range(3):
        await st._increment_daily(fake_aioredis, 1.0)
    assert await st._get_daily_count(fake_aioredis) == 3

    day.today = "2026-09-25"
    await st._increment_daily(fake_aioredis, 0.1)  # sem leitura prévia do contador no novo dia
    assert await st._get_daily_count(fake_aioredis) == 1
    assert await st._get_daily_usd(fake_aioredis) == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_increment_sets_two_day_expiry_on_counters(fake_aioredis, day):
    await st._increment_daily(fake_aioredis, 0.3)
    assert 0 < await fake_aioredis.ttl(st.REDIS_DAILY_COUNTER_KEY) <= 2 * 86400
    assert 0 < await fake_aioredis.ttl(st.REDIS_DAILY_USD_KEY) <= 2 * 86400


def test_decode_stats_accepts_bytes_and_rejects_non_dicts():
    assert st._decode_stats(b'{"count": 2}') == {"count": 2}
    assert st._decode_stats("[1, 2]") == {}
    assert st._decode_stats("{oops") == {}
    assert st._decode_stats(None) == {}


@pytest.mark.asyncio
async def test_save_model_stats_persists_and_tracks_the_model(fake_aioredis):
    await st._save_model_stats(fake_aioredis, "openrouter/a/m", {"count": 4, "mean_reward": 0.9})
    assert await st._load_model_stats(fake_aioredis, "openrouter/a/m") == {"count": 4, "mean_reward": 0.9}
    assert await fake_aioredis.smembers(st.REDIS_MODELS_SET_KEY) == {b"openrouter/a/m"}
    many = await st._load_many_model_stats(fake_aioredis, ["openrouter/a/m", "openrouter/none"])
    assert many == {"openrouter/a/m": {"count": 4, "mean_reward": 0.9}, "openrouter/none": {}}
    assert await st._load_many_model_stats(fake_aioredis, []) == {}
