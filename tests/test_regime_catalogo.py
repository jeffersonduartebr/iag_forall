# Objective: The regime's OpenRouter catalogue pool — only text, only while the explorer is on and under its daily
# caps, never the configured candidates, and never an exception into the route.
from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
from app.services.regime import sorteio


def _instalar(monkeypatch, *, enabled=True, configurado=True, contagem=0, usd=0.0, pool=None, erro=None):
    cfg = SimpleNamespace(enabled=enabled, max_per_day=10, max_usd_per_day=1.0)

    async def _async(v):
        return v

    async def _pool(excluidos, c, modalidade):
        if erro:
            raise erro
        return pool if pool is not None else ["openrouter/x/novo", "a"]

    mods = {
        "app.openrouter_exploration_policy": {"load_exploration_config": lambda s: cfg},
        "app.openrouter_exploration_state": {
            "_get_daily_count": lambda r: _async(contagem),
            "_get_daily_usd": lambda r: _async(usd),
        },
        "app.openrouter_explorer": {
            "_build_exploration_pool": _pool,
            "_get_redis": lambda: _async(object()),
            "_openrouter_configured": lambda: configurado,
        },
    }
    for nome, attrs in mods.items():
        mod = ModuleType(nome)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, nome, mod)


def _ctx(modalidade="text"):
    return SimpleNamespace(modality=modalidade)


@pytest.mark.asyncio
async def test_pool_excludes_configured_candidates(monkeypatch):
    _instalar(monkeypatch)
    assert await sorteio._pool_catalogo(_ctx(), ["a", "b"]) == ["openrouter/x/novo"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kw",
    [
        {"enabled": False},
        {"configurado": False},
        {"contagem": 10},
        {"usd": 1.0},
        {"erro": RuntimeError("catálogo fora")},
    ],
)
async def test_pool_is_empty_when_the_explorer_cannot_run(monkeypatch, kw):
    _instalar(monkeypatch, **kw)
    assert await sorteio._pool_catalogo(_ctx(), ["a"]) == []


@pytest.mark.asyncio
async def test_pool_is_text_only(monkeypatch):
    _instalar(monkeypatch)
    assert await sorteio._pool_catalogo(_ctx("vision"), ["a"]) == []
