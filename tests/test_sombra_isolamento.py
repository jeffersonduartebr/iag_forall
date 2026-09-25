# Objective: Shadow results never reach policy state, provider health or system metrics (R2, R3, R10).
"""Real ``call_model`` against a provider built like the real ones (breaker, slots, metrics, throughput EMA)."""

from __future__ import annotations

import pybreaker
import pytest
from app.providers._base import BaseProvider, LLMResponse
from app.services.sombra import executor
from app.utils.breaker_async import guarded_by
from sombra_fakes import Modelos, capturar_registro, job, ligar

from app import providers_async as pa

BREAKER = pybreaker.CircuitBreaker(fail_max=1, reset_timeout=60, name="fake_openrouter")
_RESPOSTAS = Modelos()


class _Fake(BaseProvider):
    def __init__(self):
        super().__init__("openrouter", concurrency_limit=4)

    @guarded_by(BREAKER)
    async def generate(self, prompt, image_b64=None, **kwargs):
        await self._acquire_slot(kwargs["model"])
        try:
            if "claude-sonnet-5" in kwargs["model"]:
                self._record_metrics(kwargs["model"], 0.1, 0.0, False)
                raise RuntimeError("upstream 503")
            texto, _ = await _RESPOSTAS(model=kwargs["model"], prompt=prompt)
            self._record_metrics(kwargs["model"], 0.5, 0.01, True)
            self._record_generation_metrics(kwargs["model"], 80, 0.5)
            return LLMResponse(text=texto, latency=0.5, load_time=0.0, cost=0.01, prompt_tokens=100,
                               completion_tokens=80, model_used=kwargs["model"])
        finally:
            self._release_slot(kwargs["model"])


@pytest.fixture
def ambiente(monkeypatch, fake_redis):
    monkeypatch.setattr(pa, "is_provider_configured", lambda provider: True)
    monkeypatch.setitem(pa.ProviderFactory._instances, "openrouter", _Fake())
    monkeypatch.setitem(pa.ProviderFactory._instances, "gemini", _Fake())
    BREAKER.close()
    chamadas_politica = []
    import app.bandits as bandits
    import app.semantic_cache as cache

    for modulo, nome in ((bandits, "bandit_update"), (cache, "store_cache")):
        monkeypatch.setattr(modulo, nome, lambda *a, _n=nome, **k: chamadas_politica.append(_n))
    return fake_redis, chamadas_politica


def _estado(rds):
    metricas = {
        m.name: sorted((s.name, tuple(sorted(s.labels.items())), s.value) for s in m.samples)
        for coletor in (pa.PROV_REQ, pa.PROV_OK, pa.PROV_ERR, pa.PROV_COST, pa.GENERATION_TOKENS_PER_SECOND)
        for m in coletor.collect()
    }
    chaves = {k: rds.dump(k) for k in rds.keys() if not k.decode().startswith("shadow:")}
    return {
        "redis": chaves,
        "breaker": (BREAKER.current_state, BREAKER.fail_counter),
        "indisponivel": pa.is_provider_temporarily_unavailable("openrouter/anthropic/claude-sonnet-5"),
        "metricas": metricas,
    }


async def _rodar_n(monkeypatch, rds, ligada: bool, n: int = 3):
    ligar(monkeypatch, ligada=ligada)
    capturar_registro(monkeypatch)
    for i in range(n):
        await executor.executar(job(request_id=f"req-{i}"), rds=rds)


@pytest.mark.asyncio
async def test_policy_and_provider_state_is_identical_with_the_shadow_on_or_off(monkeypatch, ambiente):
    rds, chamadas_politica = ambiente
    inicial = _estado(rds)
    await _rodar_n(monkeypatch, rds, ligada=False)
    desligada = _estado(rds)
    await _rodar_n(monkeypatch, rds, ligada=True)
    ligada = _estado(rds)
    assert inicial == desligada == ligada
    assert chamadas_politica == []
    assert any(k.startswith(b"shadow:budget:") for k in rds.keys())  # a sombra rodou de fato


@pytest.mark.asyncio
async def test_control_the_same_calls_outside_shadow_do_change_the_state(monkeypatch, ambiente):
    """Proves the comparison above can fail: an ordinary call moves metrics, the throughput EMA and the breaker."""
    rds, _ = ambiente
    monkeypatch.setattr("app.utils.redis_client.get_redis_sync_nonblocking", lambda *a, **k: rds)
    antes = _estado(rds)
    await pa.call_model(model="openrouter/deepseek/deepseek-v4.1-flash", prompt="p")
    with pytest.raises(Exception):
        await pa.call_model(model="openrouter/anthropic/claude-sonnet-5", prompt="p")
    depois = _estado(rds)
    assert depois["metricas"] != antes["metricas"] and depois["breaker"] != antes["breaker"]
