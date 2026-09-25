# Objective: Daily budget, per-tenant hourly cap, global concurrency and suspensions of the shadow (R2, R9, R11).
"""Redis state of the shadow, all under the ``shadow:`` prefix (never shared with policy or exploration state).

- The budget day is computed in ``SHADOW_BUDGET_TZ`` (``zoneinfo``), not in the container's clock: the key
  changes at local midnight, which is also the automatic resumption.
- The per-tenant cap counts sampled requests per local hour; the key changes at the next hour.
- Global concurrency is a Redis counter (every worker process shares it) with a TTL as a leak guard.
Without Redis the shadow does not run: it cannot prove it stays within budget.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Optional
from zoneinfo import ZoneInfo

from .config import ConfigSombra
from .metricas import SHADOW_BUDGET_EXHAUSTED_HOUR, SHADOW_BUDGET_REMAINING

PREFIXO = "shadow:"
_TTL_DIA_S, _TTL_HORA_S, _TTL_VAGA_S = 3 * 86400, 2 * 3600, 600


def agora(cfg: ConfigSombra) -> dt.datetime:
    """Local time in the budget zone (monkeypatched in tests)."""
    return dt.datetime.now(ZoneInfo(cfg.fuso))


def chave_dia(cfg: ConfigSombra) -> str:
    return f"{PREFIXO}budget:{agora(cfg):%Y%m%d}"


def chave_hora_tenant(cfg: ConfigSombra, tenant: str) -> str:
    return f"{PREFIXO}tenant:{tenant}:{agora(cfg):%Y%m%d%H}"


def gasto_hoje(rds: Any, cfg: ConfigSombra) -> float:
    return float(rds.get(chave_dia(cfg)) or 0.0)


def orcamento_esgotado(rds: Any, cfg: ConfigSombra) -> bool:
    restante = cfg.orcamento_usd - gasto_hoje(rds, cfg)
    SHADOW_BUDGET_REMAINING.set(max(0.0, restante))
    if restante > 0:
        SHADOW_BUDGET_EXHAUSTED_HOUR.set(-1)  # dia novo (ou ainda com saldo): o alerta do dia anterior se apaga
    return restante <= 0


def somar_gasto(rds: Any, cfg: ConfigSombra, usd: float) -> Optional[float]:
    """Add ``usd`` to today's spend; returns the local hour (decimal) when this call exhausted the budget."""
    if usd <= 0:
        return None
    chave = chave_dia(cfg)
    antes = float(rds.get(chave) or 0.0)
    depois = float(rds.incrbyfloat(chave, float(usd)))
    rds.expire(chave, _TTL_DIA_S)
    SHADOW_BUDGET_REMAINING.set(max(0.0, cfg.orcamento_usd - depois))
    if antes < cfg.orcamento_usd <= depois:
        momento = agora(cfg)
        hora = momento.hour + momento.minute / 60
        SHADOW_BUDGET_EXHAUSTED_HOUR.set(hora)
        return hora
    return None


def contar_amostra_tenant(rds: Any, cfg: ConfigSombra, tenant: str) -> bool:
    """Count one sampled request for ``tenant`` this hour; ``False`` when it exceeds the hourly cap."""
    chave = chave_hora_tenant(cfg, tenant)
    n = int(rds.incr(chave))
    rds.expire(chave, _TTL_HORA_S)
    return n <= cfg.teto_tenant_hora


def tomar_vaga(rds: Any, cfg: ConfigSombra) -> bool:
    """Take one of the ``SHADOW_MAX_CONCURRENCY`` global slots (``False`` when all are taken)."""
    chave = f"{PREFIXO}inflight"
    n = int(rds.incr(chave))
    rds.expire(chave, _TTL_VAGA_S)
    if n > cfg.max_concorrencia:
        rds.decr(chave)
        return False
    return True


def soltar_vaga(rds: Any) -> None:
    if int(rds.decr(f"{PREFIXO}inflight")) < 0:
        rds.set(f"{PREFIXO}inflight", 0)


def gpu_ocupada() -> bool:
    """Real requests in flight on a local model: shadow calls to local models yield to them (R2)."""
    try:
        from app.providers._ollama import get_ollama_admission_snapshot

        return int(get_ollama_admission_snapshot().get("total_inflight", 0) or 0) > 0
    except Exception:
        return True  # sem sinal confiável, a sombra cede
