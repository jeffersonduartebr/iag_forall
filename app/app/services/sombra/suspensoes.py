# Objective: Record suspended intervals of the shadow (budget, per-tenant cap) and close them on resumption (R11).
"""A suspension is keyed by reason, tenant and period (local day for the budget, local hour for the tenant cap).
The key of the current period differs from the one that was suspended as soon as the day/hour turns, so the first
request accepted afterwards closes every suspension whose period is over: resumption is automatic.
"""

from __future__ import annotations

from typing import Any, Optional

from . import cotas, registro
from .config import ConfigSombra
from .linhas import agora_iso

_ABERTAS = f"{cotas.PREFIXO}suspensoes"


def _periodo(cfg: ConfigSombra, motivo: str) -> str:
    formato = "%Y%m%d%H" if motivo == "teto_tenant" else "%Y%m%d"
    return cotas.agora(cfg).strftime(formato)


def suspender(rds: Any, cfg: ConfigSombra, motivo: str, tenant: Optional[str]) -> None:
    chave = f"{motivo}:{tenant or '*'}:{_periodo(cfg, motivo)}"
    if rds.sadd(_ABERTAS, chave):
        registro.abrir_suspensao(chave, motivo, tenant, agora_iso())


def fechar_vencidas(rds: Any, cfg: ConfigSombra) -> None:
    for bruto in rds.smembers(_ABERTAS) or []:
        chave = bruto.decode() if isinstance(bruto, bytes) else str(bruto)
        motivo, _, periodo = chave.split(":", 2)
        if periodo.split(":")[-1] != _periodo(cfg, motivo):
            registro.fechar_suspensao(chave, agora_iso())
            rds.srem(_ABERTAS, bruto)
