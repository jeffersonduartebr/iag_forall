# Objective: Typed view of the SHADOW_* settings, read on every use (switchable without restart).
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Tuple

from app.settings_dynamic import settings


def _lista(valor: object) -> List[str]:
    if isinstance(valor, (list, tuple)):
        itens = list(valor)
    else:
        texto = str(valor or "").strip()
        try:
            itens = json.loads(texto) if texto.startswith("[") else texto.split(",")
        except ValueError:
            itens = texto.split(",")
    return [str(i).strip() for i in itens if str(i).strip()]


def _num(chave: str, padrao: float) -> float:
    try:
        return float(settings.get(chave, padrao))
    except (TypeError, ValueError):
        return padrao


@dataclass(frozen=True)
class ConfigSombra:
    ligada: bool
    tenants: Tuple[str, ...]
    taxa: float
    estratos: Tuple[str, ...]
    provedores: Tuple[str, ...]
    regiao: str
    max_concorrencia: int
    max_local: int
    orcamento_usd: float
    teto_tenant_hora: int
    timeout_s: float
    fuso: str
    juizes: Tuple[str, ...]


def carregar() -> ConfigSombra:
    """Current shadow configuration (settings are re-read each time: the switch works without a restart)."""
    return ConfigSombra(
        ligada=str(settings.get("SHADOW_EXECUTION_ENABLED", "0")).strip().lower() in ("1", "true"),
        tenants=tuple(_lista(settings.get("SHADOW_TENANT_ALLOWLIST", ""))),
        taxa=min(1.0, max(0.0, _num("SHADOW_SAMPLE_RATE", 0.15))),
        estratos=tuple(_lista(settings.get("SHADOW_STRATA", ""))),
        provedores=tuple(_lista(settings.get("SHADOW_PROVIDER_ALLOWLIST", ""))),
        regiao=str(settings.get("SHADOW_REQUIRED_CLOUD_REGION", "") or "").strip(),
        max_concorrencia=max(1, int(_num("SHADOW_MAX_CONCURRENCY", 16))),
        max_local=max(0, int(_num("SHADOW_LOCAL_MAX_CONCURRENCY", 1))),
        orcamento_usd=max(0.0, _num("SHADOW_DAILY_BUDGET", 4.0)),
        teto_tenant_hora=max(0, int(_num("SHADOW_RATE_LIMIT_PER_TENANT_HOUR", 30))),
        timeout_s=max(1.0, _num("SHADOW_TIMEOUT_S", 120.0)),
        fuso=str(settings.get("SHADOW_BUDGET_TZ", "America/Fortaleza") or "America/Fortaleza"),
        juizes=tuple(_lista(settings.get("SHADOW_JUDGE_MODELS", "[]"))),
    )
