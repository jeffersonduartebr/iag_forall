# Objective: Typed view of the REGIME_* settings (re-read on every request).
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from app.settings_dynamic import settings


def _num(chave: str, padrao: float) -> float:
    try:
        return float(settings.get(chave, padrao))
    except (TypeError, ValueError):
        return padrao


@dataclass(frozen=True)
class ConfigRegime:
    tenants: Tuple[str, ...]
    epsilon: float
    teto: float
    janela: int


def carregar() -> ConfigRegime:
    brutos = str(settings.get("REGIME_EXPLORACAO_TENANTS", "") or "")
    return ConfigRegime(
        tenants=tuple(t.strip() for t in brutos.split(",") if t.strip()),
        epsilon=min(1.0, max(0.0, _num("REGIME_EPSILON", 0.15))),
        teto=min(1.0, max(0.0, _num("REGIME_TETO", 0.15))),
        janela=max(1, int(_num("REGIME_JANELA_EPISODIOS", 20))),
    )
