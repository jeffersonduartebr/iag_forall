# Objective: Typed view of the REGIME_* settings (re-read on every request).
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

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
    aquecimento_ate: Optional[dt.date] = None
    epsilon_aquecimento: float = 0.5

    def em_aquecimento(self, hoje: Optional[dt.date] = None) -> bool:
        """Warm-up runs up to and including ``aquecimento_ate`` (local day, America/Fortaleza)."""
        if self.aquecimento_ate is None:
            return False
        return (hoje or dt.datetime.now(ZoneInfo(FUSO)).date()) <= self.aquecimento_ate


FUSO = "America/Fortaleza"


def _data(chave: str) -> Optional[dt.date]:
    try:
        texto = str(settings.get(chave, "") or "").strip()
        return dt.date.fromisoformat(texto) if texto else None
    except ValueError:
        return None


def carregar() -> ConfigRegime:
    brutos = str(settings.get("REGIME_EXPLORACAO_TENANTS", "") or "")
    return ConfigRegime(
        tenants=tuple(t.strip() for t in brutos.split(",") if t.strip()),
        epsilon=min(1.0, max(0.0, _num("REGIME_EPSILON", 0.15))),
        teto=min(1.0, max(0.0, _num("REGIME_TETO", 0.15))),
        janela=max(1, int(_num("REGIME_JANELA_EPISODIOS", 20))),
        aquecimento_ate=_data("REGIME_AQUECIMENTO_ATE"),
        epsilon_aquecimento=min(1.0, max(0.0, _num("REGIME_EPSILON_AQUECIMENTO", 0.5))),
    )
