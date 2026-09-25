# Objective: Time budgets derived from each model's measured throughput (tokens/s), not fixed per workload.
"""Tempo de cada chamada = latência inicial + tokens pedidos / vazão medida do modelo.

The throughput is an EMA per model kept in Redis (``tps:<model>``), fed by every successful generation.
Until a model has been measured, a conservative default applies (local GPU models are slow; cloud is fast).
A fixed 20/25 s budget made every long answer from the T4 time out and left no time for a fallback.
"""

from __future__ import annotations

import logging
from typing import List

from ..settings_dynamic import settings

logger = logging.getLogger(__name__)
_ROTAS = ("ollama/", "openrouter/", "gemini/", "anthropic/", "openai/")
ALFA = 0.3  # peso da medição nova na EMA


def _chave(model: str) -> str:
    nome = model or ""
    rota = next((r for r in _ROTAS if nome.startswith(r)), "")
    return "tps:" + nome[len(rota):]


def _num(chave: str, padrao: float) -> float:
    try:
        valor = settings.get(chave, padrao)
        return padrao if valor in (None, "") else float(valor)
    except (TypeError, ValueError):
        return padrao


def _redis():
    try:
        from ..utils.redis_client import get_redis_sync_nonblocking

        return get_redis_sync_nonblocking()
    except Exception:
        return None


def vazao(model: str) -> float:
    """Measured tokens/s of the model (EMA), or the provider default when unmeasured."""
    padrao = _num("TPS_PADRAO_LOCAL", 10.0) if (model or "").startswith("ollama/") else _num("TPS_PADRAO_NUVEM", 40.0)
    rds = _redis()
    try:
        medido = rds.get(_chave(model)) if rds else None
        return max(float(medido), 1.0) if medido else padrao
    except Exception:
        return padrao


def registrar_vazao(model: str, tokens: int, segundos: float) -> None:
    """Fold one generation into the model's throughput EMA (best effort)."""
    if tokens < 20 or segundos <= 0:
        return  # respostas curtas medem latência inicial, não vazão
    rds = _redis()
    if rds is None:
        return
    try:
        atual = rds.get(_chave(model))
        nova = tokens / segundos if not atual else (1 - ALFA) * float(atual) + ALFA * tokens / segundos
        rds.set(_chave(model), round(nova, 2), ex=7 * 86400)
    except Exception as exc:
        logger.debug("[orcamento_tempo] EMA de vazão não atualizada: %s", exc)


def orcamento_raciocinio() -> int:
    """Thinking/reasoning token quota granted to cloud models on top of the visible answer."""
    return max(0, int(_num("REASONING_BUDGET_TOKENS", 4096)))


def tokens_totais(max_tokens: int, com_raciocinio: bool = True) -> int:
    """Visible answer ceiling plus the reasoning quota when the model reasons.

    Reasoning tokens are billed as output and count against the provider's output ceiling: without this
    headroom a reasoning model spends ``max_tokens`` thinking and returns an empty answer.
    """
    return int(max_tokens) + (orcamento_raciocinio() if com_raciocinio else 0)


def _tempo(tokens: int, tps: float) -> float:
    """Fixed overhead plus the tokens at ``tps``, with 30% headroom."""
    return _num("LATENCIA_INICIAL_S", 4.0) + 1.3 * tokens / tps


def tempo_esperado(model: str, tokens: int) -> float:
    """Seconds a call to ``model`` should need for ``tokens`` at its measured throughput."""
    return _tempo(tokens, vazao(model))


def prazo_da_requisicao(max_tokens: int, piso: float) -> float:
    """Synchronous deadline for a request: never below the workload's, room for one fallback, capped.

    Sized for a cloud model at the default rate (the model is not chosen yet); each call is then bounded
    by its own model's measured rate in :func:`prazo_da_chamada`.
    """
    necessario = _tempo(tokens_totais(max_tokens), _num("TPS_PADRAO_NUVEM", 40.0)) + reserva_de_fallback()
    return min(max(piso, necessario), _num("PRAZO_MAXIMO_S", 420.0))


def prazo_da_chamada(model: str, max_tokens: int, restante: float, piso: float = 0.0) -> float:
    """Timeout of one provider call; while the deadline allows, it leaves room for a fallback attempt."""
    raciocina = not (model or "").startswith("ollama/")
    desejado = max(tempo_esperado(model, tokens_totais(max_tokens, raciocina)), piso)
    reserva = _num("RESERVA_FALLBACK_S", 45.0)
    teto = restante - reserva if restante > 2 * reserva else restante - 0.5
    return max(1.0, min(desejado, teto))


def reserva_de_fallback() -> float:
    """Seconds a fallback attempt needs; below this the chain stops."""
    return _num("RESERVA_FALLBACK_S", 45.0)


def ordem_de_fallback(candidatos: List[str], primario: str) -> List[str]:
    """Fallback order: cloud models first (a saturated GPU makes a second local try the likeliest to fail)."""
    resto = [m for m in dict.fromkeys(candidatos) if m != primario]
    return [m for m in resto if not m.startswith("ollama/")] + [m for m in resto if m.startswith("ollama/")]
