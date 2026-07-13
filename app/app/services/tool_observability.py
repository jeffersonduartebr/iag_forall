# -*- coding: utf-8 -*-
# Objective: Observability + bandit reward for tool/function-calling turns.
"""Efeitos colaterais de turnos de tool calling.

Turnos de *tool call* não têm resposta em texto para os juízes avaliarem, então o
pipeline normal de feedback (juízes → reward) é pulado (ver ``record_query_side_effects``
em ``query_runtime``). Este módulo devolve a esses turnos:

* **(a) Observabilidade** — contadores Prometheus de quantos turnos de tool foram
  emitidos, por função, quantos argumentos vieram malformados e a profundidade
  multi-turn (quantas mensagens já existiam na conversa).
* **(b) Reward para o bandit** — um sinal barato baseado na boa-formação das
  chamadas (fração de tool calls com argumentos JSON válidos). Sem isso o roteador
  nunca aprenderia com turnos de tool; com isso ele passa a preferir modelos que
  produzem tool calls efetivamente utilizáveis.

Nada aqui pode quebrar o caminho de request: toda emissão de métrica e todo update
de bandit é best-effort (``try/except``).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Sequence

from prometheus_client import Counter, Histogram

from app.observability import registry

logger = logging.getLogger(__name__)

# Métricas de tool calling — registradas no registry compartilhado para aparecerem
# em ``GET /metrics`` (definidas aqui, junto do código que as emite, em vez de inflar
# ``observability.py``, que já está no baseline de SLOC e será dividido no item #8).
TOOL_CALLS_TOTAL = Counter(
    "router_tool_calls_total",
    "Tool/function-calling turns emitted by the selected model",
    ["model"],
    registry=registry,
)
TOOL_CALL_FUNCTIONS_TOTAL = Counter(
    "router_tool_call_functions_total",
    "Individual tool functions requested across tool-calling turns",
    ["model", "function"],
    registry=registry,
)
TOOL_CALL_ARGS_INVALID = Counter(
    "router_tool_call_args_invalid_total",
    "Tool calls whose arguments failed to parse as valid JSON",
    ["model"],
    registry=registry,
)
TOOL_CALL_DEPTH = Histogram(
    "router_tool_call_depth",
    "Prior conversation turns when a tool call is emitted (multi-turn depth)",
    ["model"],
    buckets=(0, 1, 2, 3, 5, 8, 13, 21),
    registry=registry,
)


def _iter_tool_calls(tool_calls: Any) -> List[Dict[str, Any]]:
    """Return the canonical tool-call dicts from an arbitrary payload."""
    if not isinstance(tool_calls, Sequence) or isinstance(tool_calls, (str, bytes)):
        return []
    return [tc for tc in tool_calls if isinstance(tc, dict)]


def _arguments_valid(tool_call: Dict[str, Any]) -> bool:
    """Return whether one canonical tool call carries parseable JSON arguments."""
    fn = tool_call.get("function")
    if not isinstance(fn, dict):
        return False
    args = fn.get("arguments")
    if isinstance(args, dict):
        return True  # objeto já desserializado conta como válido
    if not isinstance(args, str):
        return False
    text = args.strip()
    if not text or text == "{}":
        return True  # chamada sem argumentos ("" / "{}") é bem-formada
    try:
        json.loads(text)
        return True
    except (ValueError, TypeError):
        return False


def _function_name(tool_call: Dict[str, Any]) -> str:
    """Best-effort extraction of the requested function name (for metric labels)."""
    fn = tool_call.get("function")
    if isinstance(fn, dict):
        name = fn.get("name")
        if isinstance(name, str) and name:
            return name
    return "unknown"


def tool_call_quality(tool_calls: Any) -> float:
    """Score tool-call well-formedness on ``0..1`` (fraction with valid JSON args).

    Empty or wholly-malformed payloads score ``0.0``. Used as a cheap quality proxy
    to reward the bandit for tool turns, which bypass the LLM-judge quality path.
    """
    calls = _iter_tool_calls(tool_calls)
    if not calls:
        return 0.0
    valid = sum(1 for tc in calls if _arguments_valid(tc))
    return valid / len(calls)


def record_tool_turn(
    *,
    chosen_model: str,
    tool_calls: Any,
    modality: str = "text",
    latency_s: float = 0.0,
    cost_val: float = 0.0,
    query: str = "",
    conversation_depth: int = 0,
) -> float:
    """Emit tool-call metrics and feed a well-formedness reward to the bandit.

    Returns the computed tool-call quality (``0..1``). Never raises: observability
    and reward feeding are best-effort and must not break the request path.
    """
    calls = _iter_tool_calls(tool_calls)
    quality = tool_call_quality(calls)

    try:
        TOOL_CALLS_TOTAL.labels(model=chosen_model).inc()
        TOOL_CALL_DEPTH.labels(model=chosen_model).observe(max(0, int(conversation_depth)))
        for tc in calls:
            TOOL_CALL_FUNCTIONS_TOTAL.labels(model=chosen_model, function=_function_name(tc)).inc()
            if not _arguments_valid(tc):
                TOOL_CALL_ARGS_INVALID.labels(model=chosen_model).inc()
    except Exception as exc:  # pragma: no cover - métrica jamais quebra o roteamento
        logger.debug("[tool-obs] metric emission failed: %s", exc)

    try:
        from app.bandits import bandit_update, compute_reward

        final_quality = round(quality * 10.0, 2)  # compute_reward espera quality em [0..10]
        reward = compute_reward(chosen_model, final_quality, float(latency_s or 0.0), float(cost_val or 0.0))
        bandit_update(model=chosen_model, query=query, reward=reward, modality=modality)
    except Exception as exc:
        logger.warning("[tool-obs] bandit reward for tool turn failed: %s", exc)

    return quality
