# Objective: Keep judges independent from the judged: no judge from the same company as the evaluated model.
"""Company of a model and the judge filter that enforces independence.

A model judging an answer from its own company (Gemini judging Gemma, Claude judging Claude) shares training
data and stylistic preferences with it, which biases the quality signal that feeds the bandit and the NSGA-II.
The evaluated model is carried in a ``ContextVar`` set by ``judge_answer`` so every judge path (pair, rubric,
meta-judge) applies the same rule without threading a parameter through each call.
"""

from __future__ import annotations

import functools
from contextvars import ContextVar
from typing import Any, Callable, List, Optional

#: Marker in the model name -> company. First match wins (specific before generic).
EMPRESAS = (
    ("gemma", "google"), ("gemini", "google"), ("palm", "google"),
    ("claude", "anthropic"), ("anthropic", "anthropic"),
    ("gpt", "openai"), ("openai", "openai"), ("o1-", "openai"), ("o3-", "openai"),
    ("deepseek", "deepseek"), ("granite", "ibm"), ("llama", "meta"), ("qwen", "alibaba"),
    ("mistral", "mistral"), ("mixtral", "mistral"), ("phi", "microsoft"), ("grok", "xai"),
    ("glm", "zhipu"), ("z-ai/", "zhipu"), ("kimi", "moonshot"), ("moonshotai/", "moonshot"),
)
_ROTAS = ("ollama/", "openrouter/", "gemini/", "anthropic/", "openai/")

MODELO_AVALIADO: ContextVar[Optional[str]] = ContextVar("modelo_avaliado", default=None)


def empresa(model: str) -> str:
    """Company behind a model id (``ollama/gemma4:12b`` -> ``google``); unknown ids are their own company."""
    nome = (model or "").lower()
    rota = next((r for r in _ROTAS if nome.startswith(r)), "")
    corpo = nome[len(rota):]
    for marcador, dona in EMPRESAS:
        if marcador in corpo:
            return dona
    return {"gemini/": "google", "anthropic/": "anthropic", "openai/": "openai"}.get(rota, corpo or nome)


def elegiveis(modelos: List[str], avaliado: Optional[str] = None) -> List[str]:
    """Judges allowed for the evaluated model (the context's, when not given)."""
    alvo = avaliado or MODELO_AVALIADO.get()
    if not alvo:
        return list(modelos)
    proibida = empresa(alvo)
    return [m for m in modelos if empresa(m) != proibida]


def com_modelo_avaliado(func: Callable[..., Any]) -> Callable[..., Any]:
    """Accept ``evaluated_model=`` on an async judge entry point and expose it to every judge path below."""

    @functools.wraps(func)
    async def envolvida(*args: Any, evaluated_model: Optional[str] = None, **kwargs: Any) -> Any:
        token = MODELO_AVALIADO.set(evaluated_model)
        try:
            return await func(*args, **kwargs)
        finally:
            MODELO_AVALIADO.reset(token)

    return envolvida
