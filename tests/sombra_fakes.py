# Objective: Shared fakes for the shadow-execution tests (config, job, providers, judges, persistence).
from __future__ import annotations

import dataclasses
import json
from typing import Any, Dict, List

from app.services.sombra import config, registro
from app.services.sombra.config import ConfigSombra

JUIZES = (
    "gemini/gemini-3.1-pro-preview",
    "openrouter/anthropic/claude-opus-5.5",
    "openrouter/x-ai/grok-4.7",
    "openrouter/openai/gpt-5.6-sol",
)


def cfg(**kw: Any) -> ConfigSombra:
    base = ConfigSombra(
        ligada=True, tenants=("ifrn-caso1",), taxa=1.0, estratos=("disciplina", "faixa_incerteza", "modalidade"),
        provedores=("gemini/", "openrouter/", "ollama/"), regiao="", max_concorrencia=16, max_local=1,
        orcamento_usd=4.0, teto_tenant_hora=30, timeout_s=5.0, fuso="America/Fortaleza", juizes=JUIZES,
    )
    return dataclasses.replace(base, **kw)


def ligar(monkeypatch, **kw: Any) -> ConfigSombra:
    c = cfg(**kw)
    monkeypatch.setattr(config, "carregar", lambda: c)
    return c


def job(**kw: Any) -> Dict[str, Any]:
    base = {
        "request_id": "req-1", "episode_id": "ep-1", "participante": "P001", "tenant": "ifrn-caso1", "caso": 1,
        "estrato": {"disciplina": "bd", "faixa_incerteza": "baixa", "modalidade": "text"}, "p_nominal": 0.15,
        "regime_entrega": "aproveitamento", "p_atribuicao": None, "frozen_run_id": None, "criado_em": "2026-11-20T10:00:00",
        "modelo_entregue": "gemini/gemini-3.8-flash", "resposta_entregue": "RESPOSTA ENTREGUE",
        "custo_entregue": 0.002, "latencia_entregue": 3.1, "tokens_entrada": 900, "tokens_saida": 300,
        "candidatas": ["openrouter/deepseek/deepseek-v4.1-flash", "openrouter/anthropic/claude-sonnet-5"],
        "pergunta": "O que é normalização?", "prompt_final": "PROMPT FINAL COM CONTEXTO", "system_prompt": "SYS",
        "contexto": "CONTEXTO RAG DA REQUISICAO", "modalidade": "text", "image_b64": None, "temperatura": 0.4,
        "max_tokens": 2048, "response_format": None, "incerteza": 0.2, "grounded": True,
        "retrieval_mode": "full_retrieval", "workload_class": "knowledge_lookup", "complexidade": "moderate",
    }
    base.update(kw)
    return base


class Modelos:
    """Fake ``call_model``: candidates answer ``resposta``; judges read it and score by ``notas[modelo avaliado]``."""

    def __init__(self, resposta: str = "RESPOSTA SOMBRA", notas: Dict[str, float] | None = None, falhas=None):
        self.resposta, self.notas, self.falhas = resposta, notas or {}, falhas or {}
        self.chamadas: List[Dict[str, Any]] = []

    async def __call__(self, **kw: Any):
        self.chamadas.append(kw)
        if kw["model"] in self.falhas:
            raise self.falhas[kw["model"]]
        meta = {"call_cost_usd": 0.01, "prompt_tokens": 100, "completion_tokens": 50,
                "raw_payload": json.dumps({"model": kw["model"] + "-2026"})}
        prompt = kw["prompt"]
        if "### RUBRICA" in prompt:
            nota = next((v for k, v in self.notas.items() if k in prompt), 6.0)
            return f'<scores>{{"clareza": {nota}, "acuracia": {nota}, "alinhamento": {nota}}}</scores>', meta
        if "NÍVEIS DE ENTREGA" in prompt:
            return '<entrega>{"nivel_entrega": 1, "evidencia": "x"}</entrega>', meta
        return f"{self.resposta} {kw['model']}", meta

    def de_candidatas(self) -> List[Dict[str, Any]]:
        return [c for c in self.chamadas if c["prompt"] == "PROMPT FINAL COM CONTEXTO"]


def capturar_registro(monkeypatch) -> Dict[str, list]:
    gravado: Dict[str, list] = {"linhas": [], "abertas": [], "fechadas": []}
    monkeypatch.setattr(registro, "gravar", lambda linhas: gravado["linhas"].extend(linhas) or len(gravado["linhas"]))
    monkeypatch.setattr(registro, "abrir_suspensao", lambda chave, *a: gravado["abertas"].append(chave))
    monkeypatch.setattr(registro, "fechar_suspensao", lambda chave, fim: gravado["fechadas"].append(chave))
    return gravado
