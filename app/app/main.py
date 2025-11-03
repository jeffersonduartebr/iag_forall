# -*- coding: utf-8 -*-
"""
main.py
----------------------------------------------------
Ponto de entrada da API principal do LLM Router Stack.

Inclui:
- Endpoint /metrics integrado ao registry global (Prometheus multiprocess).
- Warmup assíncrono com preload de modelos Ollama e base RAG.
- Roteamento híbrido (Bandit + NSGA-II + RAG + Juízes).
- Administração dinâmica via Redis + MariaDB.
- Persistência de logs (query_log) com recompensa.
"""

import json
import os
import time
import asyncio
import logging
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Header, Body, Response
from fastapi.responses import JSONResponse
from .prometheus_setup import setup_prometheus, prometheus_metrics
from .metrics_collector import _ensure_model_metrics_table
from .settings_dynamic import settings
from .schemas import QueryRequest, QueryResponse, CandidateResult, RouteDecision
from .observability import (
    setup_logging,
    logger,
    registry,
    render_metrics_response,
    API_REQUESTS,
    API_LATENCY,
    ROUTER_CHOSEN,
    CANDIDATE_COST,
    CANDIDATE_LAT,
    BANDIT_REWARD,
)
from .providers import _ensure_ollama_model
from .router_core import route_and_answer
from .rag_local import add_document
from .bandits import bandit_update, compute_reward
from .utils.redis_client import get_redis
from .routers import rag_router

# ------------------------------------------------------
# Inicialização e configuração global
# ------------------------------------------------------
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
setup_logging()
logging.getLogger("chromadb.telemetry").setLevel(logging.ERROR)

# Inicializa Prometheus antes do app
setup_prometheus()
app = FastAPI(title="LLM Router (Hybrid Bandit + NSGA-II + RAG + Judges)")

# ------------------------------------------------------
# Roteador de RAG (upload e gestão de documentos)
# ------------------------------------------------------
app.include_router(rag_router.router)

# ------------------------------------------------------
# Métricas Prometheus (usando registry global)
# ------------------------------------------------------
@app.get("/metrics")
def metrics():
    """Endpoint Prometheus com suporte a multiprocess."""
    data, ctype = render_metrics_response()
    return Response(content=data, media_type=ctype)

# ------------------------------------------------------
# Rotina de warmup assíncrona
# ------------------------------------------------------
@app.on_event("startup")
async def on_startup():
    async def _bg():
        try:
            logger.info("[warmup] Iniciando rotina de inicialização...")

            # Espera Redis ficar pronto
            r = get_redis(max_wait_s=45)
            if r is None:
                logger.warning("[warmup] Redis indisponível — seguindo sem cache.")
            # 0) Garante tabela de métricas para evitar erros de leitura
            try:
                _ensure_model_metrics_table()
            except Exception as e:
                logger.warning("[warmup] Falha ao garantir model_metrics: %s", e)

            # Garante modelos Ollama
            ollama_models = [
                m.replace("ollama/", "") for m in settings.CANDIDATE_MODELS_LIST
                if isinstance(m, str) and m.startswith("ollama/")
            ]
            essentials = [settings.EMBED_MODEL]
            for j in settings.JUDGE_MODELS:
                if isinstance(j, str) and j.startswith("ollama/"):
                    essentials.append(j.replace("ollama/", ""))
                elif isinstance(j, str) and (":" in j and "/" not in j):
                    essentials.append(j)
            all_models = list({*ollama_models, *essentials})

            for model in all_models:
                try:
                    await asyncio.to_thread(_ensure_ollama_model, model)
                except Exception as e:
                    logger.warning(f"[warmup] Falha ao verificar modelo '{model}': {e}")

            # Documento base de exemplo no RAG
            await add_document(
                "intro",
                "NSGA-II é um algoritmo evolutivo multiobjetivo usado em otimização de Pareto.",
            )

            # Execução de teste inicial
            samples = [
                "Explique o que é NSGA-II e onde é aplicado.",
                "Escreva um código Python que calcule média de uma coluna CSV.",
                "O que é RAG em sistemas de IA?",
            ]
            for s in samples:
                try:
                    _ = await route_and_answer(s)
                    logger.info(f"[warmup] Execução de teste concluída: '{s[:40]}...'")
                except Exception as e:
                    logger.warning(f"[warmup] Falha no teste de prompt: {e}")

            logger.info("[warmup] Inicialização concluída ✅")

        except Exception as e:
            logger.exception(f"[warmup] Falhou: {e}")

    asyncio.create_task(_bg())

# ------------------------------------------------------
# Endpoint principal de inferência / roteamento
# ------------------------------------------------------
@app.post("/query", response_model=QueryResponse)
async def route_query(req: QueryRequest):
    start = time.time()
    API_REQUESTS.inc()
    if not req or not isinstance(req.query, str) or not req.query.strip():
        raise HTTPException(status_code=400, detail="Campo 'query' é obrigatório.")
    logger.info(f"[query] Nova requisição recebida: '{req.query[:80]}...'")

    try:
        result = await route_and_answer(
            query=req.query,
            system_prompt=(getattr(req, "system_prompt", "") or ""),
            use_rag=getattr(req, "enable_rag_for_answer", False),
            max_tokens=(req.max_tokens or settings.MAX_TOKENS_DEFAULT),
            temperature=(req.temperature or settings.TEMPERATURE_DEFAULT),
        )
    except Exception as e:
        logger.exception(f"[router] Erro interno durante o roteamento: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    API_LATENCY.observe(time.time() - start)

    chosen_model = result["model"]
    latency = float(result["latency_s"])
    quality = float(result["quality"])
    cost = float(result["cost_per_1k"])
    text = result["answer"]

    # Recompensa NSGA-II / Bandit
    try:
        reward = compute_reward(chosen_model or "unknown", quality, latency)
        bandit_update(chosen_model or "unknown", req.query, reward)
        BANDIT_REWARD.observe(reward)
    except Exception as e:
        logger.warning(f"[bandit] Falha ao calcular/atualizar recompensa: {e}")
        reward = 0.0

    # Persistência
    try:
        from .services.query_service import insert_query_log
        insert_query_log(
            query=req.query,
            model=chosen_model,
            response=text,
            latency=latency,
            cost=cost,
            quality=quality,
            reward=reward,
        )
        logger.info(f"[db] query_log registrado para modelo={chosen_model}")
    except Exception as e:
        logger.warning(f"[db] Falha ao registrar query_log: {e}")

    ROUTER_CHOSEN.labels(model=chosen_model or "cached").inc()
    CANDIDATE_COST.observe(cost)
    CANDIDATE_LAT.observe(latency)

    route = RouteDecision(
        chosen_model=chosen_model or "cached",
        objectives={"cost": cost, "latency": latency, "neg_quality": max(0.0, 10.0 - quality)},
        pareto_front=[],
        explanation=f"q={quality:.2f}, c={cost:.4f}, l={latency:.2f}",
    )

    candidate = CandidateResult(
        model=chosen_model or "cached",
        output=text,
        latency_s=latency,
        prompt_tokens=0,
        completion_tokens=0,
        estimated_cost_usd=cost,
        judge_scores=[],
        quality_score=quality,
    )

    return QueryResponse(answer=text, model=chosen_model, route=route, candidates=[candidate])

# ------------------------------------------------------
# Administração de settings
# ------------------------------------------------------
def _require_admin(token: Optional[str]):
    if token != settings.ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Token inválido.")

@app.get("/admin/settings")
def get_all_settings(x_admin_token: Optional[str] = Header(None)):
    _require_admin(x_admin_token)
    return settings.snapshot(only_known=False)

@app.put("/admin/settings")
def set_settings(payload: Dict[str, Any] = Body(...), x_admin_token: Optional[str] = Header(None)):
    _require_admin(x_admin_token)
    if not isinstance(payload, dict) or not payload:
        raise HTTPException(status_code=400, detail="Payload inválido.")
    for k, v in payload.items():
        settings.set(k, v, actor="api", source="admin")
    return {"ok": True, "applied": sorted(payload.keys())}

@app.get("/admin/judges", response_model=List[str])
async def get_judges():
    return settings.JUDGE_MODELS

@app.post("/admin/judges", response_model=List[str])
async def update_judges(models: List[str], x_admin_token: str = Header(None)):
    _require_admin(x_admin_token)
    if not models or not all(isinstance(m, str) for m in models):
        raise HTTPException(status_code=400, detail="A lista de juízes deve conter strings válidas.")
    settings.set("JUDGE_MODELS", models, actor="api", source="admin")
    return settings.JUDGE_MODELS

# ------------------------------------------------------
# Healthcheck
# ------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "ollama_base": settings.OLLAMA_BASE_URL,
        "models_preloaded": True,
        "dynamic_settings": settings.snapshot(only_known=True),
    }
