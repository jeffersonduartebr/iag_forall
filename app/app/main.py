import json
import os
import time
import asyncio
import logging
from typing import List
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import PlainTextResponse

from .settings import settings
from .schemas import QueryRequest, QueryResponse, CandidateResult, RouteDecision
from .observability import *
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
logger = logging.getLogger(__name__)

logging.getLogger("chromadb.telemetry").setLevel(logging.ERROR)


app = FastAPI(title="LLM Router (Hybrid Bandit + NSGA-II + RAG + Judges)")

# ------------------------------------------------------
# Roteador de RAG (upload e gestão de documentos)
# ------------------------------------------------------
app.include_router(rag_router.router)

# ------------------------------------------------------
# Métricas Prometheus
# ------------------------------------------------------
@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest().decode("utf-8"))

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
                logger.warning("[warmup] Redis indisponível — seguindo sem cache por enquanto.")

            # 1️⃣ Garante modelos disponíveis no Ollama
            ollama_models = [
                settings.OLLAMA_MODEL,
                os.getenv("EMBED_MODEL", "nomic-embed-text"),
                "phi4",
                "deepseek-r1:8b",
            ]
            for model in ollama_models:
                try:
                    _ensure_ollama_model(model)
                except Exception as e:
                    logger.warning(f"[warmup] Falha ao verificar modelo '{model}': {e}")

            # 2️⃣ Adiciona documento base no RAG
            add_document(
                "intro",
                "NSGA-II is a multi-objective evolutionary algorithm used for Pareto optimization."
            )

            # 3️⃣ Executa requisições de teste (await async)
            samples = [
                "Explique em 3 tópicos o que é NSGA-II e onde é aplicado.",
                "Escreva um snippet Python que lê um CSV e calcula a média de uma coluna.",
                "Resuma boas práticas para documentação de APIs REST.",
            ]
            for s in samples:
                try:
                    _ = await route_and_answer(s)
                    logger.info(f"[warmup] Execução de teste concluída: '{s[:40]}...'")
                except Exception as e:
                    logger.warning(f"[warmup] Falha no teste de prompt: {e}")

            logger.info("[warmup] Rotina de inicialização concluída com sucesso ✅")

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
    logger.info(f"[query] Nova requisição recebida: '{req.query[:80]}...'")

    try:
        result = await route_and_answer(
            query=req.query,
            system_prompt=getattr(req, "system_prompt", ""),
            use_rag=getattr(req, "enable_rag_for_answer", False),
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

    # Recompensa acoplada ao NSGA-II
    reward = compute_reward(chosen_model or "unknown", quality, latency)
    bandit_update(chosen_model or "unknown", req.query, reward)

    # Atualiza métricas Prometheus
    BANDIT_REWARD.observe(reward)
    ROUTER_CHOSEN.labels(model=chosen_model or "cached").inc()
    CANDIDATE_COST.observe(cost)
    CANDIDATE_LAT.observe(latency)

    # Monta saída compatível
    route = RouteDecision(
        chosen_model=chosen_model or "cached",
        objectives={"cost": cost, "latency": latency, "neg_quality": max(0.0, 10.0 - quality)},
        pareto_front=[],
        explanation=f"reward={reward:.3f}, q={quality:.2f}, c={cost:.4f}, l={latency:.2f}",
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
# Administração: gerenciamento de juízes
# ------------------------------------------------------
@app.get("/admin/judges", response_model=List[str])
async def get_judges():
    return settings.JUDGE_MODELS


@app.post("/admin/judges", response_model=List[str])
async def update_judges(models: List[str], x_admin_token: str = Header(None)):
    if x_admin_token != settings.ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Token inválido.")

    if not models or not all(isinstance(m, str) for m in models):
        raise HTTPException(status_code=400, detail="A lista de juízes deve conter strings válidas.")

    old = getattr(settings, "JUDGE_MODELS", [])
    settings.JUDGE_MODELS = models
    logger.info(f"[Admin] Juízes atualizados de {old} para {models}")
    with open(".env", "a") as f:
        f.write(f"\nJUDGE_MODELS={json.dumps(models)}\n")

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
    }

