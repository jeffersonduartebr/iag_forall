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

            # 1️⃣ Garante modelos disponíveis no Ollama (AGORA DINÂMICO)
            logger.info("[warmup] Iniciando rotina de inicialização...")
            
            # Pega todos os modelos candidatos que são do Ollama
            ollama_models_from_settings = [
                m.replace("ollama/", "") for m in settings.CANDIDATE_MODELS_LIST 
                if m.startswith("ollama/")
            ]
            
            # Adiciona outros modelos essenciais (embedding, juízes)
            essentials = [
                os.getenv("EMBED_MODEL", "nomic-embed-text"),
            ]
            for judge_model in settings.JUDGE_MODELS:
                 if judge_model.startswith("ollama/"):
                    essentials.append(judge_model.replace("ollama/", ""))
                 # Heurística para modelos locais sem prefixo
                 elif ":" in judge_model and not "/" in judge_model:
                    essentials.append(judge_model)

            # Combina e remove duplicatas
            all_ollama_models = list(set(ollama_models_from_settings + essentials))
            
            logger.info(f"[warmup] Garantindo modelos Ollama: {all_ollama_models}")
            
            for model in all_ollama_models:
                if not model: continue
                try:
                    await asyncio.to_thread(_ensure_ollama_model, model)
                except Exception as e:
                    logger.warning(f"[warmup] Falha ao verificar modelo '{model}': {e}")

            # 2️⃣ Adiciona documento base no RAG
            await add_document(
                            "intro",
                            "NSGA-II is a multi-objective evolutionary algorithm used for Pareto optimization."
                        )

            # 3️⃣ Executa requisições de teste (await async)
            samples = [
                "Explique em 3 tópicos o que é NSGA-II e onde é aplicado.",
                "Escreva um snippet Python que lê um CSV e calcula a média de uma coluna.",
                "Resuma boas práticas para documentação de APIs REST.",
                "Quem foi Ada Lovelace e qual sua principal contribuição para a computação?",
                "Descreva o processo de fotossíntese em termos simples para um estudante.",
                "Qual a diferença entre Docker e uma Máquina Virtual (VM)?",
                "Crie uma função Javascript que busca dados de uma API usando 'fetch' e trata a resposta.",
                "O que é 'inflação' e como ela afeta o poder de compra?",
                "Escreva um parágrafo curto sobre a importância da ética no desenvolvimento de IA.",
                "Quais são os três principais tipos de machine learning? (Supervisionado, Não Supervisionado, Reforço)"
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
            max_tokens=req.max_tokens,
            temperature=req.temperature
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
    # Atualiza custos acumulados e economia
    ROUTER_MODEL_COST.labels(model=chosen_model).inc(cost)
    ROUTER_COST_PER_QUERY.set(cost)
    if "ollama" in chosen_model:
        ROUTER_COST_SAVINGS.inc(cost * 0.8)  # Exemplo: assume 80% de economia
        ROUTER_LOCAL_USAGE_RATIO.set(1.0)
    else:
        ROUTER_LOCAL_USAGE_RATIO.set(0.0)

    # Atualiza qualidade média (EMA simples)
    ROUTER_QUALITY_AVG.labels(model=chosen_model).set(quality)


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
    models_str = ",".join(models)
    # Função auxiliar síncrona (se não existir, adicione-a)
    def _write_env_file_sync(models_json: str):
        """Escreve de forma síncrona no arquivo .env."""
        try:
            # Use "w" (write) ou "a" (append) dependendo da sua estratégia
            # Usar "a" (append) é mais simples
            with open(".env", "a") as f:
                # Adiciona aspas ao redor da string
                f.write(f'\nJUDGE_MODELS="{models_json}"\n')
        except Exception as e:
            logger.error(f"[Admin] Falha ao escrever no .env: {e}")

    await asyncio.to_thread(_write_env_file_sync, models_str)

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

