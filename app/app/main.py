from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from .settings import settings
from .schemas import QueryRequest, QueryResponse, CandidateResult, JudgeScore, RouteDecision
from .observability import *
from .providers import safe_call_model as call_model
from .bandits import select_model as bandit_select, update_model as bandit_update
from .router_strategy import choose_top2_models, update_metrics
from .rag import retrieve_context
import os, time, asyncio, logging

# ------------------------------------------------------
# Configurações gerais
# ------------------------------------------------------
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Router (Hybrid Bandit + NSGA-II + RAG + Judges)")

# ------------------------------------------------------
# Endpoint de métricas
# ------------------------------------------------------
@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest().decode("utf-8"))

# ------------------------------------------------------
# Warmup assíncrono
# ------------------------------------------------------
@app.on_event("startup")
async def on_startup():
    async def _bg():
        try:
            logger.info("warmup_begin")
            samples = [
                "Explique em 3 tópicos o que é NSGA-II e onde é aplicado.",
                "Escreva um snippet Python que lê um CSV e calcula a média de uma coluna.",
                "Traceback: KeyError: 'id' durante um ETL. O que verificar?",
                "Resuma boas práticas para documentação de APIs REST.",
            ]
            for s in samples:
                await _simulate_call(s)
            logger.info("warmup_end")
        except Exception as e:
            logger.exception(f"warmup_failed: {e}")

    asyncio.create_task(_bg())

# ------------------------------------------------------
# Função auxiliar para teste de modelos no warmup
# ------------------------------------------------------
async def _simulate_call(q: str):
    """Executa chamadas simuladas durante o warmup e registra desempenho por modelo."""
    logger.info(f"[warmup] iniciando teste com prompt: '{q[:60]}...'")

    candidate_models = [
        settings.OLLAMA_MODEL,
        settings.COMMERCIAL_MODEL_1,
        settings.COMMERCIAL_MODEL_2,
    ]
    top2 = choose_top2_models(candidate_models, min_quality=settings.QUALITY_MIN, query_text=q)

    for model_name in top2:
        try:
            start_time = time.time()
            api_base = settings.OLLAMA_BASE_URL if model_name == settings.OLLAMA_MODEL else None
            text, meta = call_model(
                model=model_name,
                prompt=q,
                max_tokens=512,
                temperature=0.5,
                api_base=api_base,
            )
            elapsed = time.time() - start_time
            latency = meta.get("latency_s", 0.0) if isinstance(meta, dict) else 0.0
            text_len = len(text or "")
            logger.info(
                f"[warmup] Modelo={model_name} | Tempo={elapsed:.2f}s | "
                f"Tamanho={text_len} chars | Latência={latency:.2f}s"
            )
        except Exception as e:
            logger.error(f"[warmup] Falha ao testar {model_name}: {e}")

# ------------------------------------------------------
# Endpoint principal
# ------------------------------------------------------
@app.post("/query", response_model=QueryResponse)
async def route_query(req: QueryRequest):
    start = time.time()
    API_REQUESTS.inc()
    resp = await _route_logic(req)
    API_LATENCY.observe(time.time() - start)
    return resp

# ------------------------------------------------------
# Lógica central do roteamento LLM
# ------------------------------------------------------
async def _route_logic(req: QueryRequest) -> QueryResponse:
    candidate_models = [
        settings.OLLAMA_MODEL,
        settings.COMMERCIAL_MODEL_1,
        settings.COMMERCIAL_MODEL_2,
    ]

    bandit_choice = bandit_select(
        candidate_models, req.query,
        temperature=req.temperature,
        enable_rag_for_answer=req.enable_rag_for_answer
    )
    BANDIT_SELECT.labels(model=bandit_choice).inc()

    top2 = choose_top2_models(candidate_models, min_quality=settings.QUALITY_MIN, query_text=req.query)
    if bandit_choice in top2:
        chosen_order = [bandit_choice] + [m for m in top2 if m != bandit_choice]
    else:
        chosen_order = [
            bandit_choice,
            top2[0] if top2 else (
                candidate_models[0] if bandit_choice != candidate_models[0] else candidate_models[1]
            ),
        ]

    rag_context_for_answer = retrieve_context(req.query) if req.enable_rag_for_answer else ""
    full_prompt = req.query if not rag_context_for_answer else f"""Use o contexto se relevante:
---
{rag_context_for_answer}
---
Usuário: {req.query}
"""

    async def run_once(model_name: str):
        api_base = settings.OLLAMA_BASE_URL if model_name == settings.OLLAMA_MODEL else None
        logger.info(f"[router] Chamando modelo {model_name} (base={api_base or 'litellm default'})")

        text, meta = call_model(
            model=model_name,
            prompt=full_prompt,
            max_tokens=min(req.max_tokens, settings.MAX_TOKENS),
            temperature=req.temperature,
            api_base=api_base,
        )

        # Normaliza meta
        if not isinstance(meta, dict):
            logger.warning(f"[router] Meta retornou tipo {type(meta)}, redefinindo para dict padrão.")
            meta = {"usage": {"prompt_tokens": 0, "completion_tokens": 0}, "latency_s": 0.0}

        usage = meta.get("usage", {})
        pt = int(usage.get("prompt_tokens", 0))
        ct = int(usage.get("completion_tokens", 0))
        latency = float(meta.get("latency_s", 0.0))
        per_1k = settings.COSTS_USD_PER_1K.get(model_name, 0.5)
        cost = per_1k * ((pt + ct) / 1000.0)

        from . import judges
        scores = await judges.judge_answer(req.query, text, use_rag=settings.ENABLE_RAG_FOR_JUDGES)
        for s in scores:
            JUDGE_SCORE.labels(judge_id=s["judge_id"]).observe(s["score"])
        quality = sum(s["score"] for s in scores) / len(scores) if scores else 0.0

        CANDIDATE_COST.observe(cost)
        CANDIDATE_LAT.observe(latency)
        return text, pt, ct, latency, cost, scores, quality

    # Primeira execução
    chosen_model = chosen_order[0]
    text, pt, ct, latency, cost, scores, quality = await run_once(chosen_model)

    norm_cost = cost * 100.0
    reward = float(quality) - 0.2 * latency - 0.1 * norm_cost
    BANDIT_REWARD.observe(reward)
    bandit_update(chosen_model, req.query, reward, temperature=req.temperature, enable_rag_for_answer=req.enable_rag_for_answer)
    BANDIT_UPDATE.labels(model=chosen_model).inc()

    tried_fallback = False
    if quality < settings.QUALITY_MIN and len(chosen_order) > 1:
        second_model = chosen_order[1]
        text2, pt2, ct2, latency2, cost2, scores2, quality2 = await run_once(second_model)
        norm_cost2 = cost2 * 100.0
        reward2 = float(quality2) - 0.2 * latency2 - 0.1 * norm_cost2
        BANDIT_REWARD.observe(reward2)
        bandit_update(second_model, req.query, reward2, temperature=req.temperature, enable_rag_for_answer=req.enable_rag_for_answer)
        BANDIT_UPDATE.labels(model=second_model).inc()

        first_tuple = (quality, cost, chosen_model, text, pt, ct, latency, scores)
        second_tuple = (quality2, cost2, second_model, text2, pt2, ct2, latency2, scores2)
        best = second_tuple if (quality2 > settings.QUALITY_MIN and quality2 > quality) or (
            quality2 >= quality - 0.2 and cost2 < cost
        ) else first_tuple

        if best is second_tuple:
            FALLBACK_USED.labels(first_model=chosen_model, second_model=second_model).inc()
            chosen_model, text, pt, ct, latency, cost, scores, quality = (
                best[2],
                best[3],
                best[4],
                best[5],
                best[6],
                best[1],
                best[7],
                best[0],
            )
            tried_fallback = True

    update_metrics(chosen_model, cost=cost, latency=latency, quality=quality)
    ROUTER_CHOSEN.labels(model=chosen_model).inc()

    chosen_candidate = CandidateResult(
        model=chosen_model,
        output=text,
        latency_s=latency,
        prompt_tokens=pt,
        completion_tokens=ct,
        estimated_cost_usd=cost,
        judge_scores=[JudgeScore(**s) for s in scores],
        quality_score=quality,
    )

    explanation = (
        f"Bandit escolheu {chosen_model}; fallback={'sim' if tried_fallback else 'não'}. "
        f"Q={quality:.2f}, $={cost:.4f}, lat={latency:.2f}s."
    )

    route = RouteDecision(
        chosen_model=chosen_model,
        objectives={"cost": cost, "latency": latency, "neg_quality": max(0.0, 10.0 - quality)},
        pareto_front=[],
        explanation=explanation,
    )

    return QueryResponse(answer=text, model=chosen_model, route=route, candidates=[chosen_candidate])
