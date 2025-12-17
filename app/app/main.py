# -*- coding: utf-8 -*-
"""
main.py (CORRIGIDO: Telemetria Chroma Desativada)
------------------------------------------
API Principal.
"""

# ==============================================================================
# 🚨 FIX: Desativar Telemetria do ChromaDB ANTES de qualquer import
# ==============================================================================
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
os.environ["SCARF_NO_ANALYTICS"] = "true"

import json
import time
import asyncio
import logging
from typing import List, Dict, Any, Optional

import requests
from fastapi import FastAPI, HTTPException, Header, Body, Response
from fastapi.responses import JSONResponse

from .prometheus_setup import setup_prometheus
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
)
from .router_core import route_and_answer
from .utils.redis_client import get_redis
from .routers import rag_router
from .vectorstore import init_vectorstore, add_document as vs_add_document
from .tasks import task_process_feedback

# Configuração de Logging
setup_logging()
# Força silêncio no logger do Chroma
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("posthog").setLevel(logging.CRITICAL)

setup_prometheus()

app = FastAPI(
    title="LLM/VLM Router (Hybrid Bandit + RAG + Celery Feedback)",
    version="3.1.2",
    description="API de Roteamento Inteligente Multimodal com Persistência Robusta."
)

# --- HELPER DE CONVERSÃO ---
def safe_parse_json(payload: Any) -> Any:
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except Exception:
            return payload
    return payload


OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
)
VLM_OLLAMA_MODELS = list(getattr(settings, "VLM_OLLAMA_MODELS", []))


async def preload_ollama_models():
    try:
        logger.info("[ollama-preload] Iniciando verificação...")
        candidates_raw = os.getenv("CANDIDATE_MODELS_LIST", "[]")
        judges_raw = os.getenv("JUDGE_MODELS", "[]")
        main_model = os.getenv("OLLAMA_MODEL", "")
        embed_model = settings.get("EMBED_TEXT_MODEL", "all-minilm")

        all_models = []
        for raw in (candidates_raw, judges_raw):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    all_models.extend(parsed)
            except json.JSONDecodeError:
                all_models.extend(raw.split(","))

        if main_model:
            all_models.append(main_model)

        for name in VLM_OLLAMA_MODELS:
            all_models.append(f"ollama/{name}")

        if embed_model:
            if not embed_model.startswith("ollama/"):
                all_models.append(f"ollama/{embed_model}")
            else:
                all_models.append(embed_model)
        
        all_models.append("ollama/all-minilm")

        all_models = [m.strip() for m in all_models if m.strip().startswith("ollama/")]
        all_models = list(set(all_models))

        if not all_models:
            return

        try:
            resp = await asyncio.to_thread(requests.get, f"{OLLAMA_HOST}/api/tags", timeout=10)
            resp.raise_for_status()
            available = {m["name"] for m in resp.json().get("models", [])}
        except Exception:
            available = set()

        for model in all_models:
            name = model.split("/", 1)[1]
            if any(name in avail for avail in available) or f"{name}:latest" in available:
                logger.info(f"[ollama-preload] '{name}' já disponível.")
                continue

            logger.info(f"[ollama-preload] Baixando '{name}'...")
            try:
                with requests.post(
                    f"{OLLAMA_HOST}/api/pull",
                    json={"name": name},
                    stream=True,
                    timeout=1200,
                ) as r:
                    for _ in r.iter_lines(decode_unicode=True): pass 
                logger.info(f"[ollama-preload] '{name}' OK.")
            except Exception as e:
                logger.error(f"[ollama-preload] Falha ao baixar '{name}': {e}")

        logger.info("[ollama-preload] Concluído. ✅")
    except Exception as e:
        logger.exception(f"[ollama-preload] Erro geral: {e}")


app.include_router(rag_router.router)


@app.get("/metrics")
def metrics():
    data, ctype = render_metrics_response()
    return Response(content=data, media_type=ctype)


@app.on_event("startup")
async def startup_event():
    async def _bg():
        try:
            logger.info("[warmup] Iniciando serviços...")
            r = get_redis()
            if r is None: logger.warning("[warmup] Redis indisponível.")
            
            try: _ensure_model_metrics_table()
            except: pass

            try:
                init_vectorstore()
                logger.info("[warmup] Vectorstore OK.")
            except Exception as e:
                logger.warning(f"[warmup] Vectorstore init falhou: {e}")

            await preload_ollama_models()

            try:
                await vs_add_document(
                    modality="text", doc_id="intro",
                    text="NSGA-II é um algoritmo de otimização multiobjetivo baseado em dominância de Pareto.",
                    metadata={"warmup": True},
                )
            except Exception as e:
                logger.warning(f"[warmup] Falha doc teste: {e}")

            # =================================================================
            # 🔥 SMOKE TESTS (10 Áreas do Conhecimento)
            # =================================================================
            tests = [
                "Explique a diferença entre processamento síncrono e assíncrono em Python.",
                "Quais foram as principais consequências geopolíticas da Queda do Muro de Berlim?",
                "Como funciona o princípio da incerteza de Heisenberg em termos simples?",
                "Escreva um breve poema no estilo barroco sobre a passagem do tempo.",
                "Se um trem viaja a 120km/h, quanto tempo leva para percorrer 450km?",
                "Descreva o processo de fotossíntese e sua importância para a vida na Terra.",
                "Qual a relação entre taxa de juros e inflação em uma economia de mercado?",
                "Resuma o conceito de 'Imperativo Categórico' de Immanuel Kant.",
                "O que são direitos fundamentais segundo a Constituição Brasileira de 1988?",
                "Quais são as principais características culturais e culinárias do Nordeste brasileiro?",
                "Explique como ocorreu a separação dos continentes ao longo dos milênios.",
                "Se eu tenho um triângulo retângulo cujos valores dos catetos são 8 e 10, qual é o valor da hipotenusa?",
                "Cite três características do mercantilismo."
            ]

            logger.info(f"[warmup] Executando {len(tests)} smoke tests diversificados...")
            
            for i, t in enumerate(tests, 1):
                try:
                    start_t = time.time()
                    await route_and_answer(query=t, modality="text", use_cache=False)
                    elapsed = time.time() - start_t
                    logger.info(f"[warmup] Teste {i}/10 OK ({elapsed:.2f}s): '{t[:40]}...'")
                except Exception as e:
                    logger.warning(f"[warmup] Falha no teste {i} ('{t[:20]}...'): {e}")

            logger.info("[warmup] Sistema pronto e aquecido. ✅")
        except Exception as e:
            logger.exception(f"[warmup] Erro crítico: {e}")

    asyncio.create_task(_bg())


@app.post("/query", response_model=QueryResponse)
async def route_query(req: QueryRequest):
    start = time.time()
    API_REQUESTS.inc()

    if not req or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query obrigatória.")

    logger.info(f"[query] '{req.query[:60]}...' (mod={req.modality})")

    modality = (req.modality or "text").lower()
    image_input = req.image_b64
    if not image_input and req.images and len(req.images) > 0:
        image_input = req.images[0]
    
    if image_input and modality == "text":
        modality = "vision"

    try:
        result = await route_and_answer(
            query=req.query,
            system_prompt=req.system_prompt or "",
            use_rag=bool(req.enable_rag_for_answer or req.enable_rag_for_image),
            max_tokens=req.max_tokens or settings.MAX_TOKENS_DEFAULT,
            temperature=req.temperature or settings.TEMPERATURE_DEFAULT,
            modality=modality,
            image_b64=image_input,
            rag_modality=(req.rag_modality or "text").lower(),
            use_cache=req.use_cache
        )
    except Exception as e:
        logger.exception(f"[router] Erro: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    duration = time.time() - start
    API_LATENCY.observe(duration)

    chosen_model = result["model"]
    ROUTER_CHOSEN.labels(model=chosen_model).inc()
    CANDIDATE_COST.observe(result["cost_per_1k"])
    CANDIDATE_LAT.observe(result["latency_s"])

    raw_payload_str = result.get("metadata", {}).get("raw_payload")
    
    try:
        task_process_feedback.delay(
            query=req.query,
            answer=result["answer"],
            chosen_model=chosen_model,
            modality=result["modality"],
            latency_s=result["latency_s"],
            cost_val=result["cost_per_1k"],
            image_b64=image_input,
            raw_payload=raw_payload_str,
            prompt_tokens=result.get("metadata", {}).get("prompt_tokens", 0),
            completion_tokens=result.get("metadata", {}).get("completion_tokens", 0)
        )
    except Exception as e:
        logger.error(f"[main] Falha ao despachar tarefa Celery: {e}")

    parsed_payload = safe_parse_json(raw_payload_str)
    route_raw = result.get("route", {})
    candidates_raw = result.get("candidates", [])

    return QueryResponse(
        answer=result["answer"],
        model=chosen_model,
        modality=result["modality"],
        image_output_b64=result.get("image_output_b64"),
        route=RouteDecision(**route_raw),
        candidates=[CandidateResult(**c) for c in candidates_raw],
        payload=parsed_payload
    )


def _require_admin(token: Optional[str]):
    if token != settings.ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Token inválido.")

@app.get("/admin/settings", tags=["Admin"])
def get_settings(x_admin_token: Optional[str] = Header(None)):
    _require_admin(x_admin_token)
    return settings.snapshot()

@app.put("/admin/settings", tags=["Admin"])
def update_settings(payload: Dict[str, Any], x_admin_token: Optional[str] = Header(None)):
    _require_admin(x_admin_token)
    for k, v in payload.items():
        val = json.dumps(v) if isinstance(v, (list, dict)) else str(v)
        settings.set(k, val, actor="api", source="admin")
    return {"status": "updated"}

@app.get("/health", tags=["Ops"])
def health():
    return {"status": "ok", "timestamp": time.time()}
