# -*- coding: utf-8 -*-
"""
main.py (Production-Ready with Rate Limiting, Compression, Health Checks)
-------------------------------------------------------------------------
API Principal with enterprise features:
- Rate limiting
- Gzip compression
- Deep health checks
- Circuit breaker integration
- Request deduplication
- API versioning support
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
import concurrent.futures
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Response, Request, APIRouter
from fastapi.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.gzip import GZipMiddleware

from .api import admin_router, eval_router, feedback_router, governance_router, ops_router
from .api.deps import require_admin as _require_admin
from .prometheus_setup import setup_prometheus
from .correlation import (
    set_correlation_id,
    get_correlation_id,
    clear_correlation_id,
    CORRELATION_ID_HEADER,
)
from .metrics_collector import _ensure_model_metrics_table
from .settings_dynamic import settings, start_reload_listener, stop_reload_listener
from .settings_dynamic import validate_critical_settings
from .schemas import QueryRequest, QueryResponse, CandidateResult, RouteDecision
from .config.constants import GZIP_MIN_SIZE
from .middleware.rate_limit import (
    RateLimitMiddleware,
    periodic_cleanup as rate_limit_cleanup,
)
from .middleware.backpressure import BackpressureMiddleware
from .observability import (
    setup_logging,
    logger,
    render_metrics_response,
    API_REQUESTS,
    API_LATENCY,
    ROUTER_CHOSEN,
    CANDIDATE_COST,
    CANDIDATE_LAT,
    TOTAL_COST_USD,
    COST_BY_PROVIDER,
    TOKENS_INPUT_TOTAL,
    TOKENS_OUTPUT_TOTAL,
)
from .router_core import start_background_services, stop_background_services
from .providers_async import close_http_client
from .utils.redis_client import get_redis, close_redis
from .db import close_engine
from .routers import rag_router
from .vectorstore import init_vectorstore, add_document as vs_add_document
from .services.query_runtime import process_query_request, record_query_side_effects
from .services.governance_runtime import ensure_runtime_support_tables
from .router_core import route_and_answer

# Configuração de Logging
setup_logging()
# Força silêncio no logger do Chroma
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("posthog").setLevel(logging.CRITICAL)

setup_prometheus()


# ==============================================================================
# FastAPI App with Versioning
# ==============================================================================
@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Executa lifespan."""
    await startup_event()
    try:
        yield
    finally:
        await shutdown_event()


app = FastAPI(
    title="LLM/VLM Router (Hybrid Bandit + RAG + Celery Feedback)",
    version="3.2.0",
    description="API de Roteamento Inteligente Multimodal com Persistência Robusta.",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.include_router(admin_router)
app.include_router(governance_router)
app.include_router(eval_router)
app.include_router(feedback_router)
app.include_router(ops_router)

# Create versioned router
v1_router = APIRouter(prefix="/v1", tags=["v1"])


# ==============================================================================
# Middleware Stack
# ==============================================================================

class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """
    Middleware that handles correlation ID propagation.
    - Extracts correlation ID from incoming X-Correlation-ID header
    - Generates a new one if not present
    - Adds the correlation ID to the response headers
    """

    async def dispatch(self, request: Request, call_next):
        # Get correlation ID from header or generate new one
        """Executa dispatch."""
        correlation_id = request.headers.get(CORRELATION_ID_HEADER)
        set_correlation_id(correlation_id)

        try:
            response = await call_next(request)
            # Add correlation ID to response headers
            response.headers[CORRELATION_ID_HEADER] = get_correlation_id() or ""
            return response
        finally:
            clear_correlation_id()


# Add middleware in order (last added = first executed)
app.add_middleware(CorrelationIdMiddleware)
app.add_middleware(BackpressureMiddleware)  # Global concurrency limit
app.add_middleware(RateLimitMiddleware)  # Uses Redis-based rate limiting
app.add_middleware(GZipMiddleware, minimum_size=GZIP_MIN_SIZE)


# --- HELPER DE CONVERSÃO ---
def safe_parse_json(payload: Any) -> Any:
    """Executa safe parse json."""
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
    """
    Preload Ollama models asynchronously using httpx.

    Uses httpx.AsyncClient for better performance compared to
    synchronous requests in async context.
    """
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

        # Use httpx.AsyncClient for async HTTP requests
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.get(f"{OLLAMA_HOST}/api/tags")
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
                # Use longer timeout for model downloads
                async with httpx.AsyncClient(timeout=1200.0) as client:
                    async with client.stream(
                        "POST",
                        f"{OLLAMA_HOST}/api/pull",
                        json={"name": name},
                    ) as response:
                        response.raise_for_status()
                        # Consume the stream to complete the download
                        async for _ in response.aiter_lines():
                            pass
                logger.info(f"[ollama-preload] '{name}' OK.")
            except Exception as e:
                logger.error(f"[ollama-preload] Falha ao baixar '{name}': {e}")

        logger.info("[ollama-preload] Concluído.")
    except Exception as e:
        logger.exception(f"[ollama-preload] Erro geral: {e}")


app.include_router(rag_router.router)


@app.get("/metrics")
def metrics():
    """Executa metrics."""
    data, ctype = render_metrics_response()
    return Response(content=data, media_type=ctype)


async def startup_event():
    """Executa startup event."""
    admin_token = (settings.ADMIN_TOKEN or "").strip()
    if not admin_token or admin_token == "changeme-please":
        raise RuntimeError("ADMIN_TOKEN must be configured with a non-default value")

    config_errors = validate_critical_settings(settings)
    if config_errors:
        raise RuntimeError("Invalid critical settings: " + "; ".join(config_errors))

    try:
        ensure_runtime_support_tables()
    except Exception as e:
        logger.warning(f"[startup] Failed to ensure roadmap tables: {e}")

    # Configure ThreadPoolExecutor for better CPU-bound task handling
    # Optimized for high-capacity environment (8+ CPU cores, 64GB RAM)
    cpu_count = os.cpu_count() or 4
    executor_workers = max(4, min(16, cpu_count * 2))
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=executor_workers)
    asyncio.get_running_loop().set_default_executor(executor)
    logger.info(f"[startup] ThreadPoolExecutor configured with {executor_workers} workers")
    start_reload_listener()
    start_background_services()

    # Start periodic cleanup task for rate limiting
    asyncio.create_task(rate_limit_cleanup())

    async def _bg():
        """Executa bg."""
        try:
            logger.info("[warmup] Iniciando serviços...")
            r = get_redis()
            if r is None: logger.warning("[warmup] Redis indisponível.")
            
            try:
                _ensure_model_metrics_table()
            except Exception as e:
                logger.warning(f"[warmup] Failed to ensure model metrics table: {e}")

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
            # Smoke Tests (Opcional - Executar via pytest ou ENABLE_SMOKE_TESTS=1)
            # Run: pytest tests/test_smoke.py -v
            # =================================================================
            enable_smoke_tests = os.getenv("ENABLE_SMOKE_TESTS", "0").strip() in ("1", "true", "True")
            if enable_smoke_tests:
                logger.info("[warmup] ENABLE_SMOKE_TESTS=1, executando smoke tests...")
                try:
                    # Single warmup query instead of full suite
                    start_t = time.time()
                    await route_and_answer(
                        query="Qual é a capital do Brasil?",
                        modality="text",
                        use_cache=False
                    )
                    elapsed = time.time() - start_t
                    logger.info(f"[warmup] Smoke test OK ({elapsed:.2f}s)")
                except Exception as e:
                    logger.warning(f"[warmup] Smoke test falhou: {e}")
            else:
                logger.info("[warmup] Smoke tests desabilitados. Use ENABLE_SMOKE_TESTS=1 ou 'pytest tests/test_smoke.py'")

            logger.info("[warmup] Sistema pronto e aquecido. ✅")
        except Exception as e:
            logger.exception(f"[warmup] Erro crítico: {e}")

    asyncio.create_task(_bg())


async def shutdown_event():
    """
    Graceful shutdown handler.
    Properly closes Redis, HTTP, and database connections.
    """
    logger.info("[shutdown] Iniciando shutdown gracioso...")

    # Close HTTP connection pool
    try:
        stop_background_services()
        stop_reload_listener()
        await close_http_client()
        logger.info("[shutdown] HTTP connection pool closed")
    except Exception as e:
        logger.warning(f"[shutdown] Erro ao fechar HTTP pool: {e}")

    # Close Redis connections
    try:
        close_redis()
        logger.info("[shutdown] Redis connections closed")
    except Exception as e:
        logger.warning(f"[shutdown] Erro ao fechar Redis: {e}")

    # Close database connections
    try:
        close_engine()
        logger.info("[shutdown] Database connections closed")
    except Exception as e:
        logger.warning(f"[shutdown] Erro ao fechar database: {e}")

    logger.info("[shutdown] Shutdown completo.")


@app.post("/query", response_model=QueryResponse)
async def route_query(req: QueryRequest):
    """Executa route query."""
    start = time.time()
    API_REQUESTS.inc()

    processed = await process_query_request(req)
    result = processed["result"]
    image_input = processed["image_input"]

    duration = time.time() - start
    API_LATENCY.observe(duration)

    chosen_model = result["model"]
    ROUTER_CHOSEN.labels(model=chosen_model).inc()
    CANDIDATE_COST.observe(result["cost_per_1k"])
    CANDIDATE_LAT.observe(result["latency_s"])

    # Track detailed cost metrics
    cost_usd = result.get("cost_per_1k", 0)
    TOTAL_COST_USD.inc(cost_usd)

    # Extract provider from model name
    provider = chosen_model.split("/")[0] if "/" in chosen_model else "unknown"
    COST_BY_PROVIDER.labels(provider=provider).inc(cost_usd)

    # Track token usage
    metadata = result.get("metadata", {})
    prompt_tokens = metadata.get("prompt_tokens", 0)
    completion_tokens = metadata.get("completion_tokens", 0)
    TOKENS_INPUT_TOTAL.labels(model=chosen_model).inc(prompt_tokens)
    TOKENS_OUTPUT_TOTAL.labels(model=chosen_model).inc(completion_tokens)

    metadata = result.get("metadata", {})
    raw_payload_str = metadata.get("raw_payload")
    prompt_tokens = metadata.get("prompt_tokens", 0)
    completion_tokens = metadata.get("completion_tokens", 0)

    record_query_side_effects(req, result, image_input)

    parsed_payload = safe_parse_json(raw_payload_str)
    route_raw = result.get("route", {})
    candidates_raw = result.get("candidates", [])

    return QueryResponse(
        answer=result["answer"],
        model=chosen_model,
        modality=result["modality"],
        image_output_b64=result.get("image_output_b64"),
        correlation_id=get_correlation_id(),
        route=RouteDecision(**route_raw),
        candidates=[CandidateResult(**c) for c in candidates_raw],
        payload=parsed_payload
    )


@app.post("/query/stream", tags=["Query"])
async def route_query_stream(req: QueryRequest):
    """SSE streaming wrapper for query processing."""
    processed = await process_query_request(req)
    result = processed["result"]
    answer = result.get("answer", "")
    payload = {
        "model": result.get("model"),
        "modality": result.get("modality"),
        "metadata": result.get("metadata", {}),
    }

    async def _event_gen():
        yield "event: meta\\ndata: " + json.dumps(payload, ensure_ascii=False) + "\\n\\n"
        for token in answer.split():
            yield "event: token\\ndata: " + json.dumps({"text": token + " "}, ensure_ascii=False) + "\\n\\n"
            await asyncio.sleep(0)
        yield "event: done\\ndata: " + json.dumps({"status": "completed"}, ensure_ascii=False) + "\\n\\n"

    return StreamingResponse(_event_gen(), media_type="text/event-stream")


# ==============================================================================
# API Version 1 Routes (for future versioning)
# ==============================================================================

@v1_router.post("/query", response_model=QueryResponse)
async def v1_route_query(req: QueryRequest):
    """V1 Query endpoint - same as /query."""
    return await route_query(req)


@v1_router.get("/health")
async def v1_health():
    """V1 Health endpoint."""
    from .api.ops_routes import health

    return await health()


# Include versioned router
app.include_router(v1_router)
