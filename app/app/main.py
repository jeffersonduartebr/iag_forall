# -*- coding: utf-8 -*-
# Objective: Application runtime code for main.
"""Expose the FastAPI application, lifecycle hooks, and top-level HTTP routes.

This module is the primary API entry point for the router runtime. It is
responsible for:

- constructing the FastAPI app instance
- wiring middleware, routers, and observability exports
- orchestrating startup and shutdown tasks
- delegating query execution to the extracted service layer

Business logic intentionally stays thin here. Request-time orchestration is
delegated to service modules so this file remains focused on HTTP composition,
runtime bootstrap, and cross-cutting concerns.
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
import resource
import concurrent.futures
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Response, Request, APIRouter
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse
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
from .schemas import (
    CandidateResult,
    ConfidenceBand,
    EvidenceSnippet,
    QueryJobStatusResponse,
    QueryRequest,
    QueryResponse,
    QueuedQueryAcceptedResponse,
    ResponseDiagnostics,
    ResponseCitation,
    ResponseProvenance,
    ReviewStatus,
    RouteDecision,
    VerificationStatus,
)
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
    ROUTER_ENQUEUED_DUE_TO_DEADLINE,
    ROUTER_ENQUEUED_DUE_TO_QUEUE_WAIT,
)
from .router_core import start_background_services, stop_background_services
from .providers_async import (
    close_http_client,
    fetch_ollama_tags,
    get_http_client,
    get_ollama_admission_snapshot,
    log_process_file_descriptor_limit,
    start_provider_runtime_services,
    stop_provider_runtime_services,
    get_configured_ollama_warm_models,
    warm_ollama_model_runtime,
)
from .utils.redis_client import get_redis, close_redis
from .db import close_engine
from .query_jobs import enqueue_query_job, get_query_job_result, get_query_job_status
from .routers import rag_router
from .vectorstore import init_vectorstore, add_document as vs_add_document
from .services.query_response_builder import build_query_response
from .services.query_runtime import apply_query_runtime_profile, process_query_request, record_query_side_effects
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
    """Run application startup and shutdown work around the ASGI lifespan.

    FastAPI calls this context manager once per process. The implementation
    delegates to the explicit startup and shutdown helpers so the same behavior
    remains testable outside the ASGI server.
    """
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


def _normalize_request_modality(req: QueryRequest) -> tuple[str, str | None]:
    """Return the effective modality and first image payload for one request."""
    image_input = req.image_b64 or ((req.images or [None])[0] if getattr(req, "images", None) else None)
    modality = str(getattr(req, "modality", "text") or "text").lower()
    if image_input and modality == "text":
        modality = "vision"
    return modality, image_input


def _should_proactively_defer_query(req: QueryRequest, request: Request | None) -> tuple[bool, str, str, str]:
    """Decide whether the handler should enqueue early before sync execution starts.

    The middleware only sees coarse pressure snapshots. Once the request body is
    available here, the runtime can classify the workload and aggressively defer
    high-priority simple requests before they wait long enough to hit provider
    timeouts.
    """
    if request is None:
        return False, "overloaded", "normal", "unknown"
    if getattr(request.state, "defer_to_query_job", False):
        return True, str(getattr(request.state, "query_job_reason", "overloaded")), str(
            getattr(request.state, "query_job_pressure_state", "elevated")
        ), str(getattr(request.state, "query_job_workload_class", "unknown"))

    modality, image_input = _normalize_request_modality(req)
    runtime_profile = apply_query_runtime_profile(req, modality, image_input)
    workload_class = str(runtime_profile.get("workload_class", "unknown") or "unknown")
    runtime_hints = runtime_profile.get("runtime_hints") or {}

    if str(runtime_hints.get("interactive_priority", "normal")) != "high":
        return False, "overloaded", "normal", workload_class

    snapshot = get_ollama_admission_snapshot()
    pressure_state = str(snapshot.get("pressure_state", "normal") or "normal")
    if pressure_state == "normal":
        return False, "overloaded", pressure_state, workload_class

    current_limit = max(1, int(snapshot.get("current_limit", 1) or 1))
    total_inflight = int(snapshot.get("total_inflight", 0) or 0)
    queue_wait_ms = float(snapshot.get("max_queue_wait_ms", 0.0) or 0.0)
    sync_queue_wait_ms = max(50.0, float(settings.get("ADAPTIVE_LIMITER_SYNC_QUEUE_WAIT_MS", 250.0)))
    near_capacity = total_inflight >= max(1, current_limit - 1)
    queue_pressure = queue_wait_ms >= max(100.0, sync_queue_wait_ms * 0.5)
    should_defer = pressure_state == "congested" or near_capacity or queue_pressure
    if not should_defer:
        return False, "overloaded", pressure_state, workload_class

    reason = "ollama_queue_wait" if queue_pressure or near_capacity else "sync_deadline_protection"
    try:
        if reason == "ollama_queue_wait":
            ROUTER_ENQUEUED_DUE_TO_QUEUE_WAIT.labels(pressure_state=pressure_state).inc()
        ROUTER_ENQUEUED_DUE_TO_DEADLINE.labels(workload_class=workload_class).inc()
    except Exception:
        pass
    request.state.defer_to_query_job = True
    request.state.query_job_reason = reason
    request.state.query_job_pressure_state = pressure_state
    request.state.query_job_workload_class = workload_class
    return True, reason, pressure_state, workload_class

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
        """Propagate a correlation ID across the current request/response cycle.

        The middleware reads the incoming header when present, generates one
        otherwise, and ensures the final response echoes the active correlation
        ID so logs and client-side traces can be stitched together.
        """
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


OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
)
VLM_OLLAMA_MODELS = list(getattr(settings, "VLM_OLLAMA_MODELS", []))


async def preload_ollama_models():
    """
    Preload Ollama models asynchronously using the shared provider HTTP client.

    Model discovery and downloads are routed through the provider runtime so
    startup warmup does not create extra ad hoc HTTP clients or duplicate
    `/api/tags` polling pressure.
    """
    try:
        logger.info("[ollama-preload] Iniciando verificação...")
        candidates_raw = os.getenv("CANDIDATE_MODELS_LIST", "[]")
        judges_raw = os.getenv("JUDGE_MODELS", "[]")
        main_model = os.getenv("OLLAMA_MODEL", "")
        embed_model = settings.get("EMBED_TEXT_MODEL", "all-minilm")

        configured_warm_models = get_configured_ollama_warm_models()
        all_models = list(configured_warm_models)
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
        all_models = list(dict.fromkeys(all_models))

        if not all_models:
            return

        try:
            available = {m["name"] for m in await fetch_ollama_tags(force_refresh=True)}
        except Exception:
            available = set()

        for model in all_models:
            name = model.split("/", 1)[1]
            if any(name in avail for avail in available) or f"{name}:latest" in available:
                logger.info(f"[ollama-preload] '{name}' já disponível.")
                continue

            logger.info(f"[ollama-preload] Baixando '{name}'...")
            try:
                client = await get_http_client()
                async with client.stream(
                    "POST",
                    f"{OLLAMA_HOST}/api/pull",
                    json={"name": name},
                    timeout=1200.0,
                ) as response:
                    response.raise_for_status()
                    # Consume the stream to complete the download
                    async for _ in response.aiter_lines():
                        pass
                logger.info(f"[ollama-preload] '{name}' OK.")
            except Exception as e:
                logger.error(f"[ollama-preload] Falha ao baixar '{name}': {e}")

        if str(settings.get("OLLAMA_WARMUP_GENERATE_ENABLED", "1")).strip() == "1":
            warm_targets = configured_warm_models or all_models[: max(1, min(3, len(all_models)))]
            for model in warm_targets:
                try:
                    await warm_ollama_model_runtime(model)
                except Exception as e:
                    logger.warning(f"[ollama-preload] Falha ao aquecer runtime '{model}': {e}")

        logger.info("[ollama-preload] Concluído.")
    except Exception as e:
        logger.exception(f"[ollama-preload] Erro geral: {e}")


app.include_router(rag_router.router)


@app.get("/metrics")
def metrics():
    """Return the Prometheus exposition payload for the current process.

    The route keeps the HTTP-facing layer intentionally thin. Metric rendering
    is delegated to the observability module so this endpoint only translates
    the serialized payload into a regular FastAPI `Response`.
    """
    data, ctype = render_metrics_response()
    return Response(content=data, media_type=ctype)


async def startup_event():
    """Initialize runtime dependencies and schedule non-blocking warmup tasks.

    Startup validates critical configuration, prepares shared services, installs
    background listeners, and schedules warmup work that should not block API
    availability longer than necessary. Fail-fast validation remains here
    because serving traffic with invalid admin or runtime configuration would be
    operationally unsafe.
    """
    admin_token = (settings.ADMIN_TOKEN or "").strip()
    if not admin_token or admin_token == "changeme-please":
        raise RuntimeError("ADMIN_TOKEN must be configured with a non-default value")

    config_errors = validate_critical_settings(settings)
    if config_errors:
        raise RuntimeError("Invalid critical settings: " + "; ".join(config_errors))

    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        logger.info("[startup] RLIMIT_NOFILE soft=%s hard=%s", soft, hard)
    except Exception as e:
        logger.warning(f"[startup] Failed to read RLIMIT_NOFILE: {e}")

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
    log_process_file_descriptor_limit("startup")
    await start_provider_runtime_services()

    # Start periodic cleanup task for rate limiting
    asyncio.create_task(rate_limit_cleanup())

    async def _bg():
        """Perform deferred warmup work after the HTTP server is accepting traffic.

        The background task primes optional dependencies such as Redis,
        vectorstore state, Ollama model availability, and a lightweight
        end-to-end smoke path. Failures are logged as degraded startup rather
        than crashing the process because most of this work is opportunistic.
        """
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
    """Release long-lived runtime resources during process shutdown.

    Shutdown mirrors startup ordering in reverse: background services are
    stopped first, then shared HTTP clients, Redis connections, and database
    pools are closed. Each teardown step is isolated so one failure does not
    prevent the remaining cleanup actions from running.
    """
    logger.info("[shutdown] Iniciando shutdown gracioso...")

    # Close HTTP connection pool
    try:
        stop_background_services()
        stop_reload_listener()
        await stop_provider_runtime_services()
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


async def route_query(req: QueryRequest, request: Request | None = None):
    """Handle the primary synchronous query endpoint.

    The route delegates most orchestration to `process_query_request`, then
    records request-level metrics, updates cost and token counters, triggers
    side effects such as persistence and feedback scheduling, and finally maps
    the service result into the public `QueryResponse` schema.
    """
    start = time.time()
    API_REQUESTS.inc()

    should_defer, _, _, _ = _should_proactively_defer_query(req, request)
    if should_defer and request is not None and getattr(request.state, "defer_to_query_job", False):
        queued = enqueue_query_job(
            req=req,
            correlation_id=get_correlation_id(),
            reason=str(getattr(request.state, "query_job_reason", "overloaded")),
            pressure_state=str(getattr(request.state, "query_job_pressure_state", "elevated")),
            route_path=request.url.path,
            identity_key=request.client.host if getattr(request, "client", None) else None,
        )
        return JSONResponse(status_code=202, content=jsonable_encoder(queued))

    processed = await process_query_request(req)
    result = processed["result"]
    image_input = processed["image_input"]

    duration = time.time() - start
    API_LATENCY.observe(duration)

    chosen_model = result["model"]
    estimated_cost_usd = result.get("estimated_cost_usd", result.get("cost_per_1k", 0.0))
    ROUTER_CHOSEN.labels(model=chosen_model).inc()
    CANDIDATE_COST.observe(estimated_cost_usd)
    CANDIDATE_LAT.observe(result["latency_s"])

    # Track detailed cost metrics
    cost_usd = estimated_cost_usd
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

    correlation_id = get_correlation_id()
    metadata = result.get("metadata", {})
    metadata["correlation_id"] = correlation_id

    record_query_side_effects(req, result, image_input)
    return build_query_response(result, correlation_id)


async def route_query_stream(req: QueryRequest, request: Request | None = None):
    """Expose a simple Server-Sent Events wrapper around the standard query flow.

    The current implementation does not stream provider tokens directly. It
    first resolves the full answer through the same runtime service used by the
    synchronous endpoint, then emits metadata, whitespace-delimited token
    chunks, and a terminal completion event to consumers expecting an SSE
    contract.
    """
    should_defer, _, _, _ = _should_proactively_defer_query(req, request)
    if should_defer and request is not None and getattr(request.state, "defer_to_query_job", False):
        queued = enqueue_query_job(
            req=req,
            correlation_id=get_correlation_id(),
            reason=str(getattr(request.state, "query_job_reason", "overloaded")),
            pressure_state=str(getattr(request.state, "query_job_pressure_state", "elevated")),
            route_path=request.url.path,
            identity_key=request.client.host if getattr(request, "client", None) else None,
        )
        return JSONResponse(status_code=202, content=jsonable_encoder(queued))

    processed = await process_query_request(req)
    result = processed["result"]
    answer = result.get("answer", "")
    payload = {
        "model": result.get("model"),
        "modality": result.get("modality"),
        "metadata": result.get("metadata", {}),
    }

    async def _event_gen():
        """Yield the SSE event sequence for the already-computed answer payload."""
        yield "event: meta\\ndata: " + json.dumps(payload, ensure_ascii=False) + "\\n\\n"
        for token in answer.split():
            yield "event: token\\ndata: " + json.dumps({"text": token + " "}, ensure_ascii=False) + "\\n\\n"
            await asyncio.sleep(0)
        yield "event: done\\ndata: " + json.dumps({"status": "completed"}, ensure_ascii=False) + "\\n\\n"

    return StreamingResponse(_event_gen(), media_type="text/event-stream")


@app.post("/query", response_model=QueryResponse | QueuedQueryAcceptedResponse)
async def http_route_query(req: QueryRequest, request: Request):
    """Bind the primary query helper to the public HTTP route."""
    return await route_query(req, request)


@app.post("/query/stream", tags=["Query"])
async def http_route_query_stream(req: QueryRequest, request: Request):
    """Bind the streaming query helper to the public HTTP route."""
    return await route_query_stream(req, request)


# ==============================================================================
# API Version 1 Routes (for future versioning)
# ==============================================================================

@app.get("/query/jobs/{job_id}", response_model=QueryJobStatusResponse, tags=["Query"])
async def query_job_status(job_id: str):
    """Return the current status of one queued query job."""
    return get_query_job_status(job_id)


@app.get("/query/jobs/{job_id}/result", response_model=QueryResponse, tags=["Query"])
async def query_job_result(job_id: str):
    """Return the completed result of one queued query job."""
    return get_query_job_result(job_id)


async def v1_route_query(req: QueryRequest, request: Request | None = None):
    """Provide the versioned alias for the primary query endpoint.

    Version one currently preserves the exact behavior of `/query`; the wrapper
    exists so clients can pin to an explicit API version before future route
    evolution introduces incompatible changes.
    """
    return await route_query(req, request)


@v1_router.post("/query", response_model=QueryResponse | QueuedQueryAcceptedResponse)
async def v1_http_route_query(req: QueryRequest, request: Request):
    """Bind the versioned query helper to the public HTTP route."""
    return await v1_route_query(req, request)


@v1_router.get("/health")
async def v1_health():
    """Provide the versioned alias for the operational health endpoint."""
    from .api.ops_routes import health

    return await health()


@v1_router.get("/query/jobs/{job_id}", response_model=QueryJobStatusResponse)
async def v1_query_job_status(job_id: str):
    """Provide the versioned alias for queued-query status lookups."""
    return get_query_job_status(job_id)


@v1_router.get("/query/jobs/{job_id}/result", response_model=QueryResponse)
async def v1_query_job_result(job_id: str):
    """Provide the versioned alias for queued-query result lookups."""
    return get_query_job_result(job_id)


# Include versioned router
app.include_router(v1_router)
