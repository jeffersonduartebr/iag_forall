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
import gzip
from typing import List, Dict, Any, Optional
from io import BytesIO

import requests
from fastapi import FastAPI, HTTPException, Header, Body, Response, Request, APIRouter
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.gzip import GZipMiddleware

from .prometheus_setup import setup_prometheus
from .correlation import (
    set_correlation_id,
    get_correlation_id,
    clear_correlation_id,
    CORRELATION_ID_HEADER,
)
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
    RATE_LIMIT_EXCEEDED,
    TOTAL_COST_USD,
    COST_BY_PROVIDER,
    TOKENS_INPUT_TOTAL,
    TOKENS_OUTPUT_TOTAL,
    COMPONENT_HEALTH,
)
from .router_core import route_and_answer
from .utils.redis_client import get_redis
from .routers import rag_router
from .vectorstore import init_vectorstore, add_document as vs_add_document
from .tasks import task_process_feedback
from .health import (
    get_full_health_check,
    get_liveness_check,
    get_readiness_check,
)
from .reliability import get_circuit_breaker_manager, get_cascade_detector
from .error_handling import log_error, create_error_response, ErrorCategory
from .user_feedback import UserFeedbackRequest, process_feedback, get_feedback_stats
from .ab_testing import (
    ABTestManager,
    get_ab_test_manager,
    ExperimentCreateRequest,
    ExperimentStatus,
)

# Configuração de Logging
setup_logging()
# Força silêncio no logger do Chroma
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("posthog").setLevel(logging.CRITICAL)

setup_prometheus()

# ==============================================================================
# Rate Limiting Configuration
# ==============================================================================
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))  # requests per window
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds


class RateLimitStore:
    """Simple in-memory rate limiting store. For production, use Redis."""

    def __init__(self):
        self._store: Dict[str, List[float]] = {}
        self._lock = asyncio.Lock()

    async def is_rate_limited(self, client_ip: str, max_requests: int, window_seconds: int) -> bool:
        """Check if client is rate limited."""
        now = time.time()
        cutoff = now - window_seconds

        async with self._lock:
            # Get or create request history
            if client_ip not in self._store:
                self._store[client_ip] = []

            # Clean old entries
            self._store[client_ip] = [t for t in self._store[client_ip] if t > cutoff]

            # Check if over limit
            if len(self._store[client_ip]) >= max_requests:
                return True

            # Add current request
            self._store[client_ip].append(now)
            return False

    async def cleanup(self):
        """Remove old entries to prevent memory bloat."""
        now = time.time()
        cutoff = now - 3600  # Clean entries older than 1 hour

        async with self._lock:
            for ip in list(self._store.keys()):
                self._store[ip] = [t for t in self._store[ip] if t > cutoff]
                if not self._store[ip]:
                    del self._store[ip]


rate_limit_store = RateLimitStore()


# ==============================================================================
# FastAPI App with Versioning
# ==============================================================================
app = FastAPI(
    title="LLM/VLM Router (Hybrid Bandit + RAG + Celery Feedback)",
    version="3.2.0",
    description="API de Roteamento Inteligente Multimodal com Persistência Robusta.",
    docs_url="/docs",
    redoc_url="/redoc",
)

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
        correlation_id = request.headers.get(CORRELATION_ID_HEADER)
        set_correlation_id(correlation_id)

        try:
            response = await call_next(request)
            # Add correlation ID to response headers
            response.headers[CORRELATION_ID_HEADER] = get_correlation_id() or ""
            return response
        finally:
            clear_correlation_id()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using sliding window algorithm.
    """

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks and metrics
        if request.url.path in ["/health", "/healthz", "/ready", "/metrics", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)

        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()

        # Check rate limit
        if await rate_limit_store.is_rate_limited(client_ip, RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW):
            RATE_LIMIT_EXCEEDED.labels(client_ip=client_ip).inc()
            logger.warning(
                "rate_limit_exceeded",
                client_ip=client_ip,
                limit=RATE_LIMIT_REQUESTS,
                window=RATE_LIMIT_WINDOW,
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": True,
                    "message": "Rate limit exceeded. Please try again later.",
                    "retry_after": RATE_LIMIT_WINDOW,
                },
                headers={
                    "Retry-After": str(RATE_LIMIT_WINDOW),
                    "X-RateLimit-Limit": str(RATE_LIMIT_REQUESTS),
                    "X-RateLimit-Window": str(RATE_LIMIT_WINDOW),
                },
            )

        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(RATE_LIMIT_REQUESTS)
        response.headers["X-RateLimit-Window"] = str(RATE_LIMIT_WINDOW)

        return response


# Add middleware in order (last added = first executed)
app.add_middleware(CorrelationIdMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(GZipMiddleware, minimum_size=1000)  # Compress responses > 1KB


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


async def periodic_cleanup():
    """Periodically clean up rate limit store to prevent memory bloat."""
    while True:
        await asyncio.sleep(300)  # Every 5 minutes
        try:
            await rate_limit_store.cleanup()
            logger.debug("[cleanup] Rate limit store cleaned")
        except Exception as e:
            logger.warning(f"[cleanup] Rate limit cleanup failed: {e}")


@app.on_event("startup")
async def startup_event():
    # Start periodic cleanup task
    asyncio.create_task(periodic_cleanup())

    async def _bg():
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
            use_cache=req.use_cache,
            timeout_seconds=req.timeout_seconds,
        )
    except asyncio.TimeoutError:
        error_info = log_error(
            asyncio.TimeoutError("Request timed out"),
            category=ErrorCategory.PROVIDER_TIMEOUT,
        )
        raise HTTPException(
            status_code=504,
            detail=create_error_response(error_info),
        )
    except Exception as e:
        error_info = log_error(e)
        logger.exception(f"[router] Erro: {e}")
        raise HTTPException(
            status_code=500,
            detail=create_error_response(error_info),
        )

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
    uncertainty_score = metadata.get("uncertainty_score", 0.5)

    # Combine raw_payload with uncertainty_score for query_log persistence
    combined_payload = {
        "raw_payload": raw_payload_str,
        "uncertainty_score": uncertainty_score,
    }

    try:
        task_process_feedback.delay(
            query=req.query,
            answer=result["answer"],
            chosen_model=chosen_model,
            modality=result["modality"],
            latency_s=result["latency_s"],
            cost_val=result["cost_per_1k"],
            image_b64=image_input,
            raw_payload=combined_payload,
            prompt_tokens=metadata.get("prompt_tokens", 0),
            completion_tokens=metadata.get("completion_tokens", 0)
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
        correlation_id=get_correlation_id(),
        route=RouteDecision(**route_raw),
        candidates=[CandidateResult(**c) for c in candidates_raw],
        payload=parsed_payload
    )


def _require_admin(token: Optional[str]):
    if token != settings.ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Token inválido.")


# ==============================================================================
# Admin Endpoints
# ==============================================================================

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


@app.get("/admin/circuit-breakers", tags=["Admin"])
def get_circuit_breakers(x_admin_token: Optional[str] = Header(None)):
    """Get status of all circuit breakers."""
    _require_admin(x_admin_token)
    manager = get_circuit_breaker_manager()
    return {
        "circuit_breakers": manager.get_all_statuses(),
        "timestamp": time.time(),
    }


@app.post("/admin/circuit-breakers/{model_name}/reset", tags=["Admin"])
def reset_circuit_breaker(model_name: str, x_admin_token: Optional[str] = Header(None)):
    """Reset a specific circuit breaker."""
    _require_admin(x_admin_token)
    manager = get_circuit_breaker_manager()
    success = manager.reset_breaker(model_name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Circuit breaker for '{model_name}' not found")
    return {"status": "reset", "model": model_name}


@app.get("/admin/cascade-status", tags=["Admin"])
def get_cascade_status(x_admin_token: Optional[str] = Header(None)):
    """Get cascade failure detection status."""
    _require_admin(x_admin_token)
    detector = get_cascade_detector()
    return detector.get_status()


# ==============================================================================
# User Feedback Endpoint (Phase 3.2)
# ==============================================================================

@app.post("/feedback", tags=["Feedback"])
def submit_feedback(request: UserFeedbackRequest):
    """
    Submit user feedback for a model response.

    Supports:
    - thumbs_up / thumbs_down
    - rating (1-5 stars)
    - explicit_quality (0-10 score)
    """
    try:
        result = process_feedback(request)
        return {
            "status": "accepted",
            "user_quality": result.user_quality,
            "blended_quality": result.blended_quality,
            "model": result.model,
            "reward": round(result.reward, 4),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"[feedback] Error processing feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to process feedback")


@app.get("/feedback/stats", tags=["Feedback"])
def feedback_stats(model: Optional[str] = None, hours: int = 24):
    """Get feedback statistics."""
    return get_feedback_stats(model=model, hours=hours)


# ==============================================================================
# A/B Testing Endpoints (Phase 4)
# ==============================================================================

@app.get("/admin/experiments", tags=["A/B Testing"])
def list_experiments(
    status: Optional[str] = None,
    x_admin_token: Optional[str] = Header(None),
):
    """List all A/B experiments."""
    _require_admin(x_admin_token)

    if not settings.AB_TESTING_ENABLED:
        return {"error": "A/B testing is disabled", "experiments": []}

    manager = get_ab_test_manager()
    status_filter = ExperimentStatus(status) if status else None
    experiments = manager.list_experiments(status=status_filter)

    return {
        "experiments": [exp.to_dict() for exp in experiments],
        "total": len(experiments),
    }


@app.post("/admin/experiments", tags=["A/B Testing"])
def create_experiment(
    request: ExperimentCreateRequest,
    x_admin_token: Optional[str] = Header(None),
):
    """Create a new A/B experiment."""
    _require_admin(x_admin_token)

    if not settings.AB_TESTING_ENABLED:
        raise HTTPException(status_code=400, detail="A/B testing is disabled")

    try:
        manager = get_ab_test_manager()
        experiment = manager.create_experiment(request)
        return {"status": "created", "experiment": experiment.to_dict()}
    except Exception as e:
        logger.exception(f"[experiments] Error creating experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/experiments/{experiment_id}", tags=["A/B Testing"])
def get_experiment(
    experiment_id: str,
    x_admin_token: Optional[str] = Header(None),
):
    """Get a specific experiment."""
    _require_admin(x_admin_token)

    manager = get_ab_test_manager()
    experiment = manager.get_experiment(experiment_id)

    if not experiment:
        raise HTTPException(status_code=404, detail=f"Experiment not found: {experiment_id}")

    return experiment.to_dict()


@app.post("/admin/experiments/{experiment_id}/start", tags=["A/B Testing"])
def start_experiment(
    experiment_id: str,
    x_admin_token: Optional[str] = Header(None),
):
    """Start an experiment."""
    _require_admin(x_admin_token)

    try:
        manager = get_ab_test_manager()
        experiment = manager.start_experiment(experiment_id)
        return {"status": "started", "experiment": experiment.to_dict()}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Experiment not found: {experiment_id}")


@app.post("/admin/experiments/{experiment_id}/pause", tags=["A/B Testing"])
def pause_experiment(
    experiment_id: str,
    x_admin_token: Optional[str] = Header(None),
):
    """Pause an experiment."""
    _require_admin(x_admin_token)

    try:
        manager = get_ab_test_manager()
        experiment = manager.pause_experiment(experiment_id)
        return {"status": "paused", "experiment": experiment.to_dict()}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Experiment not found: {experiment_id}")


@app.post("/admin/experiments/{experiment_id}/complete", tags=["A/B Testing"])
def complete_experiment(
    experiment_id: str,
    x_admin_token: Optional[str] = Header(None),
):
    """Complete an experiment."""
    _require_admin(x_admin_token)

    try:
        manager = get_ab_test_manager()
        experiment = manager.complete_experiment(experiment_id)
        return {"status": "completed", "experiment": experiment.to_dict()}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Experiment not found: {experiment_id}")


@app.get("/admin/experiments/{experiment_id}/results", tags=["A/B Testing"])
def get_experiment_results(
    experiment_id: str,
    x_admin_token: Optional[str] = Header(None),
):
    """Get aggregated results for an experiment."""
    _require_admin(x_admin_token)

    manager = get_ab_test_manager()
    return manager.get_experiment_results(experiment_id)


@app.delete("/admin/experiments/{experiment_id}", tags=["A/B Testing"])
def delete_experiment(
    experiment_id: str,
    x_admin_token: Optional[str] = Header(None),
):
    """Delete an experiment."""
    _require_admin(x_admin_token)

    manager = get_ab_test_manager()
    success = manager.delete_experiment(experiment_id)

    if not success:
        raise HTTPException(status_code=404, detail=f"Experiment not found: {experiment_id}")

    return {"status": "deleted", "experiment_id": experiment_id}


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
    return await health()


# Include versioned router
app.include_router(v1_router)

# ==============================================================================
# Health Check Endpoints
# ==============================================================================

@app.get("/health", tags=["Ops"])
async def health():
    """Deep health check with all component statuses."""
    result = await get_full_health_check()

    # Update Prometheus metrics for component health
    for name, component in result.get("components", {}).items():
        COMPONENT_HEALTH.labels(component=name).set(1 if component.get("healthy") else 0)

    # Return appropriate status code
    status_code = 200 if result["status"] == "healthy" else (
        503 if result["status"] == "unhealthy" else 200
    )
    return JSONResponse(content=result, status_code=status_code)


@app.get("/healthz", tags=["Ops"])
async def liveness():
    """Kubernetes liveness probe - is the app running?"""
    return await get_liveness_check()


@app.get("/ready", tags=["Ops"])
async def readiness():
    """Kubernetes readiness probe - is the app ready to serve traffic?"""
    result = await get_readiness_check()
    status_code = 200 if result["status"] == "ready" else 503
    return JSONResponse(content=result, status_code=status_code)
