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
import concurrent.futures
import secrets
import uuid
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
from io import BytesIO

import httpx
from fastapi import FastAPI, HTTPException, Header, Body, Response, Request, APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.gzip import GZipMiddleware
from celery.result import AsyncResult

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
    rate_limit_store,
    periodic_cleanup as rate_limit_cleanup,
)
from .middleware.backpressure import BackpressureMiddleware
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
    TOTAL_COST_USD,
    COST_BY_PROVIDER,
    TOKENS_INPUT_TOTAL,
    TOKENS_OUTPUT_TOTAL,
    COMPONENT_HEALTH,
)
from .router_core import route_and_answer, start_background_services, stop_background_services
from .providers_async import close_http_client, ProviderCallError, ProviderCircuitOpenError
from .utils.redis_client import get_redis, close_redis
from .db import close_engine
from .routers import rag_router
from .vectorstore import init_vectorstore, add_document as vs_add_document
from .tasks import task_execute_eval_run, task_process_feedback
from .celery_app import celery_app
from .health import (
    get_full_health_check,
    get_liveness_check,
    get_readiness_check,
)
from .runtime_state import reset_runtime_state
from .reliability import get_circuit_breaker_manager, get_cascade_detector
from .error_handling import log_error, create_error_response, ErrorCategory
from .user_feedback import UserFeedbackRequest, process_feedback, get_feedback_stats
from .ab_testing import (
    ABTestManager,
    get_ab_test_manager,
    ExperimentCreateRequest,
    ExperimentStatus,
)
from .guardrails import check_input_guardrails, sanitize_output_guardrails
from .roadmap_features import (
    ensure_roadmap_tables,
    check_tenant_budget,
    record_tenant_usage,
    set_tenant_budget,
    get_tenant_budget,
    get_usage_summary,
    log_audit_event,
    list_audit_events,
    create_policy_version,
    activate_policy_version,
    get_active_policy,
    list_policy_versions,
    create_eval_run,
    update_eval_run_status,
    get_eval_run,
    list_eval_run_results,
    list_eval_runs,
    eval_significance_report,
    grant_role,
    revoke_role,
    list_roles,
    check_access,
)

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
        ensure_roadmap_tables()
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


async def _process_query_request(req: QueryRequest) -> Dict[str, Any]:
    """Process one query request with governance, guardrails and experimentation hooks."""
    if not req or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query obrigatória.")

    input_decision = check_input_guardrails(req.query)
    if not input_decision.allowed:
        raise HTTPException(
            status_code=400,
            detail={
                "error": True,
                "category": "guardrail_block",
                "message": "Conteúdo bloqueado por política de segurança.",
                "reasons": input_decision.reasons,
            },
        )

    pre_budget = check_tenant_budget(req.tenant_id)
    if not pre_budget.allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "error": True,
                "category": "tenant_budget_exceeded",
                "reason": pre_budget.reason,
                "daily_spent": pre_budget.daily_spent,
                "monthly_spent": pre_budget.monthly_spent,
                "daily_limit": pre_budget.daily_limit,
                "monthly_limit": pre_budget.monthly_limit,
            },
        )

    modality = (req.modality or "text").lower()
    image_input = req.image_b64
    if not image_input and req.images and len(req.images) > 0:
        image_input = req.images[0]
    if image_input and modality == "text":
        modality = "vision"

    selected_policy = req.policy_version
    active_policy = get_active_policy()
    if not selected_policy and active_policy:
        selected_policy = active_policy.get("version")

    assigned_variant = None
    if req.experiment_id and settings.AB_TESTING_ENABLED:
        try:
            manager = get_ab_test_manager()
            assignment = manager.get_assignment(
                req.experiment_id,
                req.user_key or req.tenant_id or f"anon:{hash(req.query)}",
            )
            if assignment:
                variant_name, variant_cfg = assignment
                assigned_variant = {"name": variant_name, "config": variant_cfg}
                # Variant may override policy version.
                selected_policy = variant_cfg.get("policy_version", selected_policy)
        except Exception as e:
            logger.warning(f"[query] Failed experiment assignment: {e}")

    logger.info(
        "[query] '%s...' (mod=%s, tenant=%s, policy=%s, exp=%s)",
        req.query[:60],
        modality,
        req.tenant_id or "-",
        selected_policy or "-",
        req.experiment_id or "-",
    )

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
        raise HTTPException(status_code=504, detail=create_error_response(error_info))
    except ProviderCircuitOpenError as e:
        error_info = log_error(e, category=ErrorCategory.CIRCUIT_OPEN, model=e.model)
        raise HTTPException(status_code=503, detail=create_error_response(error_info))
    except ProviderCallError as e:
        category_map = {
            "provider_timeout": ErrorCategory.PROVIDER_TIMEOUT,
            "provider_rate_limit": ErrorCategory.PROVIDER_RATE_LIMIT,
            "provider_unavailable": ErrorCategory.PROVIDER_UNAVAILABLE,
        }
        category = category_map.get(e.category, ErrorCategory.PROVIDER_UNAVAILABLE)
        status_code = 504 if category == ErrorCategory.PROVIDER_TIMEOUT else (429 if category == ErrorCategory.PROVIDER_RATE_LIMIT else 502)
        error_info = log_error(e, category=category, model=e.model)
        raise HTTPException(status_code=status_code, detail=create_error_response(error_info))
    except Exception as e:
        error_info = log_error(e)
        logger.exception(f"[router] Erro: {e}")
        raise HTTPException(status_code=500, detail=create_error_response(error_info))

    answer_clean, output_guardrail_tags = sanitize_output_guardrails(result.get("answer", ""))
    result["answer"] = answer_clean
    metadata = result.setdefault("metadata", {})
    metadata["guardrail_output_tags"] = output_guardrail_tags
    metadata["policy_version"] = selected_policy
    metadata["experiment_id"] = req.experiment_id
    metadata["experiment_variant"] = assigned_variant
    metadata["tenant_id"] = req.tenant_id

    return {
        "result": result,
        "image_input": image_input,
        "selected_policy": selected_policy,
        "assigned_variant": assigned_variant,
        "modality": modality,
    }


@app.post("/query", response_model=QueryResponse)
async def route_query(req: QueryRequest):
    """Executa route query."""
    start = time.time()
    API_REQUESTS.inc()

    processed = await _process_query_request(req)
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

    try:
        record_tenant_usage(
            req.tenant_id,
            cost_usd=float(cost_usd),
            tokens_in=int(prompt_tokens or 0),
            tokens_out=int(completion_tokens or 0),
            requests=1,
        )
    except Exception as e:
        logger.warning(f"[query] Failed to record tenant usage: {e}")

    if req.experiment_id and settings.AB_TESTING_ENABLED:
        try:
            manager = get_ab_test_manager()
            variant = (metadata.get("experiment_variant") or {}).get("name")
            if variant:
                manager.record_result(req.experiment_id, variant, "quality", float(metadata.get("quality", 0.0) or 0.0))
                manager.record_result(req.experiment_id, variant, "latency", float(result.get("latency_s", 0.0)))
                manager.record_result(req.experiment_id, variant, "cost", float(cost_usd))
        except Exception as e:
            logger.warning(f"[query] Failed to record experiment metrics: {e}")

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
    processed = await _process_query_request(req)
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


def _require_admin(token: Optional[str]):
    """Executa require admin."""
    configured = (settings.ADMIN_TOKEN or "").strip()
    previous = (settings.ADMIN_TOKEN_PREVIOUS or "").strip()

    ok = False
    if configured and token:
        ok = secrets.compare_digest(token, configured)
        if not ok and previous:
            ok = secrets.compare_digest(token, previous)

    if not ok:
        raise HTTPException(status_code=401, detail="Token inválido.")


def _parse_header_roles(x_user_roles: Optional[str]) -> List[str]:
    """Parse comma-separated roles from header."""
    if not x_user_roles:
        return []
    return [r.strip() for r in str(x_user_roles).split(",") if r.strip()]


def _require_admin_or_role(
    *,
    admin_token: Optional[str],
    user_id: Optional[str],
    user_roles_header: Optional[str],
    required_roles: List[str],
    tenant_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Authorize request by admin token or RBAC role."""
    try:
        _require_admin(admin_token)
        return {"authorized_by": "admin_token", "roles": ["admin"]}
    except HTTPException:
        decision = check_access(
            user_id=user_id,
            tenant_id=tenant_id,
            required_roles=required_roles,
            header_roles=_parse_header_roles(user_roles_header),
        )
        if decision.allowed:
            return {"authorized_by": "rbac", "roles": decision.roles}
    raise HTTPException(status_code=403, detail={"error": True, "message": "Acesso negado.", "required_roles": required_roles})


# ==============================================================================
# Admin Endpoints
# ==============================================================================

@app.get("/admin/settings", tags=["Admin"])
def get_settings(x_admin_token: Optional[str] = Header(None)):
    """Obtém settings."""
    _require_admin(x_admin_token)
    return settings.snapshot()


@app.put("/admin/settings", tags=["Admin"])
def update_settings(payload: Dict[str, Any], x_admin_token: Optional[str] = Header(None)):
    """Executa update settings."""
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


@app.post("/admin/runtime/reset", tags=["Admin"])
def reset_runtime(x_admin_token: Optional[str] = Header(None)):
    """
    Reset internal runtime/singleton state for operational recovery.
    """
    _require_admin(x_admin_token)
    reset_runtime_state()
    return {"status": "reset"}


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
# Governance / Policy / Eval Endpoints (Roadmap MVP)
# ==============================================================================

@app.put("/admin/budgets/{tenant_id}", tags=["Governance"])
def upsert_tenant_budget(
    tenant_id: str,
    payload: Dict[str, Any],
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
):
    """Create or update tenant budget limits."""
    auth = _require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        required_roles=["governance_admin", "platform_admin"],
        tenant_id=tenant_id,
    )
    daily = float(payload.get("daily_usd_limit", 0.0) or 0.0)
    monthly = float(payload.get("monthly_usd_limit", 0.0) or 0.0)
    enabled = bool(payload.get("enabled", True))
    set_tenant_budget(tenant_id=tenant_id, daily_usd_limit=daily, monthly_usd_limit=monthly, enabled=enabled)
    log_audit_event(
        actor=x_user_id or auth["authorized_by"],
        action="budget_upsert",
        resource="tenant_budgets",
        tenant_id=tenant_id,
        metadata={"daily_usd_limit": daily, "monthly_usd_limit": monthly, "enabled": enabled, "roles": auth["roles"]},
    )
    return {"status": "updated", "budget": get_tenant_budget(tenant_id)}


@app.get("/admin/budgets/{tenant_id}", tags=["Governance"])
def get_budget(
    tenant_id: str,
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
):
    """Get tenant budget configuration."""
    _require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        required_roles=["governance_viewer", "governance_admin", "platform_admin"],
        tenant_id=tenant_id,
    )
    return get_tenant_budget(tenant_id)


@app.get("/admin/quotas/usage", tags=["Governance"])
def get_quota_usage(
    tenant_id: Optional[str] = None,
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
):
    """Get usage summary for one or all tenants."""
    _require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        required_roles=["governance_viewer", "governance_admin", "platform_admin"],
        tenant_id=tenant_id,
    )
    return get_usage_summary(tenant_id)


@app.get("/admin/audit/events", tags=["Governance"])
def get_audit_events(
    limit: int = 100,
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
):
    """Get latest audit events."""
    _require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        required_roles=["audit_viewer", "platform_admin"],
    )
    return {"items": list_audit_events(limit=limit)}


@app.post("/admin/policies", tags=["Policy"])
def create_policy(
    payload: Dict[str, Any],
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
):
    """Create or update a policy version."""
    auth = _require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        required_roles=["policy_admin", "platform_admin"],
    )
    version = str(payload.get("version") or "").strip()
    if not version:
        raise HTTPException(status_code=400, detail="version is required")
    description = str(payload.get("description") or "")
    config = payload.get("config") or {}
    if not isinstance(config, dict):
        raise HTTPException(status_code=400, detail="config must be an object")
    create_policy_version(version=version, config=config, description=description)
    log_audit_event(
        actor=x_user_id or auth["authorized_by"],
        action="policy_upsert",
        resource="policy_versions",
        metadata={"version": version, "roles": auth["roles"]},
    )
    return {"status": "created_or_updated", "version": version}


@app.post("/admin/policies/{version}/activate", tags=["Policy"])
def activate_policy(
    version: str,
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
):
    """Activate one policy version."""
    auth = _require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        required_roles=["policy_admin", "platform_admin"],
    )
    if not activate_policy_version(version):
        raise HTTPException(status_code=404, detail=f"Policy not found: {version}")
    log_audit_event(
        actor=x_user_id or auth["authorized_by"],
        action="policy_activate",
        resource="policy_versions",
        metadata={"version": version, "roles": auth["roles"]},
    )
    return {"status": "activated", "version": version}


@app.get("/admin/policies", tags=["Policy"])
def list_policies(
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
):
    """List policy versions."""
    _require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        required_roles=["policy_viewer", "policy_admin", "platform_admin"],
    )
    return {"active": get_active_policy(), "items": list_policy_versions()}


@app.post("/admin/evals/runs", tags=["Eval"])
def create_eval(
    payload: Dict[str, Any],
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
):
    """Create an eval run (MVP academic harness)."""
    auth = _require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        required_roles=["eval_admin", "researcher", "platform_admin"],
        tenant_id=str(payload.get("tenant_id")) if payload.get("tenant_id") else None,
    )
    prompts = payload.get("prompts") or []
    if not isinstance(prompts, list) or not prompts:
        raise HTTPException(status_code=400, detail="prompts must be a non-empty list")
    prompts = [str(p) for p in prompts if str(p).strip()]
    run_id = payload.get("run_id") or f"eval_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    policy_version = payload.get("policy_version")
    tenant_id = payload.get("tenant_id")
    notes = str(payload.get("notes") or "")
    create_eval_run(
        run_id=run_id,
        prompts=prompts,
        policy_version=str(policy_version) if policy_version else None,
        tenant_id=str(tenant_id) if tenant_id else None,
        notes=notes,
    )
    log_audit_event(
        actor=x_user_id or auth["authorized_by"],
        action="eval_create",
        resource="eval_runs",
        tenant_id=tenant_id,
        metadata={"run_id": run_id, "prompt_count": len(prompts), "roles": auth["roles"]},
    )
    return {"status": "queued", "run_id": run_id, "prompt_count": len(prompts)}


@app.post("/admin/evals/runs/{run_id}/execute", tags=["Eval"])
def execute_eval(
    run_id: str,
    payload: Dict[str, Any] = Body(default={}),
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
):
    """Enqueue asynchronous eval execution in Celery."""
    run = get_eval_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Eval run not found: {run_id}")
    auth = _require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        required_roles=["eval_admin", "researcher", "platform_admin"],
        tenant_id=run.get("tenant_id"),
    )
    task = task_execute_eval_run.delay(
        run_id=run_id,
        modality=str(payload.get("modality") or "text"),
        use_cache=bool(payload.get("use_cache", False)),
        max_tokens=int(payload.get("max_tokens", settings.MAX_TOKENS_DEFAULT)),
        temperature=float(payload.get("temperature", settings.TEMPERATURE_DEFAULT)),
    )
    update_eval_run_status(
        run_id,
        "queued",
        {
            "queued_at": time.time(),
            "task_id": task.id,
            "modality": str(payload.get("modality") or "text"),
            "use_cache": bool(payload.get("use_cache", False)),
            "max_tokens": int(payload.get("max_tokens", settings.MAX_TOKENS_DEFAULT)),
            "temperature": float(payload.get("temperature", settings.TEMPERATURE_DEFAULT)),
        },
    )
    log_audit_event(
        actor=x_user_id or auth["authorized_by"],
        action="eval_execute_queued",
        resource="eval_runs",
        tenant_id=run.get("tenant_id"),
        metadata={"run_id": run_id, "task_id": task.id, "roles": auth["roles"]},
    )
    return {"status": "queued", "run_id": run_id, "task_id": task.id}


@app.get("/admin/evals/runs/{run_id}", tags=["Eval"])
def get_eval(
    run_id: str,
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
):
    """Get eval run details."""
    run = get_eval_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Eval run not found: {run_id}")
    _require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        required_roles=["eval_viewer", "eval_admin", "researcher", "platform_admin"],
        tenant_id=run.get("tenant_id"),
    )
    return run


@app.get("/admin/evals/runs", tags=["Eval"])
def list_evals(
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
):
    """List eval runs."""
    _require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        required_roles=["eval_viewer", "eval_admin", "researcher", "platform_admin"],
    )
    return {"items": list_eval_runs()}


@app.get("/admin/evals/runs/{run_id}/results", tags=["Eval"])
def get_eval_results(
    run_id: str,
    limit: int = 2000,
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
):
    """Get individual result rows for one eval run."""
    run = get_eval_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Eval run not found: {run_id}")
    _require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        required_roles=["eval_viewer", "eval_admin", "researcher", "platform_admin"],
        tenant_id=run.get("tenant_id"),
    )
    return {"run_id": run_id, "items": list_eval_run_results(run_id, limit=limit)}


@app.get("/admin/evals/runs/{run_id}/significance", tags=["Eval"])
def get_eval_significance(
    run_id: str,
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
):
    """Get significance report for model comparisons in one eval run."""
    run = get_eval_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Eval run not found: {run_id}")
    _require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        required_roles=["eval_viewer", "eval_admin", "researcher", "platform_admin"],
        tenant_id=run.get("tenant_id"),
    )
    return eval_significance_report(run_id)


@app.get("/admin/evals/tasks/{task_id}", tags=["Eval"])
def get_eval_task_status(
    task_id: str,
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
):
    """Get Celery task status/result for eval execution."""
    _require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        required_roles=["eval_viewer", "eval_admin", "researcher", "platform_admin"],
    )
    task = AsyncResult(task_id, app=celery_app)
    out: Dict[str, Any] = {
        "task_id": task_id,
        "state": task.state,
        "ready": bool(task.ready()),
        "successful": bool(task.successful()) if task.ready() else False,
    }
    if task.ready():
        try:
            out["result"] = task.result
        except Exception as e:
            out["result_error"] = str(e)
    return out


@app.post("/admin/evals/tasks/{task_id}/cancel", tags=["Eval"])
def cancel_eval_task(
    task_id: str,
    terminate: bool = False,
    x_admin_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
):
    """Cancel/revoke a queued eval Celery task."""
    auth = _require_admin_or_role(
        admin_token=x_admin_token,
        user_id=x_user_id,
        user_roles_header=x_user_roles,
        required_roles=["eval_admin", "platform_admin"],
    )
    celery_app.control.revoke(task_id, terminate=terminate)
    log_audit_event(
        actor=x_user_id or auth["authorized_by"],
        action="eval_task_cancel",
        resource="celery_task",
        metadata={"task_id": task_id, "terminate": terminate, "roles": auth["roles"]},
    )
    return {"status": "revoked", "task_id": task_id, "terminate": terminate}


@app.post("/admin/rbac/grants", tags=["Governance"])
def create_role_grant(
    payload: Dict[str, Any],
    x_admin_token: Optional[str] = Header(None),
):
    """Grant a role to a user. Bootstrap is admin-token only."""
    _require_admin(x_admin_token)
    user_id = str(payload.get("user_id") or "").strip()
    role_name = str(payload.get("role_name") or "").strip()
    tenant_id = payload.get("tenant_id")
    if not user_id or not role_name:
        raise HTTPException(status_code=400, detail="user_id and role_name are required")
    grant_role(user_id=user_id, role_name=role_name, tenant_id=str(tenant_id) if tenant_id else None)
    log_audit_event(actor="admin", action="rbac_grant", resource="rbac_user_roles", tenant_id=str(tenant_id) if tenant_id else None, metadata={"user_id": user_id, "role_name": role_name})
    return {"status": "granted", "user_id": user_id, "role_name": role_name, "tenant_id": tenant_id}


@app.post("/admin/rbac/revokes", tags=["Governance"])
def delete_role_grant(
    payload: Dict[str, Any],
    x_admin_token: Optional[str] = Header(None),
):
    """Revoke a role from a user. Bootstrap is admin-token only."""
    _require_admin(x_admin_token)
    user_id = str(payload.get("user_id") or "").strip()
    role_name = str(payload.get("role_name") or "").strip()
    tenant_id = payload.get("tenant_id")
    if not user_id or not role_name:
        raise HTTPException(status_code=400, detail="user_id and role_name are required")
    removed = revoke_role(user_id=user_id, role_name=role_name, tenant_id=str(tenant_id) if tenant_id else None)
    log_audit_event(actor="admin", action="rbac_revoke", resource="rbac_user_roles", tenant_id=str(tenant_id) if tenant_id else None, metadata={"user_id": user_id, "role_name": role_name, "removed": removed})
    return {"status": "revoked", "removed": removed}


@app.get("/admin/rbac/roles", tags=["Governance"])
def get_rbac_roles(
    user_id: Optional[str] = None,
    x_admin_token: Optional[str] = Header(None),
):
    """List RBAC role bindings."""
    _require_admin(x_admin_token)
    return {"items": list_roles(user_id=user_id)}


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
