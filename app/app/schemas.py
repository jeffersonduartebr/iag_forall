# -*- coding: utf-8 -*-
# Objective: Application runtime code for schemas.
"""
schemas.py (VERSÃO COMPLETA DE PRODUÇÃO)
----------------------------------------
Define os contratos de dados para a API.
Suporta:
- Entradas Multimodais (Texto + Imagem)
- Controle Fino de RAG
- Auditoria de Decisão (Rota, Pareto, Juízes)
- Telemetria Detalhada (Tokens, Custo, Latência)
- Embeddings Vetoriais (para log e debug)
- Payloads flexíveis (Dict ou Str)
- Strong input validation
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

# ============================================================
# Enums for Validation
# ============================================================

class Modality(str, Enum):
    """Represent `Modality` within this module.

The class groups the state and behavior required for Modality."""
    TEXT = "text"
    VISION = "vision"
    MULTIMODAL = "multimodal"


class ConfidenceBand(str, Enum):
    """Represent supported confidence buckets exposed by the public response contract."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class VerificationStatus(str, Enum):
    """Represent post-generation verification states for one answer."""
    SUPPORTED = "supported"
    WEAKLY_SUPPORTED = "weakly_supported"
    UNSUPPORTED = "unsupported"


class ReviewStatus(str, Enum):
    """Represent the human-review lifecycle state for one answer."""
    AUTO_APPROVED = "auto_approved"
    NEEDS_REVIEW = "needs_review"
    REVIEWED = "reviewed"
    REJECTED = "rejected"


class QueryJobStatus(str, Enum):
    """Represent the lifecycle state of one asynchronously queued query."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


class AbstainReason(str, Enum):
    """Represent machine-readable reasons for intentionally abstaining."""
    EMPTY_ANSWER = "empty_answer"
    LOW_CONFIDENCE = "low_confidence"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class QualitySource(str, Enum):
    """Represent the provenance of one persisted quality signal."""
    JUDGE = "judge"
    BANDIT_PROXY = "bandit_proxy"
    FALLBACK_DEFAULT = "fallback_default"
    UNKNOWN = "unknown"


# ============================================================
# 1. Entrada (Request)
# ============================================================

class WorkloadHints(BaseModel):
    """Optional client hints; complexity is always detected at runtime."""
    theme: Optional[str] = Field(None, max_length=128, description="Benchmark or domain theme for telemetry.")
    benchmark_id: Optional[str] = Field(None, max_length=128, description="Catalog entry id when known.")
    expected_tokens: Optional[int] = Field(None, ge=32, le=32000, description="Desired response length floor.")


class QueryRequest(BaseModel):
    """
    Payload de entrada para o endpoint /query.
    """
    query: str = Field(
        ...,
        min_length=1,
        max_length=100000,
        description="Texto da pergunta do usuário."
    )

    # Controle de Modalidade
    modality: str = Field(
        "text",
        description="Modalidade desejada: text, vision, multimodal."
    )

    # Imagens (Suporte a lista ou item único para retrocompatibilidade)
    images: Optional[List[str]] = Field(
        None,
        max_length=10,
        description="Lista de imagens em Base64 (máximo 10)."
    )
    image_b64: Optional[str] = Field(
        None,
        max_length=10_000_000,  # ~7.5MB base64
        description="Imagem única em Base64 (Legacy)."
    )

    # Configurações de RAG
    enable_rag_for_answer: bool = Field(
        False,
        description="Ativar RAG para gerar a resposta."
    )
    enable_rag_for_image: bool = Field(
        False,
        description="Ativar RAG usando a imagem como query."
    )
    rag_modality: str = Field(
        "text",
        description="Modalidade de busca no RAG: text, vision, multimodal."
    )

    # Parâmetros de Geração (LLM)
    max_tokens: int = Field(
        512,
        ge=1,
        le=32000,
        description="Limite de tokens na resposta (1-32000)."
    )
    temperature: float = Field(
        0.5,
        ge=0.0,
        le=2.0,
        description="Criatividade do modelo (0.0 a 2.0)."
    )
    system_prompt: Optional[str] = Field(
        None,
        max_length=50000,
        description="Instrução de sistema (System Message)."
    )

    # Controle de Cache
    use_cache: bool = Field(
        True,
        description="Se False, força o processamento ignorando o cache semântico."
    )

    # Request timeout (optional override)
    timeout_seconds: Optional[int] = Field(
        None,
        ge=5,
        le=600,
        description="Timeout da requisição em segundos (5-600)."
    )

    # Roadmap MVP: governança e experimentação
    tenant_id: Optional[str] = Field(
        None,
        max_length=128,
        description="Identificador do tenant para quota, auditoria e segmentação.",
    )
    stream: bool = Field(
        False,
        description="Quando true, permite uso de endpoint de streaming SSE.",
    )
    policy_version: Optional[str] = Field(
        None,
        max_length=128,
        description="Versão de política de roteamento solicitada.",
    )
    experiment_id: Optional[str] = Field(
        None,
        max_length=128,
        description="ID de experimento A/B para atribuição de variante.",
    )
    user_key: Optional[str] = Field(
        None,
        max_length=256,
        description="Chave estável de usuário para assignment consistente em experimento.",
    )
    webhook_url: Optional[str] = Field(
        None,
        max_length=2048,
        description="URL para notificação HTTP quando job assíncrono (202) for concluído.",
    )
    workload_hints: Optional[WorkloadHints] = Field(
        None,
        description="Optional domain hints; routing complexity is inferred at runtime.",
    )

    # --- Tool / function calling (formato OpenAI, pass-through) ---
    tools: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Lista de tools no formato OpenAI: [{type:function, function:{name,description,parameters}}].",
    )
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(
        None,
        description="Controle de tools: 'auto' | 'none' | 'required' | {type:function, function:{name}}.",
    )
    messages: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Histórico multi-turn (formato OpenAI) para follow-up com resultados de tools (role:tool).",
    )

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        """Ensure query is not just whitespace."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("Query cannot be empty or whitespace only")
        return stripped

    @field_validator("modality", "rag_modality")
    @classmethod
    def validate_modality(cls, v: str) -> str:
        """Validate modality is one of the allowed values."""
        allowed = {"text", "vision", "multimodal"}
        v_lower = v.lower().strip()
        if v_lower not in allowed:
            raise ValueError(f"Modality must be one of: {allowed}")
        return v_lower


# ============================================================
# 2. Estruturas de Auditoria e Métricas
# ============================================================

class JudgeScore(BaseModel):
    """
    Registro de avaliação de um juiz específico.
    """
    judge_id: str
    score: float
    rationale: Optional[str] = None
    modality: Optional[str] = "text"


class CandidateResult(BaseModel):
    """
    Representa o resultado de um modelo candidato (antes da escolha final ou para comparação).
    """
    model: str
    modality: str = "text"

    # Saídas
    output: str = ""
    image_output_b64: Optional[str] = None

    # Métricas de Execução
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0

    # Avaliação (Quality)
    judge_scores: List[JudgeScore] = []
    quality_score: float = 0.0

    # Vetores (Opcional - geralmente omitido na API pública para economizar banda, mas útil para debug)
    embedding_vectors: Optional[Dict[str, List[float]]] = None

    # Dados Brutos do Provider (flexível para aceitar Dict ou String JSON)
    payload: Optional[Any] = None

    @model_validator(mode="before")
    @classmethod
    def map_legacy_cost_field(cls, value: Any) -> Any:
        """Accept legacy candidate payloads that still expose `cost_per_1k`."""
        if isinstance(value, dict) and "estimated_cost_usd" not in value and "cost_per_1k" in value:
            cloned = dict(value)
            cloned["estimated_cost_usd"] = cloned.get("cost_per_1k")
            return cloned
        return value

    @field_validator("estimated_cost_usd", mode="before")
    @classmethod
    def normalize_estimated_cost_usd(cls, value: Any) -> float:
        """Accept both the new public name and older internal aliases for candidate cost."""
        if value is None:
            return 0.0
        return float(value)


# ============================================================
# 3. Decisão de Roteamento
# ============================================================

class RouteDecision(BaseModel):
    """
    Explica o porquê de um modelo ter sido escolhido.
    """
    chosen_model: str
    modality_selected: str = "text"
    is_multimodal_route: bool = False

    # Objetivos otimizados (NSGA-II / Bandit)
    objectives: Dict[str, float] = Field(
        default_factory=dict,
        description="Métricas consideradas: cost, latency, quality, reward."
    )

    # Fronteira de Pareto (para visualização de trade-offs)
    pareto_front: List[Dict[str, Any]] = []

    explanation: str = ""


# ============================================================
# 4. Saída (Response)
# ============================================================

class ResponseCitation(BaseModel):
    """Represent one evidence source attached to a grounded answer."""
    doc_id: str
    rank: int = 1
    source: Optional[str] = None
    snippet: Optional[str] = None
    score: Optional[float] = None


class EvidenceSnippet(BaseModel):
    """Expose one short evidence excerpt used to support the response."""
    doc_id: str
    text: str
    rank: int = 1
    source: Optional[str] = None


class ResponseProvenance(BaseModel):
    """Group all grounding and evidence-related fields under one stable object."""
    grounded: bool = False
    knowledge_version: Optional[str] = None
    citations: List[ResponseCitation] = Field(default_factory=list)
    evidence_snippets: List[EvidenceSnippet] = Field(default_factory=list)


class ResponseDiagnostics(BaseModel):
    """Carry optional routing/debug details without polluting the public answer surface."""
    route: Optional[RouteDecision] = None
    candidates: List[CandidateResult] = Field(default_factory=list)
    payload: Optional[Any] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    uncertainty_score: Optional[float] = None
    guardrail_output_tags: List[str] = Field(default_factory=list)
    experiment_id: Optional[str] = None
    experiment_variant: Optional[Dict[str, Any]] = None
    tenant_id: Optional[str] = None
    perf_mode_enabled: Optional[bool] = None
    retrieval_mode: Optional[str] = None
    workload_class: Optional[str] = None
    detected_complexity: Optional[str] = None
    raw_payload: Optional[Any] = None


class QueryResponse(BaseModel):
    """
    Resposta final enviada ao cliente.
    """
    # Conteúdo Principal
    answer: str
    model: str
    modality: str = "text"
    image_output_b64: Optional[str] = None

    # Request Tracing
    correlation_id: Optional[str] = Field(
        None,
        description="Unique request correlation ID for end-to-end tracing"
    )

    estimated_cost_usd: Optional[float] = Field(
        None,
        ge=0.0,
        description="Estimated total cost in USD for the completed answer.",
    )

    # Qualidade operacional e confiabilidade
    confidence_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Normalized confidence score for the final answer.",
    )
    confidence_band: Optional[ConfidenceBand] = Field(
        None,
        description="Bucketed confidence label: high, medium, or low.",
    )
    abstained: bool = Field(
        False,
        description="Whether the system intentionally abstained instead of providing a normal answer.",
    )
    abstain_reason: Optional[AbstainReason] = Field(
        None,
        description="Short machine-readable reason for abstention.",
    )
    verification_status: Optional[VerificationStatus] = Field(
        None,
        description="Post-generation verification status such as supported, weakly_supported, or unsupported.",
    )
    review_status: Optional[ReviewStatus] = Field(
        None,
        description="Human-review lifecycle state for the answer.",
    )
    provenance: ResponseProvenance = Field(
        default_factory=ResponseProvenance,
        description="Grounding and provenance metadata for the answer.",
    )
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Tool/function calls solicitadas pelo modelo (formato OpenAI). O cliente executa e reenvia.",
    )
    finish_reason: Optional[str] = Field(
        None,
        description="Motivo de término: 'stop' | 'tool_calls' | 'length' | ...",
    )
    diagnostics: Optional[ResponseDiagnostics] = Field(
        None,
        description="Optional internal routing and debug information.",
    )


class QueuedQueryAcceptedResponse(BaseModel):
    """Return the accepted job contract when a query is deferred to the async queue."""
    job_id: str
    status: QueryJobStatus = QueryJobStatus.QUEUED
    poll_url: str
    result_url: str
    expires_at: float
    queue_depth: int = 0
    estimated_wait_seconds: Optional[float] = None
    poll_after_seconds: float = 1.0


class QueryJobStatusResponse(BaseModel):
    """Expose the observable state of one queued query job."""
    job_id: str
    status: QueryJobStatus
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    expires_at: Optional[float] = None
    error: Optional[Dict[str, Any]] = None
    poll_after_seconds: float = 1.0


class AdminSettingsUpdateRequest(BaseModel):
    """Wrap dynamic setting updates behind an explicit request contract."""
    settings: Dict[str, Any] = Field(default_factory=dict)


class TenantBudgetUpdateRequest(BaseModel):
    """Represent the payload used to configure tenant budget limits."""
    daily_usd_limit: float = 0.0
    monthly_usd_limit: float = 0.0
    enabled: bool = True


class PolicyCreateRequest(BaseModel):
    """Represent a policy upsert request."""
    version: str = Field(..., min_length=1, max_length=128)
    description: str = ""
    config: Dict[str, Any] = Field(default_factory=dict)


class RoleGrantRequest(BaseModel):
    """Represent a role-grant request payload."""
    user_id: str = Field(..., min_length=1, max_length=128)
    role_name: str = Field(..., min_length=1, max_length=64)
    tenant_id: Optional[str] = Field(None, max_length=128)


class RoleRevokeRequest(RoleGrantRequest):
    """Represent a role-revoke request payload."""


class ResponseReviewUpdateRequest(BaseModel):
    """Represent one reviewer decision for an answer waiting in the review queue."""
    review_status: ReviewStatus
    reviewer_notes: Optional[str] = None
    corrected_answer: Optional[str] = None


class ExpertProfileUpdateRequest(BaseModel):
    """Expert reviewer profile with areas of expertise (benchmark themes)."""
    display_name: Optional[str] = Field(None, max_length=255)
    theme_ids: List[str] = Field(default_factory=list, description="Benchmark theme ids the expert can review.")
    credentials_note: Optional[str] = Field(None, max_length=512)


class ExpertLoginRequest(BaseModel):
    """Expert portal login with email and password."""
    email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=1, max_length=256)


class ExpertAccountCreateRequest(BaseModel):
    """Admin registration payload for a new human expert."""
    display_name: str = Field(..., min_length=2, max_length=255)
    email: str = Field(..., min_length=5, max_length=255)
    phone: Optional[str] = Field(None, max_length=32)
    password: str = Field(..., min_length=8, max_length=256)


class ExpertAccountUpdateRequest(BaseModel):
    """Admin update payload for an existing expert account."""
    display_name: Optional[str] = Field(None, min_length=2, max_length=255)
    phone: Optional[str] = Field(None, max_length=32)
    password: Optional[str] = Field(None, min_length=8, max_length=256)
    enabled: Optional[bool] = None


class ExpertRubricScores(BaseModel):
    """Pedagogical rubric dimensions scored 0-10 by human experts."""
    factual_correctness: float = Field(..., ge=0.0, le=10.0)
    task_completion: float = Field(..., ge=0.0, le=10.0)
    clarity_structure: float = Field(..., ge=0.0, le=10.0)
    scaffolding: float = Field(..., ge=0.0, le=10.0)
    audience_fit: float = Field(..., ge=0.0, le=10.0)
    misconception_handling: float = Field(..., ge=0.0, le=10.0)


class ExpertPreviewRequest(BaseModel):
    """Run one benchmark query through the router for expert review."""
    query: str = Field(..., min_length=1)
    theme: Optional[str] = Field(None, max_length=128)
    benchmark_id: Optional[str] = Field(None, max_length=128)
    frozen_policy: bool = False


class ExpertAssessmentSubmitRequest(BaseModel):
    """Human expert analysis for one benchmark query + system answer."""
    benchmark_id: str = Field(..., min_length=1, max_length=128)
    theme: str = Field(..., min_length=1, max_length=128)
    query_text: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    reference: Optional[str] = None
    eval_run_id: Optional[str] = Field(None, max_length=64)
    judge_quality: Optional[float] = Field(None, ge=0.0, le=10.0)
    quality_score: float = Field(..., ge=0.0, le=10.0)
    rubric: ExpertRubricScores
    notes: Optional[str] = None


class EvalRunCreateRequest(BaseModel):
    """Represent the creation payload for one evaluation run."""
    prompts: List[str] = Field(default_factory=list)
    run_id: Optional[str] = Field(None, max_length=64)
    policy_version: Optional[str] = Field(None, max_length=128)
    tenant_id: Optional[str] = Field(None, max_length=128)
    notes: str = ""
    golden_set_id: Optional[str] = Field(None, max_length=128)
    benchmark_theme: Optional[str] = Field(None, max_length=128)
    benchmark_themes: List[str] = Field(default_factory=list)
    benchmark_sample_size: Optional[int] = Field(None, ge=1, le=500)
    benchmark_tags: List[str] = Field(default_factory=list)
    benchmark_difficulty: Optional[str] = Field(
        None,
        max_length=32,
        description="Catalog sampling filter only; routing detects complexity at runtime.",
    )
    benchmark_seed: Optional[int] = Field(None, description="Deterministic catalog sample seed.")
    benchmark_split: Optional[str] = Field(
        None,
        max_length=16,
        description="Catalog split filter: dev or held_out (primary metrics should use held_out).",
    )
    frozen_policy: bool = Field(
        False,
        description="Freeze routing weights/exploration during eval for reproducible academic runs.",
    )


class EvalRunExecuteRequest(BaseModel):
    """Represent runtime overrides when enqueueing an eval run."""
    modality: str = "text"
    use_cache: bool = False
    max_tokens: int = Field(512, ge=1, le=32000)
    temperature: float = Field(0.5, ge=0.0, le=2.0)
