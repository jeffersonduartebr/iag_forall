# -*- coding: utf-8 -*-
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
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, field_validator
from enum import Enum


# ============================================================
# Enums for Validation
# ============================================================

class Modality(str, Enum):
    """Representa a responsabilidade principal desta classe."""
    TEXT = "text"
    VISION = "vision"
    MULTIMODAL = "multimodal"


# ============================================================
# 1. Entrada (Request)
# ============================================================

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

    # Metadados de Decisão
    route: RouteDecision

    # Detalhes dos Candidatos (se houver comparação ou fallback)
    candidates: List[CandidateResult] = []

    # Payload bruto do provedor vencedor (para debug)
    # Usamos Any para aceitar tanto Dict quanto String JSON sem quebrar a validação
    payload: Optional[Any] = None
