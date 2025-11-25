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
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field

# ============================================================
# 1. Entrada (Request)
# ============================================================

class QueryRequest(BaseModel):
    """
    Payload de entrada para o endpoint /query.
    """
    query: str = Field(..., description="Texto da pergunta do usuário.")
    
    # Controle de Modalidade
    modality: str = Field("text", description="Modalidade desejada: text, vision, multimodal.")
    
    # Imagens (Suporte a lista ou item único para retrocompatibilidade)
    images: Optional[List[str]] = Field(None, description="Lista de imagens em Base64.")
    image_b64: Optional[str] = Field(None, description="Imagem única em Base64 (Legacy).")
    
    # Configurações de RAG
    enable_rag_for_answer: bool = Field(False, description="Ativar RAG para gerar a resposta.")
    enable_rag_for_image: bool = Field(False, description="Ativar RAG usando a imagem como query.")
    rag_modality: str = Field("text", description="Modalidade de busca no RAG: text, vision, multimodal.")
    
    # Parâmetros de Geração (LLM)
    max_tokens: int = Field(512, description="Limite de tokens na resposta.")
    temperature: float = Field(0.5, description="Criatividade do modelo (0.0 a 1.0).")
    system_prompt: Optional[str] = Field(None, description="Instrução de sistema (System Message).")
    
    # Controle de Cache
    use_cache: bool = Field(True, description="Se False, força o processamento ignorando o cache semântico.")


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
    
    # Metadados de Decisão
    route: RouteDecision
    
    # Detalhes dos Candidatos (se houver comparação ou fallback)
    candidates: List[CandidateResult] = []
    
    # Payload bruto do provedor vencedor (para debug)
    # Usamos Any para aceitar tanto Dict quanto String JSON sem quebrar a validação
    payload: Optional[Any] = None