from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class QueryRequest(BaseModel):
    query: str
    enable_rag_for_answer: bool = False
    max_tokens: int = 512
    temperature: float = 0.5
    system_prompt: Optional[str] = None 


class JudgeScore(BaseModel):
    judge_id: str
    score: float
    rationale: str = ""

class CandidateResult(BaseModel):
    model: str
    output: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float
    judge_scores: List[JudgeScore] = []
    quality_score: float = 0.0

class RouteDecision(BaseModel):
    chosen_model: str
    objectives: Dict[str, float]
    pareto_front: List[Dict[str, Any]] = []
    explanation: str = ""

class QueryResponse(BaseModel):
    answer: str
    model: str
    route: RouteDecision
    candidates: List[CandidateResult]
