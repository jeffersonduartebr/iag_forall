"""Módulo principal: descreve responsabilidades e integrações deste arquivo."""

import asyncio, json
from typing import Dict, Any
from ..settings import settings
from ..rag import retrieve_context
from litellm import completion

RUBRIC_BASE = """
You are a strict evaluator. Score the ASSISTANT answer for the USER question from 0 to 10.
Consider: (1) factual correctness, (2) task completion, (3) clarity & structure, (4) harmful content avoidance.
Return ONLY a JSON object: {"score": <0-10>, "rationale": "<short reason>"}.
"""

def _build_prompt(user_q: str, assistant_a: str, use_rag: bool) -> str:
    """Resumo do comportamento desta função.

    Args:
        user_q: Parâmetro de entrada.
        assistant_a: Parâmetro de entrada.
        use_rag: Parâmetro de entrada.

    Returns:
        Valor retornado pela função.
    """
    ctx = ""
    if use_rag:
        rag = retrieve_context(user_q)
        if rag:
            ctx = f"CONTEXT (may be useful, optional):\n---\n{rag}\n---\n"
    return f"""{RUBRIC_BASE}
{ctx}
USER QUESTION:
{user_q}

ASSISTANT ANSWER:
{assistant_a}
"""

def _score_sync(user_q: str, assistant_a: str, judge_id: str, use_rag: bool) -> Dict[str, Any]:
    """Resumo do comportamento desta função.

    Args:
        user_q: Parâmetro de entrada.
        assistant_a: Parâmetro de entrada.
        judge_id: Parâmetro de entrada.
        use_rag: Parâmetro de entrada.

    Returns:
        Valor retornado pela função.
    """
    prompt = _build_prompt(user_q, assistant_a, use_rag)
    resp = completion(model=settings.JUDGE_LLM_MODEL,
                      messages=[{"role":"user","content": prompt}],
                      temperature=0.0, max_tokens=128)
    text = resp.choices[0].message["content"]
    try:
        j = json.loads(text)
        score = float(j.get("score", 0))
        rationale = str(j.get("rationale", ""))[:300]
    except Exception:
        score = 0.0
        rationale = text[:280]
    score = max(0.0, min(10.0, score))
    return {"judge_id": judge_id, "score": score, "rationale": rationale}

async def score(user_q: str, assistant_a: str, idx: int) -> Dict[str, Any]:
    """Resumo do comportamento desta função.

    Args:
        user_q: Parâmetro de entrada.
        assistant_a: Parâmetro de entrada.
        idx: Parâmetro de entrada.

    Returns:
        Valor retornado pela função.
    """
    judge_id = f"llm_{idx}"
    return await asyncio.to_thread(_score_sync, user_q, assistant_a, judge_id, settings.JUDGE_USE_RAG)
