"""Módulo principal: descreve responsabilidades e integrações deste arquivo."""

import asyncio
from typing import List, Dict, Any
from ..settings import settings
from . import heuristic
from . import llm as llm_judge

async def judge_answer(query: str, answer: str, use_rag: bool = True) -> List[Dict[str, Any]]:
    """Resumo do comportamento desta função.

    Args:
        query: Parâmetro de entrada.
        answer: Parâmetro de entrada.
        use_rag: Parâmetro de entrada.

    Returns:
        Valor retornado pela função.
    """
    mode = settings.JUDGES_MODE.lower()
    tasks = []
    if mode in ("heuristic","hybrid"):
        tasks += [
            _heuristic_task("coherence", heuristic.score_coherence, query, answer),
            _heuristic_task("task_fit", heuristic.score_task_fit, query, answer),
            _heuristic_task("helpfulness", heuristic.score_helpfulness, query, answer),
        ]
    if mode in ("llm","hybrid"):
        n = max(1, settings.JUDGE_LLM_N)
        for i in range(n):
            tasks.append(llm_judge.score(query, answer, i))
    if not tasks:
        tasks = [
            _heuristic_task("coherence", heuristic.score_coherence, query, answer),
            _heuristic_task("task_fit", heuristic.score_task_fit, query, answer),
            _heuristic_task("helpfulness", heuristic.score_helpfulness, query, answer),
        ]
    results = await asyncio.gather(*tasks)
    return results

async def _heuristic_task(judge_id: str, fn, q: str, a: str) -> Dict[str, Any]:
    """Resumo do comportamento desta função.

    Args:
        judge_id: Parâmetro de entrada.
        fn: Parâmetro de entrada.
        q: Parâmetro de entrada.
        a: Parâmetro de entrada.

    Returns:
        Valor retornado pela função.
    """
    s = fn(q,a)
    return {"judge_id": judge_id, "score": s, "rationale": "heuristic"}
