# Objective: Judge subsystem code for judge.
"""Coordinate heuristic and LLM-based judges for one answer.

The router can evaluate responses using inexpensive heuristics, LLM judges, or
both. This module dispatches the configured judge mix, runs them concurrently,
and returns a normalized list of score dictionaries that the rest of the system
can aggregate.
"""

import asyncio
from typing import List, Dict, Any
from ..settings import settings
from . import heuristic
from . import llm as llm_judge

async def judge_answer(query: str, answer: str, use_rag: bool = True) -> List[Dict[str, Any]]:
    """Evaluate an answer with the judge configuration active at runtime.

    Depending on ``settings.JUDGES_MODE``, the function may schedule heuristic
    judges, LLM judges, or both. If configuration is invalid or empty, it falls
    back to the heuristic trio to avoid returning no signal at all.
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
    """Wrap a heuristic scorer in the same payload shape used by LLM judges."""
    s = fn(q,a)
    return {"judge_id": judge_id, "score": s, "rationale": "heuristic"}
