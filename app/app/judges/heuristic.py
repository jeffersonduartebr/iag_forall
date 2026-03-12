# Objective: Judge subsystem code for heuristic.
"""Provide lightweight heuristic judges for answer quality estimation.

These functions are intentionally cheap and deterministic. They are used when
the system runs in heuristic mode, as part of hybrid judging, or as a fallback
when an LLM-based judge is unavailable. The scores returned by this module are
coarse proxies rather than formal evaluation metrics.
"""

def _len_norm(t: str) -> float:
    """Map answer length to a coarse quality prior.

    Very short answers usually underperform, while moderately long answers tend
    to carry more structure and context. The function returns a bounded score on
    the same 0-10 scale used by the rest of the judging pipeline.
    """
    n = len(t or "")
    if n <= 20: return 4.0
    if n <= 80: return 7.0
    if n <= 800: return 8.5
    return 9.0

def score_coherence(q: str, a: str) -> float:
    """Estimate answer coherence from basic textual signals.

    The heuristic rewards sufficiently detailed answers and gives a small boost
    when the answer clearly echoes part of the original question, which usually
    indicates topical alignment.
    """
    if not a: return 0.0
    base = _len_norm(a)
    if q.lower()[:10] in a.lower(): base += 0.5
    return min(10.0, base)

def score_task_fit(q: str, a: str) -> float:
    """Estimate how well the answer matches the type of task requested.

    Technical prompts receive a slightly different scoring rule that favors
    outputs containing code or debugging language. For general prompts, the
    heuristic mostly distinguishes between shallow and sufficiently developed
    answers.
    """
    ql = q.lower(); al = a.lower()
    if any(k in ql for k in ["code","python","sql","docker","traceback","error"]):
        return 7.5 if any(k in al for k in ["def ","class ","select ","stack","error","fix"]) else 6.0
    return 8.0 if len(al) > 60 else 6.5

def score_helpfulness(q: str, a: str) -> float:
    """Estimate practical helpfulness from answer structure and specificity.

    The heuristic penalizes answers that are too short to be actionable and
    rewards outputs that look procedural, for example when they include explicit
    steps.
    """
    if len(a) < 30: return 5.5
    if "steps" in a.lower() or "passo" in a.lower(): return 8.5
    return 7.5
