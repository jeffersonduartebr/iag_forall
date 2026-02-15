def _len_norm(t: str) -> float:
    n = len(t or "")
    if n <= 20: return 4.0
    if n <= 80: return 7.0
    if n <= 800: return 8.5
    return 9.0

def score_coherence(q: str, a: str) -> float:
    if not a: return 0.0
    base = _len_norm(a)
    if q.lower()[:10] in a.lower(): base += 0.5
    return min(10.0, base)

def score_task_fit(q: str, a: str) -> float:
    ql = q.lower(); al = a.lower()
    if any(k in ql for k in ["code","python","sql","docker","traceback","error"]):
        return 7.5 if any(k in al for k in ["def ","class ","select ","stack","error","fix"]) else 6.0
    return 8.0 if len(al) > 60 else 6.5

def score_helpfulness(q: str, a: str) -> float:
    if len(a) < 30: return 5.5
    if "steps" in a.lower() or "passo" in a.lower(): return 8.5
    return 7.5
