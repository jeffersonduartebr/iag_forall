import logging
from typing import List, Dict, Any
from .settings import settings
from .providers import safe_call_model as call_model

logger = logging.getLogger(__name__)

# -------------------------------------------------------
# Julgamento de respostas (avaliadores / "judges")
# -------------------------------------------------------
# Avaliam a qualidade de uma resposta usando:
# - heurísticas simples (e.g., tamanho, clareza)
# - LLMs comerciais (ex: Gemini, GPT)
# - modo híbrido: heurística + LLM
# -------------------------------------------------------


async def judge_answer(query: str, answer: str, use_rag: bool = False) -> List[Dict[str, Any]]:
    """
    Retorna uma lista de julgamentos sobre a qualidade da resposta.
    Cada item é um dict: {"judge_id": str, "score": float}.
    """
    try:
        if not answer or not isinstance(answer, str):
            logger.warning("[Judges] Resposta vazia ou inválida, score=0.")
            return [{"judge_id": "heuristic", "score": 0.0}]

        mode = settings.JUDGES_MODE.lower().strip()
        results = []

        # -------------------------
        # 1️⃣ Julgamento heurístico
        # -------------------------
        if mode in ("heuristic", "hybrid"):
            base_score = heuristic_score(answer)
            results.append({"judge_id": "heuristic", "score": round(base_score, 3)})
            logger.info(f"[Judges] Heurístico: {base_score:.2f}")

        # -------------------------
        # 2️⃣ Julgamento via LLM
        # -------------------------
        if mode in ("llm", "hybrid"):
            llm_score = await llm_based_score(query, answer, use_rag)
            results.append({"judge_id": "llm", "score": round(llm_score, 3)})
            logger.info(f"[Judges] LLM: {llm_score:.2f}")

        # -------------------------
        # Retorno final (normalizado)
        # -------------------------
        valid_results = [r for r in results if isinstance(r, dict) and "score" in r]
        if not valid_results:
            logger.warning("[Judges] Nenhum julgamento válido — retornando score 0.")
            return [{"judge_id": "fallback", "score": 0.0}]

        return valid_results

    except Exception as e:
        logger.error(f"[Judges] Erro inesperado no julgamento: {e}")
        return [{"judge_id": "error", "score": 0.0}]


# ======================================================
# 🔹 Heurística simples baseada em comprimento e clareza
# ======================================================
def heuristic_score(answer: str) -> float:
    """Gera um score simples baseado no tamanho e presença de pontuação."""
    try:
        length = len(answer.strip())
        if length == 0:
            return 0.0

        score = min(1.0, 0.2 + (length / 500))  # saturação em 500 chars
        if any(p in answer for p in [".", "?", "!"]):
            score += 0.2
        return min(score, 1.0)
    except Exception:
        return 0.0


# ======================================================
# 🔹 Avaliação baseada em modelos LLM externos (dinâmica)
# ======================================================
async def llm_based_score(query: str, answer: str, use_rag: bool) -> float:
    """Executa julgamento dinâmico usando a lista de juízes configurada."""
    try:
        # Determina juízes: lista ou fallback
        judge_models = getattr(settings, "JUDGE_MODELS", None)
        if not judge_models:
            judge_models = [settings.JUDGE_LLM_MODEL] * settings.JUDGE_LLM_N

        prompt = f"""
Avalie a resposta de um assistente com base na pergunta e, se relevante, no contexto adicional.
Dê uma nota de 0 a 10 considerando:
- Correção técnica
- Clareza
- Relevância
- Organização textual

Pergunta: {query}
Resposta: {answer}
{"(O contexto RAG foi usado na geração.)" if use_rag else ""}
Responda apenas com um número entre 0 e 10.
""".strip()

        scores = []
        for idx, model in enumerate(judge_models, start=1):
            try:
                text, meta = call_model(
                    model=model,
                    prompt=prompt,
                    max_tokens=32,
                    temperature=0.0,
                )
                numeric = extract_score(text)
                scores.append(numeric)
                logger.info(f"[Judges] {model} → nota={numeric:.2f} (juiz {idx}/{len(judge_models)})")
            except Exception as e:
                logger.warning(f"[Judges] Falha no julgamento com {model}: {e}")

        if not scores:
            return 0.0

        avg = sum(scores) / len(scores)
        normalized = round(avg / 10.0, 3)
        return normalized

    except Exception as e:
        logger.error(f"[Judges] Erro no julgamento via LLM: {e}")
        return 0.0



# ======================================================
# 🔹 Utilitário para extrair número da resposta do LLM
# ======================================================
def extract_score(text: str) -> float:
    """Extrai número da resposta textual do LLM (0–10)."""
    import re
    try:
        numbers = re.findall(r"\d+(?:\.\d+)?", text or "")
        if not numbers:
            return 0.0
        val = float(numbers[0])
        return max(0.0, min(val, 10.0))
    except Exception:
        return 0.0
