import logging
import re
from typing import List, Dict, Any
from .settings import settings
from .providers import call_model
from .vectorstore import query_embedding
from .embeddings import embed_text

logger = logging.getLogger(__name__)

# ======================================================
# 🔹 Recuperação de contexto (RAG dinâmico)
# ======================================================
def get_rag_context(query: str, n_results: int = 5, max_chars: int = 1500) -> str:
    """
    Recupera contexto relevante da base vetorial (ChromaDB).
    Retorna um texto concatenado com os melhores trechos.
    Limita o tamanho total para evitar ultrapassar o limite de tokens.
    """
    try:
        query_vec = embed_text(query)
        results = query_embedding("knowledge_base", query_vec, n_results=n_results)

        if not results or "documents" not in results or not results["documents"]:
            logger.info("[Judges] Nenhum contexto RAG recuperado.")
            return ""

        docs = results["documents"][0]
        context = "\n\n".join(docs)
        context = context.strip()

        if len(context) > max_chars:
            context = context[:max_chars] + "..."
        logger.info(f"[Judges] {len(docs)} trechos recuperados via RAG.")
        return context
    except Exception as e:
        logger.error(f"[Judges] Falha ao obter contexto RAG: {e}")
        return ""


# ======================================================
# 🔹 Função principal de julgamento
# ======================================================
async def judge_answer(query: str, answer: str, use_rag: bool = False) -> List[Dict[str, Any]]:
    """
    Retorna uma lista de julgamentos sobre a qualidade da resposta.
    Cada item: {"judge_id": str, "score": float}.
    """
    try:
        if not answer or not isinstance(answer, str):
            logger.warning("[Judges] Resposta vazia ou inválida, score=0.")
            return [{"judge_id": "heuristic", "score": 0.0}]

        mode = settings.JUDGES_MODE.lower().strip()
        results = []

        # 1️⃣ Heurístico simples
        if mode in ("heuristic", "hybrid"):
            base_score = heuristic_score(answer)
            results.append({"judge_id": "heuristic", "score": round(base_score, 3)})
            logger.info(f"[Judges] Heurístico: {base_score:.2f}")

        # 2️⃣ LLM-based (com RAG opcional)
        if mode in ("llm", "hybrid"):
            llm_score = await llm_based_score(query, answer, use_rag)
            results.append({"judge_id": "llm", "score": round(llm_score, 3)})
            logger.info(f"[Judges] LLM: {llm_score:.2f}")

        valid_results = [r for r in results if "score" in r]
        if not valid_results:
            return [{"judge_id": "fallback", "score": 0.0}]
        return valid_results
    except Exception as e:
        logger.error(f"[Judges] Erro inesperado no julgamento: {e}")
        return [{"judge_id": "error", "score": 0.0}]


# ======================================================
# 🔹 Heurística simples (clareza e tamanho)
# ======================================================
def heuristic_score(answer: str) -> float:
    """Gera score com base em tamanho e pontuação."""
    try:
        length = len(answer.strip())
        if length == 0:
            return 0.0
        score = min(1.0, 0.2 + (length / 500))
        if any(p in answer for p in [".", "?", "!"]):
            score += 0.2
        return min(score, 1.0)
    except Exception:
        return 0.0


# ======================================================
# 🔹 Avaliação via modelos LLM (com RAG opcional)
# ======================================================
async def llm_based_score(query: str, answer: str, use_rag: bool) -> float:
    """
    Executa julgamento dinâmico usando múltiplos juízes LLM,
    incorporando contexto RAG se solicitado.
    """
    try:
        judge_models = getattr(settings, "JUDGE_MODELS", None)
        if not judge_models:
            judge_models = [settings.JUDGE_LLM_MODEL] * settings.JUDGE_LLM_N

        context = ""
        if use_rag:
            context = get_rag_context(query, n_results=5)
            if context:
                logger.info("[Judges] Contexto RAG será usado na avaliação LLM.")

        # prompt enxuto mas completo
        prompt = f"""
Você é um avaliador de respostas de IA.
Avalie a resposta abaixo com base nos critérios:

1️⃣ Correção técnica e factual
2️⃣ Clareza e coerência textual
3️⃣ Relevância ao que foi perguntado

Pergunta: {query}

Resposta do modelo: {answer}

{f"Contexto adicional (via RAG):\n{context}\n" if context else ""}

Retorne apenas um número entre 0 e 10.
""".strip()

        scores = []
        for idx, model in enumerate(judge_models, start=1):
            try:
                text, _ = call_model(
                    model=model,
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=32
                )
                numeric = extract_score(text)
                scores.append(numeric)
                logger.info(f"[Judges] {model} → nota={numeric:.2f} (juiz {idx}/{len(judge_models)})")
            except Exception as e:
                logger.warning(f"[Judges] Falha no julgamento com {model}: {e}")

        if not scores:
            return 0.0
        avg = sum(scores) / len(scores)
        return round(avg / 10.0, 3)
    except Exception as e:
        logger.error(f"[Judges] Erro no julgamento via LLM: {e}")
        return 0.0


# ======================================================
# 🔹 Utilitário: extrair número
# ======================================================
def extract_score(text: str) -> float:
    """Extrai número entre 0 e 10 do texto de saída do LLM."""
    try:
        numbers = re.findall(r"\d+(?:\.\d+)?", text or "")
        if not numbers:
            return 0.0
        val = float(numbers[0])
        return max(0.0, min(val, 10.0))
    except Exception:
        return 0.0
