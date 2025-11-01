"""
judges.py
----------------------------------------------------
Sistema de avaliação de respostas (juízes LLM + heurístico + RAG).
Cada juiz gera notas entre 0 e 10, que são ponderadas e retornadas ao roteador.
Integração com Prometheus, Chroma e modelos Ollama locais.
"""

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
async def get_rag_context(query: str, n_results: int = 5, max_chars: int = 1500) -> str:
    """Recupera contexto relevante da base vetorial (ChromaDB)."""
    try:
        query_vec = await embed_text(query)
        results = await query_embedding("knowledge_base", query_vec, n_results=n_results)
        if not results or "documents" not in results or not results["documents"]:
            logger.info("[Judges] Nenhum contexto RAG recuperado.")
            return ""
        docs = results["documents"][0]
        context = "\n\n".join(docs).strip()
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
    """Executa julgamento composto (heurístico + LLM) e retorna lista de notas."""
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

        # 2️⃣ Julgamento via LLM
        if mode in ("llm", "hybrid"):
            llm_score = await llm_based_score(query, answer, use_rag)
            results.append({"judge_id": "llm", "score": round(llm_score, 3)})
            logger.info(f"[Judges] LLM: {llm_score:.2f}")

        valid = [r for r in results if "score" in r]
        if not valid:
            return [{"judge_id": "fallback", "score": 0.0}]
        return valid
    except Exception as e:
        logger.error(f"[Judges] Erro inesperado no julgamento: {e}")
        return [{"judge_id": "error", "score": 0.0}]


# ======================================================
# 🔹 Heurística simples (clareza e tamanho)
# ======================================================
def heuristic_score(answer: str) -> float:
    """Score básico com base em tamanho e presença de pontuação."""
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
# 🔹 Avaliação via LLM (com RAG opcional)
# ======================================================
async def llm_based_score(query: str, answer: str, use_rag: bool) -> float:
    """Executa julgamento via múltiplos juízes LLM com ou sem contexto RAG."""
    try:
        judge_models = getattr(settings, "JUDGE_MODELS", [])
        if not judge_models:
            judge_models = [settings.JUDGE_LLM_MODEL] * getattr(settings, "JUDGE_LLM_N", 1)

        context = ""
        if use_rag:
            context = await get_rag_context(query, n_results=5)
            if context:
                logger.info("[Judges] Contexto RAG será usado na avaliação LLM.")

        rag_block = f"\nContexto adicional (via RAG):\n{context}\n" if context else ""
        prompt = (
            "Você é um avaliador especializado em qualidade de respostas de IA.\n"
            "Analise a resposta abaixo com base em:\n"
            "- Correção técnica e factual\n"
            "- Clareza e coerência\n"
            "- Relevância em relação à pergunta\n\n"
            f"Pergunta: {query}\n\n"
            f"Resposta do modelo: {answer}\n\n"
            f"{rag_block}"
            "Responda SOMENTE com um número entre 0 e 10 (use ponto decimal se necessário).\n"
            "Formato de saída obrigatório:\n<nota>\nExemplo: 8.5\n"
            "Não inclua comentários, explicações ou texto adicional."
        ).strip()

        scores = []
        for idx, model in enumerate(judge_models, start=1):
            try:
                text, _ = call_model(
                    model=model,
                    prompt=prompt,
                    temperature=0.1,
                    max_tokens=32
                )
                logger.debug(f"[Judges][RAW OUTPUT] {model}: {text!r}")
                numeric = extract_score(text)
                scores.append(numeric)
                logger.info(f"[Judges] {model} → nota={numeric:.2f} (juiz {idx}/{len(judge_models)})")
            except Exception as e:
                logger.warning(f"[Judges] Falha no julgamento com {model}: {e}")

        if not scores:
            logger.warning("[Judges] Nenhum juiz retornou nota válida. Fallback 5.0.")
            return 0.5

        avg = sum(scores) / len(scores)
        return round(avg / 10.0, 3)
    except Exception as e:
        logger.error(f"[Judges] Erro no julgamento via LLM: {e}")
        return 0.0


# ======================================================
# 🔹 Extração robusta de nota
# ======================================================
def extract_score(text: str) -> float:
    """Extrai número entre 0 e 10, tolerando formatos livres."""
    try:
        if not text:
            return 0.0
        clean = text.strip().lower()

        # 🔍 Captura padrões típicos
        match = re.search(r"(\d+(?:\.\d+)?)(?:\s*/\s*10)?", clean)
        if match:
            val = float(match.group(1))
            return max(0.0, min(val, 10.0))

        # 🔍 Fallback qualitativo (quando o modelo responde com palavras)
        if any(w in clean for w in ["excelente", "ótima", "perfeita", "correta", "impecável"]):
            return 9.0
        if any(w in clean for w in ["boa", "adequada", "razoável", "clara"]):
            return 7.0
        if any(w in clean for w in ["regular", "parcial", "mediana", "ok"]):
            return 5.0
        if any(w in clean for w in ["ruim", "fraca", "errada", "inadequada", "confusa"]):
            return 3.0
        if any(w in clean for w in ["péssima", "horrível", "inútil", "completamente errada"]):
            return 1.0

        logger.debug(f"[Judges] Nenhum número reconhecido em: {clean[:50]}...")
        return 0.0
    except Exception:
        return 0.0
