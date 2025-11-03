# -*- coding: utf-8 -*-
"""
judges.py
----------------------------------------------------
Sistema de avaliação de respostas (juízes LLM + heurístico + RAG)
com fallback dinâmico e auditoria em banco de dados.
----------------------------------------------------
Novidades:
✅ Seleção aleatória de 2 juízes a cada julgamento
✅ Fallback dinâmico baseado em JUDGE_MODELS (Redis)
✅ Registro detalhado de todos os eventos de fallback
✅ Total rastreabilidade das decisões de avaliação
"""

import logging
import re
import random
from typing import List, Dict, Any
from sqlalchemy import create_engine, text
from .settings_dynamic import settings
from .providers import call_model
from .vectorstore import query_embedding
from .embeddings import embed_text

logger = logging.getLogger(__name__)

# ======================================================
# 🔧 Banco de dados (para auditoria)
# ======================================================
DB_HOST = settings.DB_HOST
DB_USER = settings.DB_USER
DB_PASS = settings.DB_PASS
DB_NAME = settings.DB_NAME
DB_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:3306/{DB_NAME}"
engine = create_engine(DB_URL, pool_pre_ping=True)

DISAGREE_PERCENT = 0.20 # 20% de desacordo aciona meta-avaliação

# ======================================================
# 🔹 Recuperação de contexto (RAG dinâmico)
# ======================================================
async def get_rag_context(query: str, n_results: int = 5, max_chars: int = 1500) -> str:
    """Recupera contexto relevante da base vetorial (ChromaDB)."""
    try:
        query_vec = await embed_text(query)
        collection_name = settings.get("RAG_COLLECTION_NAME", "knowledge_base")
        results = await query_embedding(collection_name, query_vec, n_results=n_results)
        if not results or "documents" not in results or not results["documents"]:
            logger.info("[Judges] Nenhum contexto RAG recuperado.")
            return ""
        docs = results["documents"][0]
        context = "\n\n".join(docs).strip()
        if len(context) > max_chars:
            context = context[:max_chars] + "..."
        return context
    except Exception as e:
        logger.error(f"[Judges] Falha ao obter contexto RAG: {e}")
        return ""

# ======================================================
# 🔹 Função principal de julgamento
# ======================================================
async def judge_answer(query: str, answer: str, use_rag: bool = False) -> List[Dict[str, Any]]:
    """Retorna uma lista de julgamentos sobre a qualidade da resposta."""
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

        # 2️⃣ LLM-based (com RAG e fallback dinâmico)
        if mode in ("llm", "hybrid"):
            llm_score = await llm_based_score(query, answer, use_rag)
            results.append({"judge_id": "llm", "score": round(llm_score, 3)})
            logger.info(f"[Judges] LLM: {llm_score:.2f}")

        valid_results = [r for r in results if "score" in r]
        return valid_results or [{"judge_id": "fallback", "score": 0.0}]
    except Exception as e:
        logger.error(f"[Judges] Erro inesperado no julgamento: {e}")
        return [{"judge_id": "error", "score": 0.0}]

# ======================================================
# 🔹 Heurística simples
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
# 🔹 Avaliação via modelos LLM (com fallback dinâmico e seleção aleatória)
# ======================================================
async def llm_based_score(query: str, answer: str, use_rag: bool) -> float:
    """Executa julgamento dinâmico com fallback baseado em 2 juízes aleatórios."""
    try:
        judge_models = getattr(settings, "JUDGE_MODELS", [])
        if not judge_models:
            judge_models = ["openai/gpt-4o-mini"]

        # ✅ Seleciona dois juízes aleatórios (sem repetição)
        selected_judges = random.sample(judge_models, min(2, len(judge_models)))
        logger.info(f"[Judges] Selecionados {selected_judges} para avaliação desta resposta.")

        context = ""
        if use_rag:
            context = await get_rag_context(query, n_results=5)
            if context:
                logger.info("[Judges] Contexto RAG será usado na avaliação LLM.")

        rag_block = f"\nContexto adicional (via RAG):\n{context}\n" if context else ""
        prompt = (
            "Você é um avaliador de respostas de IA.\n"
            "Avalie a resposta abaixo considerando:\n"
            "1️⃣ Correção técnica e factual\n"
            "2️⃣ Clareza e coerência textual\n"
            "3️⃣ Relevância ao que foi perguntado\n\n"
            f"Pergunta: {query}\n\n"
            f"Resposta do modelo: {answer}\n\n"
            f"{rag_block}"
            "Responda SOMENTE com um número entre 0 e 10.\n"
            "FORMATO DE SAÍDA OBRIGATÓRIO:\n<nota>\n\n"
            "Por exemplo:\n8.7\n\n"
            "NÃO escreva mais nada além do número."
        ).strip()

        scores = []

        # ✅ Agora só percorremos os 2 juízes aleatórios
        for idx, model in enumerate(selected_judges, start=1):
            try:
                text, _ = call_model(model=model, prompt=prompt, temperature=0.2, max_tokens=32)
                numeric = extract_score(text)
                logger.info(f"[Judges] {model} → nota={numeric:.2f} (juiz {idx}/{len(selected_judges)})")

                # fallback se nota for 0
                if numeric == 0.0 and len(selected_judges) > 1:
                    next_model = selected_judges[(idx) % len(selected_judges)]
                    fb_text, _ = call_model(model=next_model, prompt=prompt, temperature=0.2, max_tokens=32)
                    fb_score = extract_score(fb_text)
                    log_fallback_event(query, answer, model, 0.0, next_model, fb_score, "zero_score")
                    logger.info(f"[Judges] Fallback {next_model} → nota={fb_score:.2f}")
                    numeric = fb_score

                scores.append(numeric)

            except Exception as e:
                logger.warning(f"[Judges] Falha no julgamento com {model}: {e}")
                try:
                    next_model = selected_judges[(idx) % len(selected_judges)]
                    fb_text, _ = call_model(model=next_model, prompt=prompt, temperature=0.2, max_tokens=32)
                    fb_score = extract_score(fb_text)
                    log_fallback_event(query, answer, model, None, next_model, fb_score, "exception")
                    logger.info(f"[Judges] Fallback {next_model} → nota={fb_score:.2f}")
                    scores.append(fb_score)
                except Exception as e2:
                    logger.error(f"[Judges] Fallback {next_model} também falhou: {e2}")
                    scores.append(0.0)

        if not scores:
            logger.warning("[Judges] Nenhum juiz retornou nota válida. Fallback 5.0.")
            return 0.5

        avg = sum(scores) / len(scores)
        return round(avg / 10.0, 3)

    except Exception as e:
        logger.error(f"[Judges] Erro no julgamento via LLM: {e}")
        return 0.0

# ======================================================
# 🔹 Auditoria de fallback
# ======================================================
def log_fallback_event(query: str, answer: str, model: str, score_before: float,
                       fallback_model: str, score_after: float, event_type: str):
    """Registra evento de fallback ou substituição de nota no banco."""
    try:
        with engine.begin() as conn:
            ddl_judge_logs = """
            CREATE TABLE IF NOT EXISTS judge_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                query TEXT,
                answer TEXT,
                judge_model VARCHAR(255),
                score_before FLOAT,
                fallback_model VARCHAR(255),
                score_after FLOAT,
                event_type VARCHAR(50)
            );
            """
            conn.execute(text(ddl_judge_logs))
            conn.execute(text("""
                INSERT INTO judge_logs
                (query, answer, judge_model, score_before, fallback_model, score_after, event_type)
                VALUES (:q, :a, :jm, :sb, :fb, :sa, :ev)
            """), {
                "q": query, "a": answer, "jm": model, "sb": score_before,
                "fb": fallback_model, "sa": score_after, "ev": event_type
            })
    except Exception as e:
        logger.error(f"[Judges] Erro ao registrar evento de auditoria: {e}")

# ======================================================
# 🔹 Extração de nota reforçada
# ======================================================
def extract_score(text: str) -> float:
    """Extrai número entre 0 e 10 do texto de saída do LLM."""
    try:
        if not text:
            return 0.0
        clean = text.strip().lower()
        match = re.search(r"(\d+(?:\.\d+)?)(?:\s*/\s*10)?", clean)
        if match:
            val = float(match.group(1))
            return max(0.0, min(val, 10.0))

        if any(w in clean for w in ["excelente", "ótima", "perfeita", "correta"]):
            return 9.0
        if any(w in clean for w in ["boa", "adequada", "razoável", "clara"]):
            return 7.0
        if any(w in clean for w in ["regular", "parcial", "mediana"]):
            return 5.0
        if any(w in clean for w in ["ruim", "fraca", "errada", "inadequada"]):
            return 3.0
        if any(w in clean for w in ["péssima", "horrível", "completamente errada"]):
            return 1.0
        return 0.0
    except Exception:
        return 0.0
