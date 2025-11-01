# router_core.py
import logging
import time
import asyncio
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from .providers import call_model, _ensure_ollama_model
from app.semantic_cache import check_cache, store_cache
from app.observability import (
    ROUTER_MODEL_COST, ROUTER_QUALITY_AVG, ROUTER_COST_SAVINGS,
    ROUTER_LOCAL_USAGE_RATIO, ROUTER_COST_PER_QUERY, ROUTER_HISTORY_ENTRIES
)
from .settings_dynamic import settings
from app.bandits import select_model
from app.judges import judge_answer
from app.metrics_collector import update_model_metrics

logger = logging.getLogger(__name__)

# ============================================================
# ⚙️ Conexão com o banco (para EMA persistente)
# ============================================================
DB_URL = f"mysql+pymysql://{settings.DB_USER}:{settings.DB_PASS}@{settings.DB_HOST}:3306/{settings.DB_NAME}"
engine = create_engine(DB_URL, pool_pre_ping=True, pool_recycle=3600)

CANDIDATE_MODELS = settings.CANDIDATE_MODELS_LIST
BLOCKED_PREFIXES = ("nomic-embed", "text-embedding", "bge-", "e5-")
EMA_HISTORY = {}

# ============================================================
# 🧱 Tabelas de EMA
# ============================================================
def _init_ema_tables():
    ddl_main = """
    CREATE TABLE IF NOT EXISTS ema_history (
        model VARCHAR(255) PRIMARY KEY,
        ema_latency FLOAT NOT NULL,
        ema_quality FLOAT NOT NULL,
        ema_cost FLOAT NOT NULL,
        updates INT DEFAULT 0,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP 
            ON UPDATE CURRENT_TIMESTAMP
    );
    """
    ddl_log = """
    CREATE TABLE IF NOT EXISTS ema_history_log (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        model VARCHAR(255) NOT NULL,
        ema_latency FLOAT NOT NULL,
        ema_quality FLOAT NOT NULL,
        ema_cost FLOAT NOT NULL,
        update_num INT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_model_created_at (model, created_at)
    );
    """
    try:
        with engine.begin() as conn:
            conn.execute(text(ddl_main))
            conn.execute(text(ddl_log))
        logger.info("[EMA] Tabelas EMA criadas/verificadas.")
    except SQLAlchemyError as e:
        logger.warning(f"[EMA] Falha ao criar tabelas: {e}")

def _load_ema_from_db():
    global EMA_HISTORY
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT * FROM ema_history"))
            EMA_HISTORY = {row["model"]: dict(row._mapping) for row in result}
        logger.info(f"[EMA] Histórico EMA carregado ({len(EMA_HISTORY)} modelos).")
    except SQLAlchemyError as e:
        logger.warning(f"[EMA] Falha ao carregar histórico EMA: {e}")

def _persist_ema_to_db(model, ema):
    try:
        with engine.begin() as conn:
            conn.execute(
                text("""
                INSERT INTO ema_history (model, ema_latency, ema_quality, ema_cost, updates)
                VALUES (:model, :lat, :qual, :cost, :upd)
                ON DUPLICATE KEY UPDATE
                    ema_latency = :lat,
                    ema_quality = :qual,
                    ema_cost = :cost,
                    updates = :upd,
                    last_updated = CURRENT_TIMESTAMP;
                """),
                dict(model=model, lat=ema["ema_latency"], qual=ema["ema_quality"],
                     cost=ema["ema_cost"], upd=ema["updates"])
            )
            conn.execute(
                text("""
                INSERT INTO ema_history_log (model, ema_latency, ema_quality, ema_cost, update_num)
                VALUES (:model, :lat, :qual, :cost, :upd);
                """),
                dict(model=model, lat=ema["ema_latency"], qual=ema["ema_quality"],
                     cost=ema["ema_cost"], upd=ema["updates"])
            )
    except SQLAlchemyError as e:
        logger.warning(f"[EMA] Falha ao persistir EMA de {model}: {e}")

_init_ema_tables()
_load_ema_from_db()

# ============================================================
# 🚀 Função principal de roteamento + cache + EMA persistente
# ============================================================
async def route_and_answer(
    query: str, 
    system_prompt: str = "", 
    use_rag: bool = False,
    max_tokens: int = None,
    temperature: float = None
):
    start_time = time.time()

    if max_tokens is None:
        max_tokens = settings.MAX_TOKENS_DEFAULT
    if temperature is None:
        temperature = settings.TEMPERATURE_DEFAULT

    # 0️⃣ Cache semântico
    cached = await check_cache(query)
    if cached:
        logger.info(
            f"[router_core] ✅ Cache HIT — sim={cached['similarity']:.2f}. "
            f"Retornando resposta do cache."
        )
        return {
            "model": "semantic_cache",
            "answer": cached["text"],
            "latency_s": round(time.time() - start_time, 3),
            "cost_per_1k": 0.0,
            "quality": 9.5,
            "metadata": {"cached": True, "similarity": cached["similarity"]},
        }

    # 1️⃣ Modelos válidos (dinâmico)
    candidate_models = settings.CANDIDATE_MODELS_LIST
    valid_models = [
        m for m in candidate_models
        if isinstance(m, str) and not any(m.startswith(prefix) for prefix in BLOCKED_PREFIXES)
    ]
    if not valid_models:
        raise RuntimeError("Nenhum modelo válido disponível para geração.")

    # 2️⃣ Seleção via Bandit/NSGA (externo)
    chosen = select_model(valid_models, query)
    logger.info(f"[router_core] Modelo selecionado (via bandit): {chosen}")

    # 3️⃣ Garante modelo local (Ollama)
    if chosen.startswith("ollama/"):
        await asyncio.to_thread(_ensure_ollama_model, chosen.replace("ollama/", ""))

    # 4️⃣ Prompt final
    sp = (system_prompt or "").strip()
    prompt = f"{sp}\n\nUsuário: {query.strip()}".strip()

    # 5️⃣ Chama o modelo LLM
    try:
        text, meta = call_model(
            model=chosen,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        logger.exception(f"[router_core] Erro ao chamar modelo '{chosen}': {e}")
        text, meta = f"[Erro ao processar com modelo {chosen}: {e}]", {"latency_s": 0.0}

    # 6️⃣ Avaliação de qualidade
    try:
        judge_scores = await judge_answer(query, text, use_rag and settings.ENABLE_RAG_FOR_JUDGES)
        valid_scores = [s["score"] for s in judge_scores if "score" in s]
        quality_score = float(np.mean(valid_scores)) if valid_scores else 0.0
        final_quality = round(quality_score * 10.0, 2)
    except Exception as e:
        logger.error(f"[router_core] Falha ao avaliar resposta: {e}")
        final_quality = 0.0

    # 7️⃣ Cache semântico (grava)
    try:
        await store_cache(query, text)
    except Exception as e:
        logger.warning(f"[router_core] Falha ao armazenar no cache: {e}")

    # 8️⃣ Resultado
    result = {
        "model": chosen,
        "answer": text,
        "latency_s": round(time.time() - start_time, 2),
        "cost_per_1k": 0.001 if "ollama" in chosen else 0.15,  # placeholder
        "quality": final_quality,
        "metadata": meta,
    }

    # Métricas Prometheus
    ROUTER_MODEL_COST.labels(model=chosen).inc(result["cost_per_1k"])
    ROUTER_QUALITY_AVG.labels(model=chosen).set(result["quality"])
    ROUTER_COST_PER_QUERY.set(result["cost_per_1k"])
    if "ollama" in chosen:
        ROUTER_COST_SAVINGS.inc(max(0.0, 0.12 - result["cost_per_1k"]))
        ROUTER_LOCAL_USAGE_RATIO.set(1.0)
    else:
        ROUTER_LOCAL_USAGE_RATIO.set(0.0)

    try:
        update_model_metrics(
            model_name=chosen,
            latency=result["latency_s"],
            quality=result["quality"],
            cost=result["cost_per_1k"]
        )
    except Exception as e:
        logger.warning(f"[router_core] Falha ao atualizar métricas dinâmicas: {e}")

    # 9️⃣ EMA persistente
    try:
        global EMA_HISTORY
        alpha = 0.3
        model_key = chosen

        if model_key not in EMA_HISTORY:
            EMA_HISTORY[model_key] = {
                "ema_latency": result["latency_s"],
                "ema_quality": result["quality"],
                "ema_cost": result["cost_per_1k"],
                "updates": 1,
            }
        else:
            prev = EMA_HISTORY[model_key]
            EMA_HISTORY[model_key]["ema_latency"] = alpha * result["latency_s"] + (1 - alpha) * prev["ema_latency"]
            EMA_HISTORY[model_key]["ema_quality"] = alpha * result["quality"] + (1 - alpha) * prev["ema_quality"]
            EMA_HISTORY[model_key]["ema_cost"] = alpha * result["cost_per_1k"] + (1 - alpha) * prev["ema_cost"]
            EMA_HISTORY[model_key]["updates"] = prev.get("updates", 0) + 1

        _persist_ema_to_db(model_key, EMA_HISTORY[model_key])

        ROUTER_HISTORY_ENTRIES.set(len(EMA_HISTORY))
        ema = EMA_HISTORY[model_key]
        logger.info(
            f"[EMA] {model_key} → "
            f"EMA(lat={ema['ema_latency']:.2f}s, "
            f"qual={ema['ema_quality']:.2f}, "
            f"custo={ema['ema_cost']:.4f}) "
            f"({ema['updates']} updates)"
        )
    except Exception as e:
        logger.warning(f"[EMA] Falha ao atualizar histórico: {e}")

    return result
