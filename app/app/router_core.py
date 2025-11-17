# -*- coding: utf-8 -*-
# router_core.py
from __future__ import annotations

import logging
import time
import asyncio
import threading
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
from app.bandits import select_model, bandit_update, compute_reward
from app.judges import judge_answer
from app.metrics_collector import update_model_metrics

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] router: %(message)s")

# ============================================================
# ⚙️ Conexão com o banco (para EMA e Query Log)
# ============================================================
DB_URL = f"mysql+pymysql://{settings.DB_USER}:{settings.DB_PASS}@{settings.DB_HOST}:3306/{settings.DB_NAME}"
engine = create_engine(DB_URL, pool_pre_ping=True, pool_recycle=3600)

CANDIDATE_MODELS = settings.CANDIDATE_MODELS_LIST
BLOCKED_PREFIXES = ("nomic-embed", "text-embedding", "bge-", "e5-")
EMA_HISTORY: dict = {}

# Quantos dias manter no histórico do query_log
LOG_RETENTION_DAYS = int(settings.get("QUERY_LOG_RETENTION_DAYS", 7))

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
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
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
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
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
            EMA_HISTORY = {row["model"]: dict(row) for row in result.mappings()}
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

# ============================================================
# 🧱 Tabela de Query Log (p/ NSGA-II e auditoria)
# ============================================================
def _init_query_log_table():
    ddl = """
    CREATE TABLE IF NOT EXISTS query_log (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        query_text TEXT,
        chosen_model VARCHAR(255) NOT NULL,
        answer LONGTEXT,
        quality FLOAT NOT NULL,
        latency_s FLOAT NOT NULL,
        cost_per_1k FLOAT NOT NULL,
        reward FLOAT DEFAULT 0.0,
        context_label VARCHAR(64) NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_created_at (created_at),
        INDEX idx_model (chosen_model)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """
    try:
        with engine.begin() as conn:
            conn.execute(text(ddl))
        logger.info("[QUERY_LOG] Tabela query_log criada/verificada.")
    except SQLAlchemyError as e:
        logger.warning(f"[QUERY_LOG] Falha ao criar tabela query_log: {e}")

def _log_query_event(
    query_text: str,
    model: str,
    answer: str,
    latency: float,
    cost: float,
    quality: float,
    reward: float,
    context_label: str | None
):
    """Registra execução real no query_log."""
    try:
        with engine.begin() as conn:
            conn.execute(
                text("""
                INSERT INTO query_log (query_text, chosen_model, answer, quality, latency_s, cost_per_1k, reward, context_label)
                VALUES (:q, :m, :a, :qual, :lat, :cst, :rew, :ctx);
                """),
                dict(q=query_text, m=model, a=answer, qual=quality, lat=latency, cst=cost, rew=reward, ctx=context_label)
            )
        logger.debug(
            f"[QUERY_LOG] Inserido: model={model}, lat={latency:.2f}s, "
            f"cost={cost:.4f}, qual={quality:.2f}, reward={reward:.3f}, ctx={context_label}"
        )
    except SQLAlchemyError as e:
        logger.warning(f"[QUERY_LOG] Falha ao inserir log de consulta: {e}")

# ============================================================
# 🧹 Limpeza diária
# ============================================================
def _cleanup_old_query_logs(retention_days: int = LOG_RETENTION_DAYS):
    while True:
        try:
            with engine.begin() as conn:
                result = conn.execute(
                    text("""
                    DELETE FROM query_log
                    WHERE created_at < (NOW() - INTERVAL :days DAY)
                    """),
                    {"days": retention_days},
                )
            deleted = result.rowcount if hasattr(result, "rowcount") else 0
            logger.info(f"[QUERY_LOG] Limpeza: {deleted} registros removidos (retendo {retention_days} dias).")
        except SQLAlchemyError as e:
            logger.warning(f"[QUERY_LOG] Falha ao limpar logs antigos: {e}")
        time.sleep(86400)  # 24h

# ============================================================
# 🏁 Inicialização
# ============================================================
_init_ema_tables()
_load_ema_from_db()
_init_query_log_table()
threading.Thread(target=_cleanup_old_query_logs, daemon=True).start()

# ============================================================
# 🚀 Função principal
# ============================================================
async def route_and_answer(
    query: str,
    system_prompt: str = "",
    use_rag: bool = False,
    max_tokens: int | None = None,
    temperature: float | None = None
):
    start_time = time.time()

    if max_tokens is None:
        max_tokens = settings.MAX_TOKENS_DEFAULT
    if temperature is None:
        temperature = settings.TEMPERATURE_DEFAULT

    # 0) Cache semântico (check)
    cached = await check_cache(query)
    if cached:
        logger.info(
            f"[router_core] ✅ Cache HIT — sim={cached.get('similarity', 0.0):.2f}. "
            f"Retornando resposta do cache."
        )
        return {
            "model": "semantic_cache",
            "answer": cached.get("text", ""),
            "latency_s": round(time.time() - start_time, 3),
            "cost_per_1k": 0.0,
            "quality": 9.5,
            "metadata": {"cached": True, "similarity": cached.get("similarity", 1.0)},
        }

    # 1) Modelos válidos
    candidate_models = settings.CANDIDATE_MODELS_LIST
    valid_models = [
        m for m in candidate_models
        if isinstance(m, str) and not any(m.startswith(prefix) for prefix in BLOCKED_PREFIXES)
    ]
    if not valid_models:
        raise RuntimeError("Nenhum modelo válido disponível para geração.")

    # 2) Seleção via Bandit
    chosen = select_model(valid_models, query)
    logger.info(f"[router_core] Modelo selecionado (via bandit): {chosen}")

    # 3) Garante modelo local (Ollama)
    if chosen.startswith("ollama/"):
        await asyncio.to_thread(_ensure_ollama_model, chosen.replace("ollama/", ""))

    # 4) Prompt final
    sp = (system_prompt or "").strip()
    prompt = f"{sp}\n\nUsuário: {query.strip()}".strip()

    # 5) Chamada ao LLM
    try:
        text_out, meta = call_model(
            model=chosen,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        logger.exception(f"[router_core] Erro ao chamar modelo '{chosen}': {e}")
        text_out, meta = f"[Erro ao processar com modelo {chosen}: {e}]", {"latency_s": 0.0, "cost_per_1k": 0.0}

    # 🔒 Normaliza SEMPRE para string (evita problemas no cache/judges/json)
    safe_answer = text_out if isinstance(text_out, str) else str(text_out)

    # 6) Avaliação de qualidade
    try:
        judge_scores = await judge_answer(
            query,
            safe_answer,
            use_rag and settings.get("ENABLE_RAG_FOR_JUDGES", "0") == "1"
        )
        valid_scores = [s["score"] for s in judge_scores if "score" in s]
        quality_score = float(np.mean(valid_scores)) if valid_scores else 0.0
        final_quality = round(quality_score * 10.0, 2)  # escala 0–10
    except Exception as e:
        logger.error(f"[router_core] Falha ao avaliar resposta: {e}")
        final_quality = 0.0

    # 7) Cache semântico (grava)
    try:
        await store_cache(query, safe_answer)
    except Exception as e:
        logger.warning(f"[router_core] Falha ao armazenar no cache: {e}")

    # 8) Resultado base
    latency_val = round(time.time() - start_time, 2)
    # custo: placeholder — se o provider retornar custo, pode sobrescrever
    cost_val = 0.001 if "ollama" in chosen else 0.15
    try:
        if isinstance(meta, dict):
            if "cost_per_1k" in meta and isinstance(meta["cost_per_1k"], (int, float)):
                cost_val = float(meta["cost_per_1k"])
            if "latency" in meta and isinstance(meta["latency"], (int, float)):
                latency_val = float(meta["latency"])
            elif "latency_s" in meta and isinstance(meta["latency_s"], (int, float)):
                latency_val = float(meta["latency_s"])
    except Exception:
        pass

    result = {
        "model": chosen,
        "answer": safe_answer,
        "latency_s": latency_val,
        "cost_per_1k": cost_val,
        "quality": final_quality,
        "metadata": meta,
    }

    # Métricas Prometheus
    try:
        ROUTER_MODEL_COST.labels(model=chosen).inc(result["cost_per_1k"])
        ROUTER_QUALITY_AVG.labels(model=chosen).set(result["quality"])
        ROUTER_COST_PER_QUERY.set(result["cost_per_1k"])
        if "ollama" in chosen:
            ROUTER_COST_SAVINGS.inc(max(0.0, 0.12 - result["cost_per_1k"]))
            ROUTER_LOCAL_USAGE_RATIO.set(1.0)
        else:
            ROUTER_LOCAL_USAGE_RATIO.set(0.0)
    except Exception:
        pass

    try:
        update_model_metrics(
            model_name=chosen,
            latency=result["latency_s"],
            quality=result["quality"],
            cost=result["cost_per_1k"]
        )
    except Exception as e:
        logger.warning(f"[router_core] Falha ao atualizar métricas dinâmicas: {e}")

    # 9) EMA persistente
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

    # 10) Reward, Bandit update e log
    try:
        reward_val = compute_reward(chosen, result["quality"], result["latency_s"], result["cost_per_1k"])
    except Exception:
        reward_val = 0.0

    try:
        # Atualiza bandit (inclui update incremental dos centróides, se implementado)
        bandit_update(model=chosen, query=query, reward=reward_val)
    except Exception as e:
        logger.warning(f"[router_core] Falha ao bandit_update: {e}")

    # Rótulo semântico (se disponível) para o log
    try:
        from app.bandits import _nearest_centroid_label  # leitura leve
        ctx_label = _nearest_centroid_label(query)
    except Exception:
        ctx_label = None

    # Grava no query_log
    try:
        _log_query_event(
            query_text=query,
            model=result["model"],
            answer=result["answer"],
            latency=result["latency_s"],
            cost=result["cost_per_1k"],
            quality=result["quality"],
            reward=reward_val,
            context_label=ctx_label
        )
    except Exception as e:
        logger.warning(f"[QUERY_LOG] Falha ao registrar evento: {e}")

    return result
# ============================================================
