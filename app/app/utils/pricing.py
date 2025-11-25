# -*- coding: utf-8 -*-
import time
import logging
from sqlalchemy import create_engine, text
from app.settings_dynamic import settings

logger = logging.getLogger("pricing")

DB_URL = f"mysql+pymysql://{settings.DB_USER}:{settings.DB_PASS}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
engine = create_engine(DB_URL, pool_pre_ping=True, pool_recycle=3600)

# Cache simples em memória: { "gpt-4o": {"in": 0.0025, "out": 0.01} }
_PRICING_CACHE = {}
_LAST_UPDATE = 0
CACHE_TTL = 300  # 5 minutos

def _refresh_pricing():
    global _PRICING_CACHE, _LAST_UPDATE
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT model, cost_input_1k, cost_output_1k FROM model_pricing")).fetchall()
            new_cache = {}
            for r in rows:
                new_cache[r[0]] = {"in": float(r[1]), "out": float(r[2])}
            
            _PRICING_CACHE = new_cache
            _LAST_UPDATE = time.time()
            # logger.info(f"Preços atualizados: {len(new_cache)} modelos.")
    except Exception as e:
        logger.error(f"Falha ao atualizar preços: {e}")

def get_model_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calcula custo total em USD."""
    if time.time() - _LAST_UPDATE > CACHE_TTL:
        _refresh_pricing()

    # Remove prefixos (ex: openai/gpt-4o -> gpt-4o)
    clean_model = model.split("/", 1)[1] if "/" in model else model
    
    # Fallbacks hardcoded se não tiver no banco
    pricing = _PRICING_CACHE.get(clean_model)
    
    if not pricing:
        # Defaults de segurança (valores aproximados de mercado)
        if "gpt-4" in clean_model: pricing = {"in": 0.0025, "out": 0.01}
        elif "gpt-3.5" in clean_model or "mini" in clean_model: pricing = {"in": 0.00015, "out": 0.0006}
        elif "claude-3-5" in clean_model: pricing = {"in": 0.003, "out": 0.015}
        elif "gemini" in clean_model: pricing = {"in": 0.000075, "out": 0.0003}
        else: pricing = {"in": 0.0, "out": 0.0} # Ollama/Local

    cost_in = (input_tokens / 1000) * pricing["in"]
    cost_out = (output_tokens / 1000) * pricing["out"]
    
    return cost_in + cost_out