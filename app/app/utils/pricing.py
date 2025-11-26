# -*- coding: utf-8 -*-
import time
import logging
from sqlalchemy import create_engine, text
from app.settings_dynamic import settings

logger = logging.getLogger("pricing")

DB_URL = f"mysql+pymysql://{settings.DB_USER}:{settings.DB_PASS}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
engine = create_engine(DB_URL, pool_pre_ping=True, pool_recycle=3600)

_PRICING_CACHE = {}
_LAST_UPDATE = 0
CACHE_TTL = 300 

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
    except Exception:
        pass

def get_model_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calcula custo total em USD."""
    if time.time() - _LAST_UPDATE > CACHE_TTL:
        _refresh_pricing()

    # Tenta encontrar o modelo exato ou o nome limpo
    pricing = _PRICING_CACHE.get(model)
    if not pricing:
        clean_model = model.split("/", 1)[1] if "/" in model else model
        pricing = _PRICING_CACHE.get(clean_model)

    if not pricing:
        # Defaults de segurança baseados nos prints (Fallback)
        m_lower = model.lower()
        if "gpt-5" in m_lower and "mini" in m_lower: pricing = {"in": 0.00025, "out": 0.002}
        elif "gpt-5" in m_lower: pricing = {"in": 0.00125, "out": 0.01}
        elif "gpt-4.1-mini" in m_lower: pricing = {"in": 0.0004, "out": 0.0016}
        elif "gpt-4o-mini" in m_lower: pricing = {"in": 0.00015, "out": 0.0006}
        elif "gpt-4o" in m_lower: pricing = {"in": 0.0025, "out": 0.01}
        
        elif "gemini-2.5-flash" in m_lower: pricing = {"in": 0.0003, "out": 0.0025}
        elif "gemini-2.5" in m_lower: pricing = {"in": 0.00125, "out": 0.01}
        
        elif "haiku" in m_lower: pricing = {"in": 0.001, "out": 0.005}
        elif "sonnet" in m_lower: pricing = {"in": 0.003, "out": 0.015}
        elif "opus" in m_lower: pricing = {"in": 0.005, "out": 0.025}
        
        else: pricing = {"in": 0.0, "out": 0.0} # Ollama/Local

    cost_in = (input_tokens / 1000) * pricing["in"]
    cost_out = (output_tokens / 1000) * pricing["out"]
    
    return cost_in + cost_out