# -*- coding: utf-8 -*-
"""
semantic_cache.py — Cache Semântico via ChromaDB (Rápido)
---------------------------------------------------------
Substitui a busca linear no MariaDB por busca ANN no ChromaDB.
"""

from __future__ import annotations
import hashlib
import logging
import asyncio
import time
from typing import Optional, Dict, Any
from functools import lru_cache

from .settings_dynamic import settings
from .embeddings import embed_text, embed_image, embed_multimodal
from .vectorstore import query_embedding, add_document

logger = logging.getLogger(__name__)

# Limiar de similaridade para cache hit (0.0 a 1.0)
# No Chroma (Cosine Distance), distance < (1 - threshold)
CACHE_THRESHOLD = float(settings.get("CACHE_THRESHOLD", 0.92))

def _compute_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def _normalize_modality(mod: str) -> str:
    mod = mod.lower().strip()
    if mod in ("vision", "image"): return "vision"
    if mod == "multimodal": return "multimodal"
    return "text"

# Cache em memória RAM (L1) para evitar hit no Chroma (L2) para queries idênticas
@lru_cache(maxsize=1024)
def _lru_get(key: str): return None
def _lru_store(key: str, val: Any): 
    try: _lru_get.cache_clear() # Limpeza simples se encher
    except: pass

async def _make_embedding(query: str, modality: str, image_b64: Optional[str]):
    # Reutiliza lógica de embeddings
    try:
        if modality == "text":
            return await asyncio.to_thread(embed_text, query)
        if modality == "vision":
            if not image_b64: return await asyncio.to_thread(embed_text, query)
            return await asyncio.to_thread(embed_image, image_b64)
        emb = await asyncio.to_thread(embed_multimodal, query, image_b64)
        return emb.get("multimodal") or emb.get("text")
    except Exception as e:
        logger.warning(f"[semantic_cache] Embed fail: {e}")
        return None

async def check_cache(query: str, modality: str="text", image_b64: str=None) -> Optional[Dict]:
    """
    Verifica se existe uma resposta similar no ChromaDB.
    """
    modality = _normalize_modality(modality)
    
    # 1. L1 Cache (Exact Match em RAM)
    # Hash da query + imagem para chave rápida
    img_hash = hashlib.sha256(image_b64.encode()).hexdigest() if image_b64 else "no_img"
    full_hash = f"{modality}:{_compute_sha256(query)}:{img_hash}"

    if cached := _lru_get(full_hash):
        logger.info("[semantic_cache] L1 RAM Hit")
        return cached

    # 2. Gera Embedding
    q_emb = await _make_embedding(query, modality, image_b64)
    if q_emb is None: return None

    # 3. L2 Cache (Semantic Search no Chroma)
    try:
        # Busca na coleção de cache ("cache" é mapeado para semantic_cache_v2 no vectorstore.py)
        results = await query_embedding(
            modality="cache", 
            embedding=q_emb, 
            n_results=1
        )
        
        if not results or not results.get("documents"):
            return None

        # Chroma retorna 'distances' (Cosine Distance). 
        # Similarity = 1 - Distance.
        distance = results["distances"][0][0]
        similarity = 1.0 - distance
        
        if similarity >= CACHE_THRESHOLD:
            # O texto do documento é a Query original
            # A resposta está no metadata
            meta = results["metadatas"][0][0]
            answer_text = meta.get("answer_payload", "")
            
            if not answer_text:
                return None

            # Reconstrói objeto de resposta
            res = {
                "text": answer_text, 
                "similarity": float(similarity),
                "model_used": meta.get("model_used", "unknown"),
                "image_output_b64": meta.get("image_output_b64", None) # Se houver
            }
            
            # Atualiza L1
            _lru_store(full_hash, res)
            return res
            
    except Exception as e:
        logger.warning(f"[semantic_cache] Chroma lookup fail: {e}")
        return None

    return None

async def store_cache(query: str, answer: str, modality: str="text", image_b64: str=None, model_used: str=None):
    """
    Armazena uma resposta de alta qualidade no ChromaDB.
    """
    modality = _normalize_modality(modality)
    
    # ID único para o documento de cache
    doc_id = f"cache_{_compute_sha256(query)}_{int(time.time())}"
    
    # Metadados para recuperação
    meta = {
        "model_used": model_used or "unknown",
        "original_query": query[:100], # Snippet para debug
        "timestamp": int(time.time()),
        "modality": modality,
        "answer_payload": answer # Armazena a resposta no metadata
    }
    
    try:
        # Estratégia:
        # Documento (Text) = PERGUNTA (para match semântico com a nova pergunta)
        # Metadata = RESPOSTA (o que queremos devolver)
        
        await add_document(
            modality="cache", # Vai para semantic_cache_v2
            doc_id=doc_id,
            text=query, # O texto indexado é a pergunta
            metadata=meta
        )
        
    except Exception as e:
        logger.warning(f"[semantic_cache] Store fail: {e}")