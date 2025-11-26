# -*- coding: utf-8 -*-
"""
rag_local.py — RAG Multimodal Unificado (Com suporte Imagem -> Texto)
---------------------------------------------------------------------
Implementa a lógica de recuperação de contexto, incluindo a estratégia
de "Visual Query Generation" para RAG baseado em imagem.

Funcionalidades:
    ✔ Detecção automática de modalidade
    ✔ RAG Texto -> Texto
    ✔ RAG Imagem -> Texto (Gera descrição da imagem para buscar documentos)
    ✔ Prompt Augmentation
"""

from __future__ import annotations
import logging
import asyncio
import re
from typing import Optional, Dict, Any, List, Tuple

from .embeddings import (
    embed_text,
    embed_image,
    embed_multimodal,
)

from .vectorstore import (
    query_embedding,
    add_document,
    health_async,
)

# Importamos o call_model para gerar a descrição da imagem (Ponte Visual)
from .providers_async import call_model
from .settings_dynamic import settings

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] rag_local: %(message)s"
    )

# Modelo rápido para descrever imagens para busca (Moondream é ótimo aqui, ou Llama 3.2)
# Se não tiver configurado, usa o primeiro da lista de visão
VISION_HELPER_MODEL = "ollama/moondream:latest" 

# ================================================================
# 🔍 DETECÇÃO AUTOMÁTICA DE MODALIDADE
# ================================================================

def _auto_modality(requested: Optional[str], image_b64: Optional[str]) -> str:
    req = (requested or "text").lower().strip()

    if image_b64 and req == "multimodal":
        return "multimodal"
    if image_b64 and req in ("vision", "image"):
        return "vision"
    if image_b64 and req == "text":
        # Se tem imagem mas pediu texto, tratamos como visão para o RAG aproveitar a imagem
        return "vision"

    return "text"


# ================================================================
# 👁️ GERAÇÃO DE QUERY VISUAL (A PONTE)
# ================================================================

async def _generate_visual_search_query(image_b64: str) -> str:
    """
    Usa um VLM para descrever a imagem e criar uma string de busca textual.
    Isso permite usar a imagem para encontrar documentos de texto no Chroma.
    """
    try:
        # Tenta pegar um modelo da lista de candidatos se o helper não estiver fixo
        candidates = settings.CANDIDATE_VISION_MODELS_LIST
        model_to_use = list(candidates)[0] if candidates else "ollama/llava:7b"
        
        # Prompt focado em extração de keywords para busca
        prompt = (
            "Identifique o objeto principal, cenário ou problema nesta imagem. "
            "Gere uma única frase descritiva e técnica para ser usada como termo de busca em um banco de dados. "
            "Não use preâmbulos como 'A imagem mostra...'. Seja direto."
        )

        response, _ = await call_model(
            model=model_to_use,
            prompt=prompt,
            image_b64=image_b64,
            max_tokens=64,
            temperature=0.1
        )
        
        search_query = response.strip()
        logger.info(f"[RAG-Vision] Query gerada da imagem: '{search_query}'")
        return search_query

    except Exception as e:
        logger.warning(f"[RAG-Vision] Falha ao descrever imagem: {e}")
        return ""


# ================================================================
# 🧠 GERAÇÃO DE EMBEDDING ADEQUADO
# ================================================================

async def _compute_embedding(query: str, modality: str, image_b64: Optional[str]):
    """
    Gera o embedding adequado.
    
    Se for RAG 'vision' (Imagem -> Texto), nós NÃO geramos embedding da imagem direta
    (a menos que tenhamos um índice CLIP). Em vez disso, geramos a query visual (texto)
    e embedamos o texto.
    """
    try:
        # RAG Clássico (Texto -> Texto)
        if modality == "text":
            return await asyncio.to_thread(embed_text, query)

        # RAG Visual (Imagem -> Texto via Ponte Descritiva)
        if modality == "vision" and image_b64:
            # Se o usuário não mandou texto (só a imagem), geramos a descrição
            if not query or len(query) < 5:
                visual_query = await _generate_visual_search_query(image_b64)
                query_to_embed = visual_query if visual_query else "imagem genérica"
            else:
                # Se o usuário mandou texto junto (ex: "Como conserto isso?"), 
                # usamos o texto dele + uma descrição breve da imagem
                visual_desc = await _generate_visual_search_query(image_b64)
                query_to_embed = f"{query} {visual_desc}"
            
            # Embedamos o TEXTO resultante para buscar no banco de TEXTO
            return await asyncio.to_thread(embed_text, query_to_embed)

        # Multimodal (Conceito avançado de espaço latente compartilhado)
        # Só funciona se o vectorstore suportar embeddings multimodais nativos
        if modality == "multimodal":
            emb_dict = await asyncio.to_thread(embed_multimodal, query, image_b64)
            return emb_dict.get("multimodal") or emb_dict.get("text")

        return await asyncio.to_thread(embed_text, query)

    except Exception as e:
        logger.warning(f"[rag_local] Falha ao gerar embedding ({modality}): {e}")
        return None


# ================================================================
# 📚 RAG PRINCIPAL – Construção do Prompt Aumentado
# ================================================================

async def build_augmented_prompt(
    query: str,
    modality: str = "text",
    image_b64: Optional[str] = None,
    k: int = 3,
) -> str:
    query = (query or "").strip()
    
    # Se não tem nada, retorna vazio
    if not query and not image_b64:
        return ""

    rag_mode = _auto_modality(modality, image_b64)
    
    # Define em qual coleção buscar.
    # Estratégia: Para Image->Text, buscamos na coleção de TEXTO (onde estão os manuais)
    target_collection_modality = "text" if rag_mode == "vision" else rag_mode

    # Gera o embedding (se for visão, gera descrição -> embedding de texto)
    emb = await _compute_embedding(query, rag_mode, image_b64)
    
    if emb is None:
        return query

    try:
        res = await query_embedding(
            modality=target_collection_modality, # Busca na coleção de texto
            embedding=emb,
            n_results=k
        )
    except Exception as e:
        logger.warning(f"[rag_local] Erro na consulta RAG ({rag_mode}): {e}")
        return query

    docs = res.get("documents", [[]])
    top_docs: List[str] = docs[0] if docs and isinstance(docs[0], list) else []

    if not top_docs:
        logger.info("[rag_local] Nenhum documento relevante encontrado.")
        return query

    context = "\n\n".join(top_docs)
    logger.info(f"[rag_local] Recuperados {len(top_docs)} documentos para enriquecer o prompt.")

    # Prompt Engenharia para instruir o modelo a usar o contexto
    return (
        "INSTRUÇÃO DE CONTEXTO (RAG):\n"
        "Use as informações técnicas abaixo recuperadas do banco de dados para auxiliar na sua resposta.\n"
        "Se a imagem fornecida contradizer o texto, dê prioridade ao que você vê na imagem.\n"
        "------ CONTEXTO RECUPERADO ------\n"
        f"{context}\n"
        "---------------------------------\n"
        f"PERGUNTA DO USUÁRIO: {query}"
    )


# ================================================================
# 📝 ADICIONAR DOCUMENTO AO RAG
# ================================================================

async def add_document_local(
    doc_id: str,
    text: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    modality: str = "text",
    image_b64: Optional[str] = None,
) -> bool:
    try:
        await add_document(
            modality=modality,
            doc_id=doc_id,
            text=text,
            image_b64=image_b64,
            metadata=metadata,
        )
        return True
    except Exception as e:
        logger.error(f"[rag_local] Falha ao adicionar documento {doc_id}: {e}")
        return False


# ================================================================
# 🩺 HEALTHCHECK
# ================================================================

async def health() -> Dict[str, Any]:
    try:
        chroma_ok = await health_async()
    except Exception:
        chroma_ok = False

    return {
        "vectorstore": chroma_ok,
        "status": "ok" if chroma_ok else "fail"
    }