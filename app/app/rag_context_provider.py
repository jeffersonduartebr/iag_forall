# app/rag_context_provider.py
import logging
import asyncio

# ✅ CORRIGIDO: Importa a função correta do vectorstore
from .vectorstore import query_embedding
# ✅ CORRIGIDO: Importa o gerador de embedding
from .embeddings import embed_text
# ✅ CORRIGIDO: Importa o settings centralizado
from .settings_dynamic import settings

logger = logging.getLogger(__name__)

# ✅ CORRIGIDO: Lê o nome da coleção do settings
#    (Usando o mesmo nome que 'rag_local.py' usa)
RAG_COLLECTION_NAME = settings.get("RAG_COLLECTION_NAME", "docs")


async def get_rag_context(query: str, k: int = 3) -> str:
    """
    (CORRIGIDO) Retorna os k trechos mais relevantes para o prompt do juiz LLM.
    """
    try:
        # 1. Gerar embedding da query
        embedding = await embed_text(query)
        if not embedding:
            logger.warning("[rag_context_provider] Falha ao gerar embedding para a query.")
            return ""

        # 2. Consultar o vectorstore
        results = await query_embedding(
            collection_name=RAG_COLLECTION_NAME,
            embedding=embedding,
            n_results=k
        )
        
        if not results:
            logger.info("[rag_context_provider] Nenhum documento encontrado no vectorstore.")
            return ""

        # 3. Processar resultados
        docs = results.get("documents", [[]])
        top_docs: list[str] = docs[0] if docs and isinstance(docs[0], list) else []
        
        if not top_docs:
            return ""

        context = "\n\n".join(top_docs)
        logger.info(f"[rag_context_provider] {len(top_docs)} trechos recuperados para o contexto.")
        return context
        
    except Exception as e:
        logger.error(f"[rag_context_provider] Falha ao buscar contexto RAG: {e}")
        return ""