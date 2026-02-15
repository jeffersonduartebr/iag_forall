"""
rag_router.py (CORRIGIDO: Importação e Async)
----------------------------------------------------
Roteador FastAPI para gerenciamento dinâmico do RAG.
Permite upload de PDFs, Markdown e TXT, processando e
inserindo no ChromaDB via vectorstore unificado.
"""

import os
import uuid
import re
import logging
import fitz  # PyMuPDF
from fastapi import APIRouter, UploadFile, File, HTTPException

from pydantic import BaseModel
from typing import Dict, Any, Optional


from app.providers_async import call_model
from app.vectorstore import add_document
from app.settings_dynamic import settings

logger = logging.getLogger("rag_router")

router = APIRouter(prefix="/rag", tags=["RAG"])

# ============================================================
# ⚙️ Configurações
# ============================================================
COLLECTION_NAME = settings.get("RAG_COLLECTION_NAME", "knowledge_base")
SUMMARY_MODEL = settings.get("SUMMARY_MODEL", "ollama/phi4:latest")
CHUNK_SIZE = 800
OVERLAP = 100


# ============================================================
# ✂️ Utilitários
# ============================================================
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP):
    """Divide o texto em blocos de tamanho fixo com sobreposição."""
    text = re.sub(r"\s+", " ", text.strip())
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extrai texto bruto de um PDF recebido via upload."""
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            return " ".join(page.get_text("text") for page in doc)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Falha ao processar PDF: {e}")


async def summarize_text(text: str) -> tuple[str, str]:
    """Gera título e resumo com LLM."""
    try:
        prompt = f"""
Gere um TÍTULO curto (até 10 palavras) e um RESUMO de até 3 frases para o texto abaixo.

Texto:
{text[:1500]}

Responda no formato:
TÍTULO: ...
RESUMO: ...
"""
        # ✅ CORREÇÃO 2: Adicionado 'await' pois o novo call_model é async
        response, _ = await call_model(
            model=SUMMARY_MODEL,
            prompt=prompt,
            max_tokens=256,
            temperature=0.3
        )
        
        if not response:
            return "Documento Processado", "Resumo indisponível."

        title_match = re.search(r"(?i)t[ií]tulo[:：]\s*(.+)", response)
        summary_match = re.search(r"(?i)resumo[:：]\s*(.+)", response)
        
        title_text = title_match.group(1).strip() if title_match else "Documento sem título"
        summary_text = summary_match.group(1).strip() if summary_match else "Sem resumo gerado"
        
        return title_text, summary_text
    except Exception as e:
        logger.warning(f"[rag_router] Falha ao resumir texto: {e}")
        return "Documento genérico", "Resumo não disponível"


# ============================================================
# 🚀 Endpoint principal: /rag/add_doc
# ============================================================
@router.post("/add_doc")
async def add_doc(file: UploadFile = File(...)):
    """
    Recebe um arquivo (PDF, MD, TXT), gera embeddings e insere no ChromaDB.
    Retorna o título e o resumo automático.
    """
    try:
        filename = file.filename
        ext = os.path.splitext(filename)[-1].lower()

        if ext not in [".pdf", ".md", ".txt"]:
            raise HTTPException(
                status_code=400,
                detail="Formato inválido. Envie .pdf, .md ou .txt"
            )

        content = await file.read()
        text_content = ""

        if ext == ".pdf":
            text_content = extract_text_from_pdf(content)
        else:
            text_content = content.decode("utf-8", errors="ignore")

        if not text_content.strip():
            raise HTTPException(status_code=400, detail="Arquivo vazio ou ilegível.")

        fragments = chunk_text(text_content)
        logger.info(f"[rag_router] {filename}: {len(fragments)} fragmentos gerados.")

        # 🔹 Gera título e resumo
        title, summary = await summarize_text(" ".join(fragments[:3]))
        logger.info(f"[rag_router] 📘 {title}\n📝 {summary}")

        inserted_count = 0
        
        # 🔹 Processamento e Inserção
        for idx, frag in enumerate(fragments):
            try:
                doc_id = str(uuid.uuid4())
                
                success = await add_document(
                    modality="text",
                    doc_id=doc_id,
                    text=frag,
                    metadata={
                        "filename": filename, 
                        "chunk_index": idx,
                        "title": title,
                        "source": "upload_api"
                    }
                )
                
                if success:
                    inserted_count += 1
            
            except Exception as e:
                logger.error(f"[rag_router] Falha ao inserir fragmento {idx}: {e}")

        return {
            "file": filename,
            "title": title,
            "summary": summary,
            "fragments_total": len(fragments),
            "fragments_inserted": inserted_count,
            "collection": "text_embeddings"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[rag_router] Erro inesperado: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    


class IngestRequest(BaseModel):
    """Representa a responsabilidade principal desta classe."""
    text: str
    doc_id: str
    metadata: Dict[str, Any]
    collection_name: Optional[str] = None # Para suportar "Coleção por Curso"

@router.post("/ingest")
async def ingest_text(req: IngestRequest):
    """
    Endpoint para ingestão direta de texto (usado por serviços externos como o Auditor).
    """
    try:
        # Se um nome de coleção específico for passado (ex: course_101), 
        # o vectorstore deve ser capaz de lidar ou usamos metadata filtering.
        # Aqui, vamos injetar o collection_name nos metadados para garantir isolamento lógico
        if req.collection_name:
            req.metadata["target_collection"] = req.collection_name

        success = await add_document(
            modality="text", # Força texto
            doc_id=req.doc_id,
            text=req.text,
            metadata=req.metadata
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Falha ao inserir no Vectorstore")
            
        return {"status": "ok", "doc_id": req.doc_id}
        
    except Exception as e:
        logger.error(f"[RAG API] Erro na ingestão: {e}")
        raise HTTPException(status_code=500, detail=str(e))
