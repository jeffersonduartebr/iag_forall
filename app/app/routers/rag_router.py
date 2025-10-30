"""
rag_router.py
----------------------------------------------------
Roteador FastAPI para gerenciamento dinâmico do RAG.
Permite upload de PDFs, Markdown e TXT, gerando embeddings
e armazenando no ChromaDB.
"""

import os
import uuid
import re
import logging
import fitz  # PyMuPDF
from fastapi import APIRouter, UploadFile, File, HTTPException

# ✅ Importes corrigidos e compatíveis com vectorstore.py
from app.vectorstore import insert_embedding, _get_or_create_collection_sync as get_or_create_collection
from app.embeddings import embed_text
from app.providers import call_model

logger = logging.getLogger("rag_router")

router = APIRouter(prefix="/rag", tags=["RAG"])

# ============================================================
# ⚙️ Configurações
# ============================================================
COLLECTION_NAME = os.getenv("RAG_COLLECTION", "knowledge_base")
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "ollama/phi4")
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
        response, _ = call_model(
            model=SUMMARY_MODEL,
            prompt=prompt,
            max_tokens=256,
            temperature=0.3
        )
        title = re.search(r"(?i)t[ií]tulo[:：]\s*(.+)", response)
        summary = re.search(r"(?i)resumo[:：]\s*(.+)", response)
        title_text = title.group(1).strip() if title else "Documento sem título"
        summary_text = summary.group(1).strip() if summary else "Sem resumo gerado"
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
        text = ""

        if ext == ".pdf":
            text = extract_text_from_pdf(content)
        else:
            text = content.decode("utf-8", errors="ignore")

        if not text.strip():
            raise HTTPException(status_code=400, detail="Arquivo vazio ou ilegível.")

        fragments = chunk_text(text)
        logger.info(f"[rag_router] {filename}: {len(fragments)} fragmentos gerados.")

        # 🔹 Gera título e resumo
        title, summary = await summarize_text(" ".join(fragments[:3]))
        logger.info(f"[rag_router] 📘 {title}\n📝 {summary}")

        # 🔹 Cria ou obtém a coleção de embeddings
        get_or_create_collection(COLLECTION_NAME)

        inserted = 0
        for frag in fragments:
            try:
                emb = await embed_text(frag)
                doc_id = str(uuid.uuid4())
                await insert_embedding(COLLECTION_NAME, doc_id, frag, emb, metadata={"filename": filename, "chunk": idx})

                inserted += 1
            except Exception as e:
                logger.error(f"[rag_router] Falha ao inserir fragmento: {e}")

        return {
            "file": filename,
            "title": title,
            "summary": summary,
            "fragments": inserted,
            "collection": COLLECTION_NAME
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[rag_router] Erro inesperado: {e}")
        raise HTTPException(status_code=500, detail=str(e))
