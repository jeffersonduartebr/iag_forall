"""
populate_vectorstore.py
-------------------------------------------------------
Popula a base vetorial (ChromaDB) com textos, PDFs e
Markdowns para uso em RAG pelos juízes LLM e demais módulos.

Inclui:
✅ Geração de embeddings locais (via Ollama)
✅ Fragmentação automática de texto
✅ Geração de título e resumo contextual
✅ Logging detalhado
"""

import os
import re
import uuid
import logging
from pathlib import Path
from app.vectorstore import insert_embedding, get_or_create_collection
from app.embeddings import embed_text
from app.providers import call_model

import fitz  # PyMuPDF

# ============================================================
# ⚙️ CONFIGURAÇÃO DE LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("populate_vectorstore")

# ============================================================
# 🧠 PARÂMETROS GERAIS
# ============================================================
COLLECTION_NAME = os.getenv("RAG_COLLECTION", "knowledge_base")
DATA_DIR = os.getenv("RAG_DATA_DIR", "./data/rag_docs")
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "ollama/phi4")
CHUNK_SIZE = 1000
OVERLAP = 150

# ============================================================
# ✂️ UTILITÁRIOS DE DIVISÃO DE TEXTO
# ============================================================
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP):
    """Divide texto longo em fragmentos com sobreposição."""
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

# ============================================================
# 🧩 CARREGADORES DE DOCUMENTOS
# ============================================================
def load_text_file(path: Path) -> list[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        return chunk_text(text)
    except Exception as e:
        logger.error(f"[populate] Falha ao ler texto {path.name}: {e}")
        return []


def load_pdf_file(path: Path) -> list[str]:
    try:
        doc = fitz.open(path)
        text = " ".join(page.get_text("text") for page in doc)
        doc.close()
        return chunk_text(text)
    except Exception as e:
        logger.error(f"[populate] Falha ao processar PDF {path.name}: {e}")
        return []

# ============================================================
# 🧠 RESUMO AUTOMÁTICO (LLM)
# ============================================================
def summarize_text(text: str, model: str = SUMMARY_MODEL) -> tuple[str, str]:
    """
    Gera um título e resumo automático usando um modelo LLM.
    Retorna (título, resumo).
    """
    try:
        prompt = f"""
Você é um assistente especializado em síntese de documentos técnicos.
Gere um TÍTULO curto (até 10 palavras) e um RESUMO de até 3 frases para o texto abaixo.

Texto:
{text[:1500]}

Responda no formato:
TÍTULO: ...
RESUMO: ...
"""
        response, _ = call_model(
            model=model,
            prompt=prompt,
            max_tokens=256,
            temperature=0.3
        )

        # Extrai título e resumo da resposta
        title_match = re.search(r"(?i)t[ií]tulo[:：]\s*(.+)", response)
        summary_match = re.search(r"(?i)resumo[:：]\s*(.+)", response)
        title = title_match.group(1).strip() if title_match else "Documento sem título"
        summary = summary_match.group(1).strip() if summary_match else "Sem resumo gerado"
        return title, summary

    except Exception as e:
        logger.warning(f"[populate] Falha ao resumir texto: {e}")
        return "Documento genérico", "Resumo não disponível"

# ============================================================
# 🗂️ COLETA DE ARQUIVOS
# ============================================================
def gather_documents(folder: str) -> dict[str, list[str]]:
    """Retorna {arquivo: [fragmentos]} suportando .txt, .md, .pdf."""
    docs = {}
    if not os.path.exists(folder):
        logger.warning(f"[populate] Pasta {folder} não encontrada — criando...")
        os.makedirs(folder, exist_ok=True)
        return docs

    for file in os.listdir(folder):
        path = Path(os.path.join(folder, file))
        if not path.is_file():
            continue

        ext = path.suffix.lower()
        if ext in [".txt", ".md"]:
            fragments = load_text_file(path)
        elif ext == ".pdf":
            fragments = load_pdf_file(path)
        else:
            continue

        if fragments:
            docs[path.name] = fragments
            logger.info(f"[populate] {path.name}: {len(fragments)} fragmentos extraídos.")
    return docs

# ============================================================
# 🧩 POPULAÇÃO DO VECTORSTORE
# ============================================================
def populate_vectorstore():
    """Gera embeddings e insere documentos no ChromaDB."""
    logger.info(f"[populate] Iniciando população da coleção '{COLLECTION_NAME}'...")
    all_docs = gather_documents(DATA_DIR)
    if not all_docs:
        logger.warning("[populate] Nenhum documento encontrado para indexar.")
        return

    collection = get_or_create_collection(COLLECTION_NAME)
    total_fragments = 0

    for fname, fragments in all_docs.items():
        logger.info(f"[populate] Gerando resumo para '{fname}'...")
        title, summary = summarize_text(" ".join(fragments[:3]))
        logger.info(f"[populate] 📘 {title}\n📝 {summary}")

        for frag in fragments:
            try:
                emb = embed_text(frag)
                doc_id = str(uuid.uuid4())
                insert_embedding(COLLECTION_NAME, doc_id, frag, emb)
                total_fragments += 1
            except Exception as e:
                logger.error(f"[populate] Falha ao inserir fragmento de {fname}: {e}")

    logger.info(f"[populate] ✅ Concluído: {len(all_docs)} documentos, {total_fragments} fragmentos inseridos.")

# ============================================================
# 🚀 EXECUÇÃO DIRETA
# ============================================================
if __name__ == "__main__":
    logger.info("[populate] Script iniciado.")
    populate_vectorstore()
