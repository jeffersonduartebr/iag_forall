# -*- coding: utf-8 -*-
"""
populate_vectorstore.py (Versão Final: OCR + Deduplicação + Metadados)
----------------------------------------------------------------------
Script de ingestão de documentos para o RAG.

Funcionalidades:
1. Lê arquivos de /app/data/rag_docs (.pdf, .txt, .md).
2. Detecta PDF escaneado e aplica OCR (Tesseract) automaticamente.
3. Divide texto em chunks.
4. Gera ID Determinístico (Hash) para evitar duplicatas.
5. Gera Título e Resumo via LLM.
6. Insere no ChromaDB (Vectorstore).
"""

import os
import re
import uuid
import hashlib
import logging
import asyncio
from pathlib import Path
from typing import List, Tuple, Optional

# Bibliotecas de PDF
import fitz  # PyMuPDF

# Bibliotecas de OCR (Tenta importar, se falhar, segue sem OCR)
try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# --- IMPORTS DO PROJETO ---
# Usamos imports absolutos (app.*) para funcionar dentro do container
from app.app.vectorstore import add_document, get_or_create_collection_async
from app.app.embeddings import embed_text
from app.app.providers_async import call_model
from app.app.settings_dynamic import settings

# ============================================================
# ⚙️ CONFIGURAÇÃO
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] populate: %(message)s",
)
logger = logging.getLogger("populate_vectorstore")

COLLECTION_NAME = settings.get("RAG_COLLECTION_NAME", "knowledge_base")
DATA_DIR = settings.get("RAG_DATA_DIR", "/app/data/rag_docs")
SUMMARY_MODEL = settings.get("SUMMARY_MODEL", "ollama/phi4:latest")

# Configuração de Chunking (Fixo por enquanto, mais seguro e rápido)
CHUNK_SIZE = 1000
OVERLAP = 150


# ============================================================
# 🔑 GERAÇÃO DE ID (DEDUPLICAÇÃO)
# ============================================================
def generate_deterministic_id(filename: str, chunk_index: int, content: str) -> str:
    """
    Gera um Hash SHA-256 único para o fragmento.
    Garante que o mesmo texto no mesmo arquivo gere sempre o mesmo ID.
    Isso evita duplicatas no ChromaDB se o script rodar várias vezes.
    """
    raw_key = f"{filename}::{chunk_index}::{content}"
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


# ============================================================
# ✂️ UTILITÁRIOS DE TEXTO
# ============================================================
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    """Divide texto longo em fragmentos com sobreposição."""
    # Normaliza espaços
    text = re.sub(r"\s+", " ", text.strip())
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ============================================================
# 👁️ LÓGICA DE OCR (PDF ESCANEADO)
# ============================================================
def _perform_ocr(path: Path) -> str:
    """Converte páginas do PDF em imagens e roda Tesseract."""
    logger.info(f"[OCR] Iniciando reconhecimento ótico em: {path.name}...")
    text_accum = []
    try:
        # Converte PDF para lista de imagens (thread_count=4 para agilizar)
        images = convert_from_path(str(path), thread_count=4)
        
        for i, image in enumerate(images):
            # Extrai texto da imagem (PT + EN)
            page_text = pytesseract.image_to_string(image, lang='por+eng')
            text_accum.append(page_text)
            if (i + 1) % 5 == 0:
                logger.info(f"[OCR] Processadas {i + 1} páginas...")
            
        full_text = " ".join(text_accum)
        logger.info(f"[OCR] Concluído. Extraídos {len(full_text)} caracteres.")
        return full_text
    except Exception as e:
        logger.error(f"[OCR] Erro crítico ao processar {path.name}: {e}")
        return ""


# ============================================================
# 📄 CARREGADORES DE ARQUIVOS
# ============================================================
def load_text_file(path: Path) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        return chunk_text(text)
    except Exception as e:
        logger.error(f"Falha ao ler texto {path.name}: {e}")
        return []


def load_pdf_file(path: Path) -> List[str]:
    """
    Lê PDF. Tenta extração direta (rápida). 
    Se falhar (PDF escaneado/imagem), usa OCR (lento).
    """
    full_text = ""
    try:
        # 1. Tentativa Rápida (Texto digital)
        doc = fitz.open(path)
        full_text = " ".join(page.get_text("text") for page in doc)
        doc.close()
        
        # Heurística: Se tiver menos de 100 caracteres no total, provavelmente é escaneado
        if len(full_text.strip()) < 100:
            if OCR_AVAILABLE:
                logger.warning(f"Texto insuficiente ({len(full_text)} chars) em {path.name}. Tentando OCR...")
                ocr_text = _perform_ocr(path)
                # Só substitui se o OCR achou mais coisa que o método digital
                if len(ocr_text) > len(full_text):
                    full_text = ocr_text
            else:
                logger.warning(f"PDF escaneado detectado em {path.name}, mas OCR não está instalado.")

        return chunk_text(full_text)

    except Exception as e:
        logger.error(f"Falha ao processar PDF {path.name}: {e}")
        return []


def gather_documents(folder: str) -> dict[str, List[str]]:
    """Varre a pasta e retorna {nome_arquivo: [chunks]}."""
    docs = {}
    path_obj = Path(folder)
    
    if not path_obj.exists():
        logger.warning(f"Pasta {folder} não encontrada — criando...")
        try:
            path_obj.mkdir(parents=True, exist_ok=True)
        except: pass
        return docs

    for file_path in path_obj.glob("*.*"):
        if not file_path.is_file():
            continue

        ext = file_path.suffix.lower()
        fragments = []

        if ext in [".txt", ".md"]:
            fragments = load_text_file(file_path)
        elif ext == ".pdf":
            fragments = load_pdf_file(file_path)
        else:
            continue

        if fragments:
            docs[file_path.name] = fragments
            logger.info(f"Carregado: {file_path.name} ({len(fragments)} chunks)")
    
    return docs


# ============================================================
# 🧠 RESUMO AUTOMÁTICO (LLM Async)
# ============================================================
async def summarize_text_async(text: str, model: str = SUMMARY_MODEL) -> Tuple[str, str]:
    """
    Gera título e resumo usando o LLM via providers_async.
    """
    try:
        prompt = f"""
Você é um assistente de arquivologia.
Analise o fragmento de texto abaixo e gere:
1. Um TÍTULO curto e descritivo (máx 10 palavras).
2. Um RESUMO conciso do conteúdo (máx 3 frases).

Texto:
{text[:2000]}

Responda estritamente no formato:
TÍTULO: ...
RESUMO: ...
"""
        # Chama o provider (agora async)
        response, _ = await call_model(
            model=model,
            prompt=prompt,
            max_tokens=256,
            temperature=0.3
        )

        # Extração via Regex
        title_match = re.search(r"(?i)t[ií]tulo[:：]\s*(.+)", response)
        summary_match = re.search(r"(?i)resumo[:：]\s*(.+)", response)
        
        title = title_match.group(1).strip() if title_match else "Documento sem título"
        summary = summary_match.group(1).strip() if summary_match else "Resumo não disponível"
        
        return title, summary

    except Exception as e:
        logger.warning(f"Falha ao resumir texto: {e}")
        return "Documento Genérico", "Sem resumo"


# ============================================================
# 🚀 FLUXO PRINCIPAL
# ============================================================
async def populate_vectorstore():
    logger.info(f"🚀 Iniciando população da coleção '{COLLECTION_NAME}'...")
    
    # 1. Ler arquivos do disco
    all_docs = gather_documents(DATA_DIR)
    if not all_docs:
        logger.warning("⚠️ Nenhum documento encontrado em /data/rag_docs.")
        return

    # 2. Garantir que a coleção existe
    await get_or_create_collection_async(COLLECTION_NAME)

    total_inserted = 0
    
    for fname, fragments in all_docs.items():
        logger.info(f"Processando '{fname}'...")
        
        # Gera resumo com base nos primeiros 3 chunks (contexto inicial)
        context_preview = " ".join(fragments[:3])
        title, summary = await summarize_text_async(context_preview)
        
        logger.info(f"   📘 Título: {title}")

        # Insere cada fragmento
        for idx, frag in enumerate(fragments):
            try:
                # GERA ID ÚNICO BASEADO NO CONTEÚDO
                doc_id = generate_deterministic_id(fname, idx, frag)

                meta = {
                    "source": fname,
                    "chunk_index": idx,
                    "total_chunks": len(fragments),
                    "title": title,
                    "summary_snippet": summary[:200]
                }
                
                # add_document já calcula o embedding internamente via threadpool
                # Como o ID é determinístico, o ChromaDB automaticamente fará "upsert" (atualizar ou ignorar)
                # evitando duplicatas reais no índice.
                success = await add_document(
                    modality="text",
                    doc_id=doc_id,
                    text=frag,
                    metadata=meta
                )
                
                if success:
                    total_inserted += 1
                    
            except Exception as e:
                logger.error(f"Erro ao inserir chunk {idx} de {fname}: {e}")

    logger.info("-" * 40)
    logger.info(f"✅ Concluído! {len(all_docs)} arquivos processados.")
    logger.info(f"📚 Total de fragmentos processados (Upsert): {total_inserted}")


if __name__ == "__main__":
    try:
        asyncio.run(populate_vectorstore())
    except KeyboardInterrupt:
        logger.info("Interrompido pelo usuário.")
    except Exception as e:
        logger.exception(f"Erro fatal: {e}")