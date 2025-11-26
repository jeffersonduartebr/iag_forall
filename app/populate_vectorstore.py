# -*- coding: utf-8 -*-
"""
populate_vectorstore.py (Versão Corrigida: Imports Absolutos)
-------------------------------------------------------
Popula a base vetorial (ChromaDB) com textos, PDFs e
Markdowns para uso em RAG.

Funcionalidades:
✅ Detecção automática de PDF digital vs escaneado
✅ OCR (Tesseract) se necessário
✅ Geração de resumo e título via LLM (Async)
✅ Inserção assíncrona no ChromaDB
"""

import os
import re
import uuid
import logging
import asyncio
from pathlib import Path
from typing import List, Tuple

# Bibliotecas de PDF
import fitz  # PyMuPDF

# Bibliotecas de OCR (Opcionais - requer instalação no SO)
try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ============================================================
# 🛠️ CORREÇÃO: USAR IMPORTS ABSOLUTOS (app.xyz)
# ============================================================
# O erro ocorria aqui ao usar 'from .vectorstore'
from app.vectorstore import add_document, get_or_create_collection_async
from app.embeddings import embed_text
from app.providers_async import call_model
from app.settings_dynamic import settings

# ============================================================
# ⚙️ CONFIGURAÇÃO
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] populate: %(message)s",
)
logger = logging.getLogger("populate_vectorstore")

COLLECTION_NAME = settings.get("RAG_COLLECTION_NAME", "knowledge_base")
# Fallback para data/rag_docs se não definido
DATA_DIR = settings.get("RAG_DATA_DIR", "/app/data/rag_docs")
SUMMARY_MODEL = settings.get("SUMMARY_MODEL", "ollama/phi4:latest")

CHUNK_SIZE = 1000
OVERLAP = 150


# ============================================================
# ✂️ UTILITÁRIOS DE TEXTO
# ============================================================
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
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
# 👁️ LÓGICA DE OCR (PDF ESCANEADO)
# ============================================================
def _perform_ocr(path: Path) -> str:
    """Converte páginas do PDF em imagens e roda Tesseract."""
    logger.info(f"[OCR] Iniciando reconhecimento ótico em: {path.name}...")
    text_accum = []
    try:
        # Converte PDF para lista de imagens (uma por página)
        # thread_count=4 acelera o processo
        images = convert_from_path(str(path), thread_count=4)
        
        for i, image in enumerate(images):
            # Extrai texto da imagem (lang='por+eng' para suportar PT e EN)
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
    """Retorna {nome_arquivo: [fragmentos]}."""
    docs = {}
    path_obj = Path(folder)
    
    if not path_obj.exists():
        logger.warning(f"Pasta {folder} não encontrada — criando...")
        path_obj.mkdir(parents=True, exist_ok=True)
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
        logger.info(f"   📝 Resumo: {summary[:100]}...")

        # Insere cada fragmento
        for idx, frag in enumerate(fragments):
            try:
                doc_id = str(uuid.uuid4())
                meta = {
                    "source": fname,
                    "chunk_index": idx,
                    "total_chunks": len(fragments),
                    "title": title,
                    "summary_snippet": summary[:200] # guarda parte do resumo no metadado
                }
                
                # add_document já calcula o embedding internamente via threadpool
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
    logger.info(f"📚 Total de fragmentos inseridos: {total_inserted}")


if __name__ == "__main__":
    try:
        asyncio.run(populate_vectorstore())
    except KeyboardInterrupt:
        logger.info("Interrompido pelo usuário.")
    except Exception as e:
        logger.exception(f"Erro fatal: {e}")