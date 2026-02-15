# -*- coding: utf-8 -*-
"""
sparse_index.py — Gerenciador de Índice BM25 (Busca por Palavras-Chave)
-----------------------------------------------------------------------
Implementa um índice esparso local usando o algoritmo BM25 (Okapi).
Funciona em paralelo ao ChromaDB para permitir Busca Híbrida.
"""

import os
import pickle
import logging
import re
from typing import List, Tuple
from rank_bm25 import BM25Okapi

from .settings_dynamic import settings

logger = logging.getLogger(__name__)

# Caminho para persistência do índice BM25
# Garante que o diretório data exista
DATA_DIR = settings.get("RAG_DATA_DIR", "/app/data")
INDEX_PATH = os.path.join(DATA_DIR, "bm25_index.pkl")

class SparseIndex:
    """Classe `SparseIndex`: organiza responsabilidades de sparse index."""
    def __init__(self):
        """Inicializa estado interno necessário para uso da classe."""
        self.documents: List[str] = []
        self.doc_ids: List[str] = []
        self.bm25 = None
        self.is_dirty = False
        self._load()

    def _tokenize(self, text: str) -> List[str]:
        """Tokenização simples para o BM25 (lowercase + split)."""
        text = text.lower()
        # Remove pontuação básica para melhorar o match
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()

    def add_document(self, doc_id: str, text: str):
        """Adiciona um documento ao índice em memória."""
        if not text or not text.strip():
            return
        
        # Evita duplicatas de ID (atualização simplificada: remove e adiciona)
        if doc_id in self.doc_ids:
            idx = self.doc_ids.index(doc_id)
            self.doc_ids.pop(idx)
            self.documents.pop(idx)

        self.doc_ids.append(doc_id)
        self.documents.append(text)
        self.is_dirty = True

    def commit(self):
        """Reconstrói o índice BM25 e salva no disco."""
        if not self.is_dirty:
            return

        if not self.documents:
            return

        logger.info(f"[SparseIndex] Reconstruindo índice BM25 com {len(self.documents)} docs...")
        try:
            tokenized_corpus = [self._tokenize(doc) for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
            self._save()
            self.is_dirty = False
            logger.info("[SparseIndex] Índice atualizado e salvo.")
        except Exception as e:
            logger.error(f"[SparseIndex] Erro ao commitar índice: {e}")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Retorna [(doc_id, score), ...]
        """
        if not self.bm25 or not self.documents:
            return []

        try:
            tokenized_query = self._tokenize(query)
            # O rank_bm25 retorna scores para todos os documentos
            scores = self.bm25.get_scores(tokenized_query)
            
            # Emparelha scores com IDs
            scored_results = []
            for idx, score in enumerate(scores):
                if score > 0: # Filtra irrelevantes
                    scored_results.append((self.doc_ids[idx], float(score)))
            
            # Ordena e corta
            scored_results.sort(key=lambda x: x[1], reverse=True)
            return scored_results[:top_k]
        except Exception as e:
            logger.warning(f"[SparseIndex] Erro na busca: {e}")
            return []

    def get_text(self, doc_id: str) -> str:
        """Recupera o texto original pelo ID."""
        try:
            idx = self.doc_ids.index(doc_id)
            return self.documents[idx]
        except ValueError:
            return ""

    def _save(self):
        """Executa save."""
        try:
            os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
            with open(INDEX_PATH, "wb") as f:
                pickle.dump({
                    "docs": self.documents,
                    "ids": self.doc_ids,
                    "bm25": self.bm25
                }, f)
        except Exception as e:
            logger.error(f"[SparseIndex] Erro ao salvar índice em disco: {e}")

    def _load(self):
        """Executa load."""
        if os.path.exists(INDEX_PATH):
            try:
                with open(INDEX_PATH, "rb") as f:
                    data = pickle.load(f)
                    self.documents = data["docs"]
                    self.doc_ids = data["ids"]
                    self.bm25 = data["bm25"]
                logger.info(f"[SparseIndex] Índice carregado ({len(self.documents)} docs).")
            except Exception as e:
                logger.warning(f"[SparseIndex] Índice corrompido ou antigo, iniciando novo: {e}")

# Singleton Global
sparse_index = SparseIndex()
