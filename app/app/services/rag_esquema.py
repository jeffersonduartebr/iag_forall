# Objective: RAG schema invariants and model loading shared by embeddings, vector store and reranker.
"""What makes a stored vector valid, which collection schema it lives in, and how CPU models are loaded.

- Text collection schema ``d2``: documents embedded with Nomic's ``search_document`` prefix, cosine space. Vectors of
  the older schema (query prefix for everything, L2) are not comparable, so the schema is part of the collection name.
- A failed embedding (``[0.0]``, all zeros, or a vector of another model) must never reach Chroma.
- Model loads are serialized and a failed load is retried only after a back-off: concurrent first requests used to
  load the model twice, and a failure re-downloaded it on every request.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger(__name__)

TEXT_SCHEMA = "d2"
COSINE = {"hnsw:space": "cosine"}


def vetor_utilizavel(vec: Iterable[float]) -> bool:
    """A real embedding: more than one dimension and not all zeros (the failure fallbacks are neither)."""
    valores = list(vec)
    return len(valores) > 1 and any(valores)


def com_prefixo(text: str, tarefa: str, modelo: str) -> str:
    """Nomic v1.5 task prefix: ``search_document`` for corpus chunks, ``search_query`` for queries.

    https://huggingface.co/nomic-ai/nomic-embed-text-v1.5#task-instruction-prefixes
    """
    if "nomic" not in modelo or text.startswith("search_"):
        return text
    return f"{'search_document' if tarefa == 'documento' else 'search_query'}: {text}"


class CarregadorComRecuo:
    """One lazily loaded model per process: loads are serialized; a failure is retried only after ``recuo_s``."""

    def __init__(self, nome: str, recuo_s: float = 60.0) -> None:
        self.nome, self.recuo_s = nome, recuo_s
        self.instancia: Any = None
        self._trava = threading.Lock()
        self._falhou_em: Optional[float] = None

    def obter(self, fabrica: Callable[[], Any]) -> Any:
        """The loaded model, loading it with ``fabrica`` when allowed; ``None`` while unavailable."""
        if self.instancia is not None:
            return self.instancia
        with self._trava:
            pode = self._falhou_em is None or time.monotonic() - self._falhou_em >= self.recuo_s
            if self.instancia is None and pode:
                logger.info("[rag] Carregando %s...", self.nome)
                try:
                    self.instancia = fabrica()
                except Exception as exc:
                    self._falhou_em = time.monotonic()
                    logger.error(
                        "[rag] Falha ao carregar %s (nova tentativa em %.0f s): %s", self.nome, self.recuo_s, exc
                    )
        return self.instancia

    def redefinir(self, instancia: Any = None) -> None:
        """Replace the cached model (tests, hot reload) and clear the failure back-off."""
        self.instancia, self._falhou_em = instancia, None
