# -*- coding: utf-8 -*-
# Objective: Pure assembly helpers for hybrid retrieval (fusion, trimming, provenance).
"""Deterministic helpers used by ``rag_local.build_retrieval_bundle``.

Reciprocal-rank fusion of dense + sparse hits, context/provenance trimming to
a ~4 chars/token budget, reranked-text to document matching, citations and
the augmented prompt. No I/O: ``rag_local`` re-exports them.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

SearchHits = Tuple[List[str], Dict[str, str], Dict[str, Dict[str, Any]]]


def _trim_context_to_budget(documents: List[str], token_budget: int) -> List[str]:
    """Trim retrieved documents so the assembled context stays within a soft token budget."""
    if token_budget <= 0:
        return list(documents)

    budget_chars = token_budget * 4
    total_chars = 0
    trimmed: List[str] = []
    for document in documents:
        if not document:
            continue
        remaining = budget_chars - total_chars
        if remaining <= 0:
            break
        if len(document) <= remaining:
            trimmed.append(document)
            total_chars += len(document)
            continue
        shortened = document[:remaining].rstrip()
        if shortened:
            trimmed.append(shortened)
        break
    return trimmed


def _snippet(text: str, limit: int = 220) -> str:
    """Return one compact evidence snippet extracted from a retrieved document."""
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)].rstrip() + "…"


def reciprocal_rank_fusion(
    vector_results: List[str], # Lista de Doc IDs
    bm25_results: List[str],   # Lista de Doc IDs
    k: int = 60
) -> List[str]:
    """
    Combina duas listas de resultados usando RRF.
    Score = 1 / (k + rank).
    """
    scores: Dict[str, float] = {}

    # Processa Vetorial
    for rank, doc_id in enumerate(vector_results):
        if doc_id not in scores: scores[doc_id] = 0.0
        scores[doc_id] += 1 / (k + rank + 1)

    # Processa BM25
    for rank, doc_id in enumerate(bm25_results):
        if doc_id not in scores: scores[doc_id] = 0.0
        scores[doc_id] += 1 / (k + rank + 1)

    # Ordena pelo score final
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, score in sorted_docs]


Candidate = Tuple[str, str, Dict[str, Any]]  # (doc_id, texto, metadados)


def _fuse_candidates(dense: SearchHits, sparse: SearchHits) -> List[Candidate]:
    """RRF over dense + BM25 ids, keeping the ids for which some text is known."""
    dense_ids, dense_docs, dense_meta = dense
    sparse_ids, sparse_docs, sparse_meta = sparse
    items: List[Candidate] = []
    for doc_id in reciprocal_rank_fusion(dense_ids, sparse_ids):
        text = dense_docs.get(doc_id) or sparse_docs.get(doc_id)
        if text:
            items.append((doc_id, text, dense_meta.get(doc_id) or sparse_meta.get(doc_id) or {}))
    return items


def _match_items(final_docs: List[str], items: List[Candidate]) -> List[Candidate]:
    """Map reranked texts back to their (doc_id, metadata), each doc used once."""
    matched: List[Candidate] = []
    used: set = set()
    for doc in final_docs:
        hit = next((item for item in items if item[0] not in used and item[1] == doc), None)
        if hit is not None:
            matched.append(hit)
            used.add(hit[0])
    return matched


def _trim_items(items: List[Candidate], token_budget: int) -> List[Candidate]:
    """Provenance items cut to the same ~4 chars/token budget as the context."""
    trimmed: List[Candidate] = []
    remaining = token_budget * 4 if token_budget else 0
    for doc_id, text, meta in items:
        if remaining <= 0:
            break
        if len(text) <= remaining:
            trimmed.append((doc_id, text, meta))
            remaining -= len(text)
            continue
        shortened = text[:remaining].rstrip()
        if shortened:
            trimmed.append((doc_id, shortened, meta))
        break
    return trimmed


def _provenance(items: List[Candidate]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Citations and evidence snippets, ranked from 1."""
    citations: List[Dict[str, Any]] = []
    evidence: List[Dict[str, Any]] = []
    for rank, (doc_id, text, meta) in enumerate(items, start=1):
        source = str(meta.get("source") or meta.get("title") or meta.get("uri") or "vectorstore")
        citations.append({"doc_id": str(doc_id), "rank": rank, "source": source, "snippet": _snippet(text), "score": None})
        evidence.append({"doc_id": str(doc_id), "rank": rank, "source": source, "text": _snippet(text, limit=320)})
    return citations, evidence


def _augmented_prompt(context: str, query: str) -> str:
    if not context:
        return query
    return (
        "INSTRUÇÃO DE CONTEXTO (RAG):\n"
        "Use as informações técnicas abaixo recuperadas do banco de dados para auxiliar na sua resposta.\n"
        "------ CONTEXTO RECUPERADO (Híbrido + Re-rank) ------\n"
        f"{context}\n"
        "---------------------------------\n"
        f"PERGUNTA DO USUÁRIO: {query}"
    )
