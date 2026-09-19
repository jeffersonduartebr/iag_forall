# Objective: Application runtime code for reset chroma collections.
"""Application runtime code for reset chroma collections.

This module is part of the tracked codebase and should remain aligned with the
current runtime architecture and operational documentation.
"""

from typing import Any, List, Optional, Tuple

from chromadb import PersistentClient


def _collection_dim(collection) -> Optional[int]:
    """Embedding dimension from the collection metadata, else inferred from its first stored vector."""
    meta = collection.metadata or {}
    dim = meta.get("dimension") or meta.get("embedding_dimension")
    if dim is not None:
        return dim
    try:
        first = collection.peek()
        if first and "embeddings" in first and first["embeddings"]:
            return len(first["embeddings"][0])
    except Exception:
        pass
    return None


def _print_summary(removed: List[Tuple[str, Any]], kept: List[Tuple[str, Any]]) -> None:
    print("\n=== Resumo da Limpeza ===")
    if removed:
        print("Coleções removidas (dimensão incorreta):")
        for n, d in removed:
            print(f" - {n} (dim={d})")
    else:
        print("Nenhuma coleção incompatível encontrada.")

    print("\nColeções mantidas:")
    for n, d in kept:
        print(f" - {n} (dim={d})")

    print(f"\nTotal removido: {len(removed)} | Total mantido: {len(kept)}")


def reset_incompatible_collections(chroma_path: str = "/app/chroma_data", expected_dim: int = 1024):
    """
    Remove coleções do ChromaDB cuja dimensão de embeddings é diferente da esperada.

    Args:
        chroma_path (str): Caminho para o diretório de persistência do ChromaDB.
        expected_dim (int): Dimensão esperada do modelo de embeddings atual.

    Returns:
        dict: Um resumo com listas de coleções removidas e mantidas.
    """
    client = PersistentClient(path=chroma_path)
    removed: List[Tuple[str, Any]] = []
    kept: List[Tuple[str, Any]] = []

    for collection in client.list_collections():
        try:
            dim = _collection_dim(collection)
            if dim is not None and dim != expected_dim:
                client.delete_collection(collection.name)
                removed.append((collection.name, dim))
            else:
                kept.append((collection.name, dim))
        except Exception as e:
            print(f"[WARN] Falha ao inspecionar {collection.name}: {e}")

    _print_summary(removed, kept)
    return {
        "removed": removed,
        "kept": kept,
        "total_removed": len(removed),
        "total_kept": len(kept),
    }
