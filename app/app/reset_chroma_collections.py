# Objective: Application runtime code for reset chroma collections.
"""Application runtime code for reset chroma collections.

This module is part of the tracked codebase and should remain aligned with the
current runtime architecture and operational documentation.
"""


from chromadb import PersistentClient


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
    removed = []
    kept = []

    for collection in client.list_collections():
        try:
            meta = collection.metadata or {}
            dim = meta.get("dimension") or meta.get("embedding_dimension") or None

            # tenta inferir a dimensão se não estiver nos metadados
            if dim is None:
                try:
                    first = collection.peek()
                    if first and "embeddings" in first and first["embeddings"]:
                        dim = len(first["embeddings"][0])
                except Exception:
                    pass

            # decide se deve excluir
            if dim is not None and dim != expected_dim:
                client.delete_collection(collection.name)
                removed.append((collection.name, dim))
            else:
                kept.append((collection.name, dim))
        except Exception as e:
            print(f"[WARN] Falha ao inspecionar {collection.name}: {e}")

    summary = {
        "removed": removed,
        "kept": kept,
        "total_removed": len(removed),
        "total_kept": len(kept),
    }

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
    return summary
