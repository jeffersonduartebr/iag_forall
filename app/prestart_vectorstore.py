# -*- coding: utf-8 -*-
"""
prestart_vectorstore.py
----------------------------------------------------
Verifica e sincroniza automaticamente o vetorstore (Chroma)
com o modelo de embeddings configurado.
- Detecta o modelo ativo (via .env ou LiteLLM/vLLM)
- Obtém a dimensão real (se disponível via API)
- Recria a coleção se houver divergência dimensional
----------------------------------------------------
Autor: Jefferson Igor Duarte Silva
"""

import os
import json
import shutil
import requests
from chromadb import PersistentClient
from app.reset_chroma_collections import reset_incompatible_collections
# ============================================================
# ⚙️ Configurações
# ============================================================
CHROMA_PATH = os.getenv("CHROMA_PATH", "/data/chroma")
COLLECTION_NAME = os.getenv("VECTOR_COLLECTION", "embeddings_default")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://vllm:8000/v1")
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "http://api:8000/v1")

print(f"🔍 [vectorstore] Verificando compatibilidade da coleção '{COLLECTION_NAME}' ({EMBED_MODEL})...")

# ============================================================
# 🔎 Função auxiliar para detectar dimensão real
# ============================================================
def detect_embedding_dim(model_name: str) -> int:
    """
    Detecta automaticamente a dimensão do modelo de embedding.
    Tenta via vLLM -> LiteLLM -> heurísticas conhecidas.
    """
    # 1️⃣ Tenta via vLLM (caso modelo local)
    try:
        resp = requests.get(f"{VLLM_BASE_URL}/models", timeout=5)
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            for m in data:
                if model_name.lower() in m.get("id", "").lower():
                    emb_dim = m.get("embedding_dimension") or m.get("hidden_size")
                    if emb_dim:
                        print(f"✅ [vLLM] Dimensão detectada automaticamente: {emb_dim}")
                        return int(emb_dim)
    except Exception:
        pass

    # 2️⃣ Tenta via LiteLLM / API local (caso embeddings externos)
    try:
        resp = requests.post(
            f"{LITELLM_BASE_URL}/embeddings",
            json={"model": model_name, "input": ["teste de embedding"]},
            timeout=10,
        )
        if resp.status_code == 200:
            vec = resp.json()["data"][0]["embedding"]
            emb_dim = len(vec)
            print(f"✅ [LiteLLM] Dimensão detectada automaticamente: {emb_dim}")
            return emb_dim
    except Exception:
        pass

    # 3️⃣ Heurísticas conhecidas
    heuristics = {
        "text-embedding-ada-002": 1536,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "BAAI/bge-base-en": 768,
        "BAAI/bge-large-en": 1024,
        "BAAI/bge-m3": 1024,
        "intfloat/e5-large-v2": 1024,
    }

    for key, val in heuristics.items():
        if key.lower() in model_name.lower():
            print(f"ℹ️ [heurística] Dimensão estimada: {val}")
            return val

    print(f"⚠️ [fallback] Usando EMBED_DIM declarado no .env: {EMBED_DIM}")
    return EMBED_DIM


# ============================================================
# 🧠 Verifica e corrige coleção
# ============================================================
try:
    detected_dim = detect_embedding_dim(EMBED_MODEL)
    client = PersistentClient(path=CHROMA_PATH)
    colls = client.list_collections()
    existing = [c.name for c in colls]

    if COLLECTION_NAME not in existing:
        print(f"🆕 Coleção '{COLLECTION_NAME}' inexistente — criando ({detected_dim}D).")
        client.get_or_create_collection(name=COLLECTION_NAME, metadata={"embedding_dim": detected_dim})

    else:
        collection = client.get_collection(COLLECTION_NAME)
        current_dim = None

        try:
            meta = getattr(collection, "metadata", None) or {}
            current_dim = meta.get("embedding_dim") or meta.get("dimension")
            emb_fn = getattr(collection, "_embedding_function", None)
            if not current_dim and emb_fn is not None:
                current_dim = getattr(emb_fn, "embedding_dimension", None)
        except Exception:
            current_dim = None

        if current_dim and current_dim != detected_dim:
            print(f"⚠️ Dimensão incompatível: atual={current_dim}, detectado={detected_dim}")
            backup_path = os.path.join(CHROMA_PATH, f"{COLLECTION_NAME}_backup")
            if os.path.exists(backup_path):
                shutil.rmtree(backup_path, ignore_errors=True)
            shutil.move(os.path.join(CHROMA_PATH, COLLECTION_NAME), backup_path)
            client.delete_collection(COLLECTION_NAME)
            print(reset_incompatible_collections(chroma_path=CHROMA_PATH, expected_dim=detected_dim)    )
            client.get_or_create_collection(name=COLLECTION_NAME, metadata={"embedding_dim": detected_dim})
            print(f"✅ Nova coleção '{COLLECTION_NAME}' criada com {detected_dim}D.")
        else:
            print(f"✅ Dimensão consistente ({detected_dim}D). Nenhuma ação necessária.")

except Exception as e:
    print(f"❌ [vectorstore] Falha ao verificar/criar coleção: {e}")

print("📘 [vectorstore] Verificação concluída.")

