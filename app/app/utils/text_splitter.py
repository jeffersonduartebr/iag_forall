# -*- coding: utf-8 -*-
import re
import numpy as np
import logging
from typing import List

# Importa seu gerador de embeddings existente
from app.embeddings import embed_text

logger = logging.getLogger(__name__)

def split_into_sentences(text: str) -> List[str]:
    # Regex robusto para separar sentenças (preserva pontuação)
    return re.split(r'(?<=[.?!])\s+', text)

def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0: return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))

async def semantic_chunking(
    text: str, 
    similarity_threshold_percentile: int = 20, # Corta nos 20% de menor similaridade
    max_chunk_size: int = 2000,
    min_chunk_size: int = 100 
) -> List[str]:
    """
    Divide o texto dinamicamente baseado na mudança de tópico.
    """
    # 1. Divide em sentenças
    sentences = split_into_sentences(text)
    if len(sentences) <= 1:
        return [text]

    # 2. Gera embeddings para todas as sentenças (poderia ser batch para otimizar)
    # Nota: Isso pode ser lento se o texto for um livro inteiro.
    # Idealmente, processe parágrafos grandes antes.
    
    # Para evitar muitas chamadas de API, combinamos sentenças muito curtas antes
    combined_sentences = []
    buffer = ""
    for s in sentences:
        buffer += s + " "
        if len(buffer) > 50: # Agrupa sentenças mínimas
            combined_sentences.append(buffer.strip())
            buffer = ""
    if buffer: combined_sentences.append(buffer.strip())
    
    sentences = combined_sentences
    
    # Gera vetores
    vecs = []
    for s in sentences:
        try:
            # embed_text é síncrono, mas no fluxo async usamos wrapper se necessário
            # Aqui assumimos que embed_text retorna lista de floats
            v = embed_text(s) 
            vecs.append(np.array(v))
        except Exception:
            vecs.append(np.zeros(384)) # Fallback dimensão (ajuste conforme seu modelo)

    # 3. Calcula distâncias (cosseno) entre sentenças adjacentes
    distances = []
    for i in range(len(vecs) - 1):
        sim = _cosine_similarity(vecs[i], vecs[i+1])
        distances.append(sim)

    # 4. Define o limiar de corte (Threshold)
    # O corte acontece onde a similaridade é BAIXA (vales no gráfico de similaridade)
    if not distances:
        return [text]
        
    threshold = np.percentile(distances, similarity_threshold_percentile)
    
    # 5. Constrói os chunks
    chunks = []
    current_chunk = sentences[0]
    
    for i in range(len(distances)):
        sim = distances[i]
        next_sent = sentences[i+1]
        
        # Se mudou muito o assunto (sim < threshold) OU o chunk está gigante
        if sim < threshold or len(current_chunk) > max_chunk_size:
            if len(current_chunk) >= min_chunk_size:
                chunks.append(current_chunk)
                current_chunk = next_sent
            else:
                # Se o chunk é muito pequeno, força a união mesmo mudando o assunto
                current_chunk += " " + next_sent
        else:
            # Assunto parecido, continua agrupando
            current_chunk += " " + next_sent
            
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks