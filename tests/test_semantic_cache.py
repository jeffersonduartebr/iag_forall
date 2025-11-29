import pytest
from unittest.mock import patch, AsyncMock
from app.app.semantic_cache import check_cache, _compute_sha256

def test_hash_consistency():
    """O hash deve ser determinístico."""
    t1 = "Como fazer bolo?"
    h1 = _compute_sha256(t1)
    h2 = _compute_sha256(t1)
    assert h1 == h2
    assert len(h1) == 64 # SHA256 hex

@pytest.mark.asyncio
async def test_cache_hit_logic():
    """Simula um HIT no cache vetorial."""
    
    # Mock do retorno do ChromaDB
    mock_results = {
        "distances": [[0.05]], # Similaridade = 1 - 0.05 = 0.95
        "documents": [["Resposta cacheada"]],
        "metadatas": [[{"model_used": "gpt-4", "answer_payload": "Resposta cacheada"}]]
    }
    
    with patch("app.app.semantic_cache._make_embedding", new_callable=AsyncMock) as mock_emb, \
         patch("app.app.semantic_cache.query_embedding", new_callable=AsyncMock) as mock_query:
        
        mock_emb.return_value = [0.1, 0.2, 0.3]
        mock_query.return_value = mock_results
        
        # Threshold é 0.92 no settings default
        result = await check_cache("pergunta", threshold=0.90)
        
        assert result is not None
        assert result["text"] == "Resposta cacheada"
        assert result["similarity"] > 0.90

@pytest.mark.asyncio
async def test_cache_miss_logic():
    """Simula um MISS no cache (distância alta)."""
    
    mock_results = {
        "distances": [[0.5]], # Similaridade = 0.5 (Baixa)
        "documents": [["Outra coisa"]],
        "metadatas": [[{}]]
    }
    
    with patch("app.app.semantic_cache._make_embedding", new_callable=AsyncMock), \
         patch("app.app.semantic_cache.query_embedding", new_callable=AsyncMock) as mock_query:
        
        mock_query.return_value = mock_results
        
        result = await check_cache("pergunta", threshold=0.90)
        
        assert result is None