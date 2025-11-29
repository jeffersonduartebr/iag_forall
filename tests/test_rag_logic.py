import pytest
from app.app.rag_local import reciprocal_rank_fusion

def test_rrf_logic():
    """
    Testa a fusão de rankings.
    Doc 'A' é 1º no vetor e 2º no BM25.
    Doc 'B' é 2º no vetor e 1º no BM25.
    Devem ter scores similares e estar no topo.
    """
    vec_results = ["docA", "docB", "docC"]
    bm25_results = ["docB", "docA", "docD"]
    
    k = 1 # k pequeno para facilitar conta manual
    # Score A = 1/(1+0+1) + 1/(1+1+1) = 0.5 + 0.33 = 0.83
    # Score B = 1/(1+1+1) + 1/(1+0+1) = 0.33 + 0.5 = 0.83
    # Score C = 1/(1+2+1) = 0.25
    
    merged = reciprocal_rank_fusion(vec_results, bm25_results, k=1)
    
    assert merged[0] in ["docA", "docB"]
    assert merged[1] in ["docA", "docB"]
    assert "docC" in merged
    assert "docD" in merged
    assert len(merged) == 4

def test_rrf_empty():
    """Testa listas vazias."""
    assert reciprocal_rank_fusion([], []) == []