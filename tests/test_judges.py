import pytest
import asyncio
from app import judges


@pytest.mark.asyncio
async def test_heuristic_score_valid():
    """Avalia se o score heurístico retorna dentro do intervalo esperado."""
    score = judges.heuristic_score("Esta é uma resposta com pontuação final.")
    assert 0.0 <= score <= 1.0


@pytest.mark.asyncio
async def test_heuristic_score_empty():
    """Respostas vazias devem gerar score 0."""
    score = judges.heuristic_score("")
    assert score == 0.0


@pytest.mark.asyncio
async def test_judge_answer_empty_response():
    """Se a resposta for vazia, deve retornar score 0."""
    result = await judges.judge_answer("Pergunta X", "")
    assert isinstance(result, list)
    assert result[0]["score"] == 0.0


@pytest.mark.asyncio
async def test_llm_based_score_mock(monkeypatch):
    """Substitui o modelo real por mock e verifica conversão do score."""
    async def fake_call_model(*args, **kwargs):
        return "Nota 9", {}

    monkeypatch.setattr(judges, "call_model", fake_call_model)
    score = await judges.llm_based_score("Pergunta", "Resposta", use_rag=False)
    assert 0.0 <= score <= 1.0


def test_extract_score_various_cases():
    """Testa a extração de números válidos e inválidos."""
    assert judges.extract_score("9") == 9.0
    assert judges.extract_score("nota 12") == 10.0
    assert judges.extract_score("ruim") == 0.0
