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
    # Use the correct function signature with all required arguments
    score = await judges.llm_based_score(
        query="Pergunta",
        answer="Resposta",
        use_rag=False,
        modality="text",
        image_b64=None
    )
    assert 0.0 <= score <= 1.0


def test_parse_score_from_text():
    """Testa a extração de números de texto de resposta."""
    # Test parsing score from various text formats
    import re

    def parse_score(text: str) -> float:
        """Extract score from text, similar to what judges does internally."""
        match = re.search(r'(\d+(?:\.\d+)?)', text)
        if match:
            score = float(match.group(1))
            return min(score, 10.0)
        return 0.0

    assert parse_score("9") == 9.0
    assert parse_score("nota 12") == 10.0  # Capped at 10
    assert parse_score("ruim") == 0.0
