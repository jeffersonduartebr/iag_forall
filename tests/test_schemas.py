# -*- coding: utf-8 -*-
# Objective: Test coverage for schemas behavior and regressions.
"""
test_schemas.py — Tests for Pydantic Schema Validation
--------------------------------------------------------
Comprehensive tests for schema validation including:
- QueryRequest validation and constraints
- QueryResponse structure
- Boundary testing for field limits
- Custom validator behavior
"""

import pytest
from pydantic import ValidationError


class TestModalityEnum:
    """Tests for Modality enum."""

    def test_modality_values(self):
        """Test that Modality enum has expected values."""
        from app.schemas import Modality

        assert Modality.TEXT == "text"
        assert Modality.VISION == "vision"
        assert Modality.MULTIMODAL == "multimodal"


class TestQueryRequest:
    """Tests for QueryRequest schema validation."""

    def test_valid_minimal_request(self):
        """Test minimal valid request."""
        from app.schemas import QueryRequest

        req = QueryRequest(query="What is Python?")

        assert req.query == "What is Python?"
        assert req.modality == "text"
        assert req.max_tokens == 512
        assert req.temperature == 0.5

    def test_valid_full_request(self):
        """Test request with all optional fields."""
        from app.schemas import QueryRequest

        req = QueryRequest(
            query="Describe this image",
            modality="vision",
            image_b64="base64encodeddata",
            enable_rag_for_answer=True,
            enable_rag_for_image=True,
            rag_modality="multimodal",
            max_tokens=1000,
            temperature=0.8,
            system_prompt="You are a helpful assistant.",
            use_cache=False,
            timeout_seconds=120,
        )

        assert req.query == "Describe this image"
        assert req.modality == "vision"
        assert req.enable_rag_for_answer is True
        assert req.max_tokens == 1000

    def test_query_required(self):
        """Test that query is required."""
        from app.schemas import QueryRequest

        with pytest.raises(ValidationError) as exc_info:
            QueryRequest()

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("query",) for e in errors)

    def test_query_min_length(self):
        """Test query minimum length constraint."""
        from app.schemas import QueryRequest

        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(query="")

        error_msg = str(exc_info.value).lower()
        assert "min_length" in error_msg or "too_short" in error_msg or "at least 1" in error_msg

    def test_query_whitespace_only_rejected(self):
        """Test that whitespace-only query is rejected by validator."""
        from app.schemas import QueryRequest

        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(query="   \n\t   ")

        assert "empty" in str(exc_info.value).lower() or "whitespace" in str(exc_info.value).lower()

    def test_query_max_length(self):
        """Test query maximum length constraint."""
        from app.schemas import QueryRequest

        # Create a query that exceeds max_length (100000)
        long_query = "x" * 100001

        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(query=long_query)

        assert "max_length" in str(exc_info.value) or "100000" in str(exc_info.value)

    def test_query_strips_whitespace(self):
        """Test that query validator strips whitespace."""
        from app.schemas import QueryRequest

        req = QueryRequest(query="  Hello world  ")

        assert req.query == "Hello world"

    def test_modality_validation_valid(self):
        """Test valid modality values."""
        from app.schemas import QueryRequest

        for modality in ["text", "vision", "multimodal"]:
            req = QueryRequest(query="test", modality=modality)
            assert req.modality == modality

    def test_modality_validation_invalid(self):
        """Test invalid modality value is rejected."""
        from app.schemas import QueryRequest

        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(query="test", modality="invalid_modality")

        assert "modality" in str(exc_info.value).lower()

    def test_modality_case_insensitive(self):
        """Test that modality validation is case insensitive."""
        from app.schemas import QueryRequest

        req = QueryRequest(query="test", modality="TEXT")
        assert req.modality == "text"

        req = QueryRequest(query="test", modality="Vision")
        assert req.modality == "vision"

    def test_max_tokens_boundaries(self):
        """Test max_tokens boundary validation."""
        from app.schemas import QueryRequest

        # Valid minimum
        req = QueryRequest(query="test", max_tokens=1)
        assert req.max_tokens == 1

        # Valid maximum
        req = QueryRequest(query="test", max_tokens=32000)
        assert req.max_tokens == 32000

        # Invalid: below minimum
        with pytest.raises(ValidationError):
            QueryRequest(query="test", max_tokens=0)

        # Invalid: above maximum
        with pytest.raises(ValidationError):
            QueryRequest(query="test", max_tokens=32001)

    def test_temperature_boundaries(self):
        """Test temperature boundary validation."""
        from app.schemas import QueryRequest

        # Valid minimum
        req = QueryRequest(query="test", temperature=0.0)
        assert req.temperature == 0.0

        # Valid maximum
        req = QueryRequest(query="test", temperature=2.0)
        assert req.temperature == 2.0

        # Invalid: below minimum
        with pytest.raises(ValidationError):
            QueryRequest(query="test", temperature=-0.1)

        # Invalid: above maximum
        with pytest.raises(ValidationError):
            QueryRequest(query="test", temperature=2.1)

    def test_timeout_boundaries(self):
        """Test timeout_seconds boundary validation."""
        from app.schemas import QueryRequest

        # Valid minimum
        req = QueryRequest(query="test", timeout_seconds=5)
        assert req.timeout_seconds == 5

        # Valid maximum
        req = QueryRequest(query="test", timeout_seconds=600)
        assert req.timeout_seconds == 600

        # Invalid: below minimum
        with pytest.raises(ValidationError):
            QueryRequest(query="test", timeout_seconds=4)

        # Invalid: above maximum
        with pytest.raises(ValidationError):
            QueryRequest(query="test", timeout_seconds=601)

    def test_images_list_max_length(self):
        """Test images list max length constraint."""
        from app.schemas import QueryRequest

        # Valid: 10 images
        images = ["img" + str(i) for i in range(10)]
        req = QueryRequest(query="test", images=images)
        assert len(req.images) == 10

        # Invalid: 11 images
        images = ["img" + str(i) for i in range(11)]
        with pytest.raises(ValidationError):
            QueryRequest(query="test", images=images)

    def test_rag_modality_validation(self):
        """Test rag_modality validation."""
        from app.schemas import QueryRequest

        # Valid
        req = QueryRequest(query="test", rag_modality="text")
        assert req.rag_modality == "text"

        # Invalid
        with pytest.raises(ValidationError):
            QueryRequest(query="test", rag_modality="invalid")


class TestJudgeScore:
    """Tests for JudgeScore schema."""

    def test_valid_judge_score(self):
        """Test valid JudgeScore creation."""
        from app.schemas import JudgeScore

        score = JudgeScore(
            judge_id="judge-1",
            score=8.5,
            rationale="Good answer",
            modality="text",
        )

        assert score.judge_id == "judge-1"
        assert score.score == 8.5

    def test_minimal_judge_score(self):
        """Test minimal JudgeScore with only required fields."""
        from app.schemas import JudgeScore

        score = JudgeScore(judge_id="judge-1", score=7.0)

        assert score.judge_id == "judge-1"
        assert score.score == 7.0
        assert score.rationale is None
        assert score.modality == "text"


class TestCandidateResult:
    """Tests for CandidateResult schema."""

    def test_valid_candidate_result(self):
        """Test valid CandidateResult creation."""
        from app.schemas import CandidateResult

        result = CandidateResult(
            model="gpt-4o",
            modality="text",
            output="This is the answer",
            latency_s=1.5,
            prompt_tokens=100,
            completion_tokens=50,
            estimated_cost_usd=0.001,
            quality_score=8.0,
        )

        assert result.model == "gpt-4o"
        assert result.output == "This is the answer"
        assert result.latency_s == 1.5

    def test_minimal_candidate_result(self):
        """Test minimal CandidateResult."""
        from app.schemas import CandidateResult

        result = CandidateResult(model="ollama/llama3")

        assert result.model == "ollama/llama3"
        assert result.modality == "text"
        assert result.output == ""
        assert result.latency_s == 0.0

    def test_candidate_with_judge_scores(self):
        """Test CandidateResult with judge scores."""
        from app.schemas import CandidateResult, JudgeScore

        result = CandidateResult(
            model="gpt-4o",
            judge_scores=[
                JudgeScore(judge_id="judge-1", score=8.0),
                JudgeScore(judge_id="judge-2", score=9.0),
            ],
        )

        assert len(result.judge_scores) == 2
        assert result.judge_scores[0].score == 8.0


class TestRouteDecision:
    """Tests for RouteDecision schema."""

    def test_valid_route_decision(self):
        """Test valid RouteDecision creation."""
        from app.schemas import RouteDecision

        decision = RouteDecision(
            chosen_model="gpt-4o",
            modality_selected="text",
            is_multimodal_route=False,
            objectives={"cost": 0.001, "latency": 1.5, "quality": 8.5},
            explanation="Selected based on quality score",
        )

        assert decision.chosen_model == "gpt-4o"
        assert decision.objectives["quality"] == 8.5

    def test_minimal_route_decision(self):
        """Test minimal RouteDecision."""
        from app.schemas import RouteDecision

        decision = RouteDecision(chosen_model="ollama/phi4")

        assert decision.chosen_model == "ollama/phi4"
        assert decision.modality_selected == "text"
        assert decision.is_multimodal_route is False
        assert decision.objectives == {}


class TestQueryResponse:
    """Tests for QueryResponse schema."""

    def test_valid_query_response(self):
        """Test valid QueryResponse creation."""
        from app.schemas import QueryResponse, ResponseDiagnostics, RouteDecision

        response = QueryResponse(
            answer="This is the answer to your question.",
            model="gpt-4o",
            modality="text",
            correlation_id="abc-123",
            diagnostics=ResponseDiagnostics(route=RouteDecision(chosen_model="gpt-4o"), candidates=[]),
        )

        assert response.answer == "This is the answer to your question."
        assert response.model == "gpt-4o"
        assert response.correlation_id == "abc-123"

    def test_response_with_image(self):
        """Test QueryResponse with image output."""
        from app.schemas import QueryResponse, ResponseDiagnostics, RouteDecision

        response = QueryResponse(
            answer="Here's the generated image",
            model="dall-e-3",
            modality="vision",
            image_output_b64="base64imagedata",
            diagnostics=ResponseDiagnostics(
                route=RouteDecision(
                    chosen_model="dall-e-3",
                    modality_selected="vision",
                    is_multimodal_route=True,
                ),
            ),
        )

        assert response.image_output_b64 == "base64imagedata"
        assert response.diagnostics.route.is_multimodal_route is True

    def test_response_with_candidates(self):
        """Test QueryResponse with candidate results."""
        from app.schemas import CandidateResult, QueryResponse, ResponseDiagnostics, RouteDecision

        response = QueryResponse(
            answer="Final answer",
            model="gpt-4o",
            diagnostics=ResponseDiagnostics(
                route=RouteDecision(chosen_model="gpt-4o"),
                candidates=[
                    CandidateResult(model="gpt-4o", output="Answer 1"),
                    CandidateResult(model="claude-3", output="Answer 2"),
                ],
            ),
        )

        assert len(response.diagnostics.candidates) == 2

    def test_response_with_grounding_and_review_metadata(self):
        """QueryResponse should accept confidence, grounding, and review fields."""
        from app.schemas import (
            ConfidenceBand,
            EvidenceSnippet,
            QueryResponse,
            ResponseCitation,
            ResponseDiagnostics,
            ResponseProvenance,
            ReviewStatus,
            RouteDecision,
            VerificationStatus,
        )

        response = QueryResponse(
            answer="A fotossintese transforma luz em energia quimica.",
            model="ollama/qwen3.5:4b",
            diagnostics=ResponseDiagnostics(route=RouteDecision(chosen_model="ollama/qwen3.5:4b")),
            confidence_score=0.82,
            confidence_band=ConfidenceBand.HIGH,
            provenance=ResponseProvenance(
                grounded=True,
                citations=[ResponseCitation(doc_id="doc-1", rank=1, source="manual", snippet="A fotossintese converte luz...", score=0.91)],
                evidence_snippets=[EvidenceSnippet(doc_id="doc-1", text="A fotossintese converte luz em energia.", rank=1)],
                knowledge_version="text_embeddings_nomic_embed_text|sparse_v1",
            ),
            verification_status=VerificationStatus.SUPPORTED,
            review_status=ReviewStatus.AUTO_APPROVED,
        )

        assert response.provenance.grounded is True
        assert response.provenance.citations[0].doc_id == "doc-1"
        assert response.review_status == ReviewStatus.AUTO_APPROVED


class TestSchemaJsonSerialization:
    """Tests for JSON serialization of schemas."""

    def test_query_request_to_json(self):
        """Test QueryRequest JSON serialization."""
        import json

        from app.schemas import QueryRequest

        req = QueryRequest(query="Test query", max_tokens=100)
        json_str = req.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["query"] == "Test query"
        assert parsed["max_tokens"] == 100

    def test_query_response_to_json(self):
        """Test QueryResponse JSON serialization."""
        import json

        from app.schemas import QueryResponse, ResponseDiagnostics, RouteDecision

        response = QueryResponse(
            answer="Test answer",
            model="test-model",
            diagnostics=ResponseDiagnostics(route=RouteDecision(chosen_model="test-model")),
        )
        json_str = response.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["answer"] == "Test answer"
        assert parsed["diagnostics"]["route"]["chosen_model"] == "test-model"

    def test_query_response_serializes_citations(self):
        """Grounding metadata should survive JSON serialization."""
        import json

        from app.schemas import QueryResponse, ResponseDiagnostics, ResponseProvenance, RouteDecision

        response = QueryResponse(
            answer="Resposta ancorada",
            model="test-model",
            diagnostics=ResponseDiagnostics(route=RouteDecision(chosen_model="test-model")),
            provenance=ResponseProvenance(
                grounded=True,
                citations=[{"doc_id": "doc-1", "rank": 1, "source": "kb"}],
                evidence_snippets=[{"doc_id": "doc-1", "text": "trecho", "rank": 1}],
            ),
            verification_status="supported",
        )
        parsed = json.loads(response.model_dump_json())

        assert parsed["provenance"]["grounded"] is True
        assert parsed["provenance"]["citations"][0]["doc_id"] == "doc-1"
        assert parsed["verification_status"] == "supported"


class TestSchemaEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_unicode_query(self):
        """Test Unicode characters in query."""
        from app.schemas import QueryRequest

        req = QueryRequest(query="Qual é o significado da vida? 生命的意义是什么？")

        assert "生命的意义" in req.query

    def test_emoji_in_query(self):
        """Test emoji characters in query."""
        from app.schemas import QueryRequest

        req = QueryRequest(query="What does this emoji mean? 🤔")

        assert "🤔" in req.query

    def test_newlines_in_query(self):
        """Test newlines in query are preserved (after strip)."""
        from app.schemas import QueryRequest

        req = QueryRequest(query="First line\nSecond line\nThird line")

        assert "\n" in req.query

    def test_payload_accepts_dict(self):
        """Test that payload field accepts dict."""
        from app.schemas import CandidateResult

        result = CandidateResult(
            model="test",
            payload={"key": "value", "nested": {"a": 1}},
        )

        assert result.payload["key"] == "value"

    def test_payload_accepts_string(self):
        """Test that payload field accepts string."""
        from app.schemas import CandidateResult

        result = CandidateResult(model="test", payload='{"json": "string"}')

        assert isinstance(result.payload, str)

    def test_very_long_answer(self):
        """Test that very long answers are accepted."""
        from app.schemas import QueryResponse, ResponseDiagnostics, RouteDecision

        long_answer = "x" * 100000  # 100K chars

        response = QueryResponse(
            answer=long_answer,
            model="test",
            diagnostics=ResponseDiagnostics(route=RouteDecision(chosen_model="test")),
        )

        assert len(response.answer) == 100000
