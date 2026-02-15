# -*- coding: utf-8 -*-
"""
test_integration_pipeline.py — Integration tests for the full request pipeline
------------------------------------------------------------------------------
Tests end-to-end request flow through the API including:
- /query endpoint
- Correlation ID propagation
- Response structure validation
- Error handling at API level
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    # Mock external dependencies before importing the app
    with patch("app.main.get_redis", return_value=MagicMock()), \
         patch("app.main.init_vectorstore"), \
         patch("app.main._ensure_model_metrics_table"), \
         patch("app.main.preload_ollama_models", new_callable=AsyncMock), \
         patch("app.main.vs_add_document", new_callable=AsyncMock), \
         patch("app.main.route_and_answer", new_callable=AsyncMock) as mock_route:

        # Default mock response
        mock_route.return_value = {
            "answer": "Test answer",
            "model": "ollama/phi4:latest",
            "modality": "text",
            "image_output_b64": None,
            "latency_s": 1.0,
            "cost_per_1k": 0.001,
            "metadata": {
                "raw_payload": "{}",
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "load_time": 0.5
            },
            "route": {
                "chosen_model": "ollama/phi4:latest",
                "modality_selected": "text",
                "is_multimodal_route": False,
                "objectives": {"latency": 1.0, "cost": 0.001, "uncertainty": 0.5},
                "pareto_front": [],
                "explanation": "Selected via bandit"
            },
            "candidates": []
        }

        from app.main import app
        yield TestClient(app), mock_route


class TestQueryEndpoint:
    """Tests for the /query endpoint."""

    def test_query_returns_valid_response(self, client):
        """Basic query should return a valid response structure."""
        test_client, mock_route = client

        response = test_client.post(
            "/query",
            json={"query": "What is Python?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "model" in data
        assert "route" in data

    def test_query_includes_correlation_id(self, client):
        """Response should include a correlation ID."""
        test_client, mock_route = client

        response = test_client.post(
            "/query",
            json={"query": "What is Python?"}
        )

        assert response.status_code == 200

        # Check response header
        assert "X-Correlation-ID" in response.headers

        # Check response body
        data = response.json()
        assert "correlation_id" in data or data.get("correlation_id") is not None

    def test_query_propagates_incoming_correlation_id(self, client):
        """When client sends X-Correlation-ID, it should be used."""
        test_client, mock_route = client
        custom_id = "test-correlation-12345"

        response = test_client.post(
            "/query",
            json={"query": "Test query"},
            headers={"X-Correlation-ID": custom_id}
        )

        assert response.status_code == 200
        assert response.headers.get("X-Correlation-ID") == custom_id

    def test_empty_query_returns_400(self, client):
        """Empty query should return 400 or 422 error."""
        test_client, mock_route = client

        response = test_client.post(
            "/query",
            json={"query": "   "}
        )

        # Pydantic validation errors return 422 Unprocessable Entity
        assert response.status_code in [400, 422]

    def test_query_with_image_sets_vision_modality(self, client):
        """Query with image should use vision modality."""
        test_client, mock_route = client

        mock_route.return_value = {
            "answer": "Vision answer",
            "model": "ollama/llava:latest",
            "modality": "vision",
            "image_output_b64": None,
            "latency_s": 2.0,
            "cost_per_1k": 0.005,
            "metadata": {"raw_payload": "{}", "prompt_tokens": 100, "completion_tokens": 50, "load_time": 0.0},
            "route": {
                "chosen_model": "ollama/llava:latest",
                "modality_selected": "vision",
                "is_multimodal_route": True,
                "objectives": {},
                "pareto_front": [],
                "explanation": ""
            },
            "candidates": []
        }

        response = test_client.post(
            "/query",
            json={
                "query": "Describe this image",
                "image_b64": "base64encodedimage"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["modality"] == "vision"

    def test_query_with_rag_enabled(self, client):
        """Query with RAG enabled should work correctly."""
        test_client, mock_route = client

        response = test_client.post(
            "/query",
            json={
                "query": "What is NSGA-II?",
                "enable_rag_for_answer": True
            }
        )

        assert response.status_code == 200
        # Verify route_and_answer was called with use_rag=True
        call_kwargs = mock_route.call_args[1]
        assert call_kwargs.get("use_rag") == True


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_ok(self, client):
        """Health endpoint should return status and components."""
        test_client, _ = client

        response = test_client.get("/health")

        # May be degraded in test environment
        assert response.status_code in [200, 503]
        data = response.json()
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert "timestamp" in data
        assert "components" in data


class TestMetricsEndpoint:
    """Tests for the /metrics endpoint."""

    def test_metrics_returns_prometheus_format(self, client):
        """Metrics endpoint should return Prometheus format."""
        test_client, _ = client

        response = test_client.get("/metrics")

        assert response.status_code == 200
        # Prometheus metrics are text-based
        assert "text/plain" in response.headers.get("content-type", "") or \
               "text/plain" in str(response.headers)


class TestErrorHandling:
    """Tests for error handling in the API."""

    def test_router_error_returns_500(self, client):
        """When router fails, should return 500."""
        test_client, mock_route = client
        mock_route.side_effect = Exception("Internal router error")

        response = test_client.post(
            "/query",
            json={"query": "Test query"}
        )

        assert response.status_code == 500

    def test_missing_query_field_returns_422(self, client):
        """Missing required field should return 422."""
        test_client, _ = client

        response = test_client.post(
            "/query",
            json={}
        )

        assert response.status_code == 422


class TestResponseStructure:
    """Tests for validating response structure."""

    def test_response_has_all_required_fields(self, client):
        """Response should have all required fields from QueryResponse schema."""
        test_client, mock_route = client

        response = test_client.post(
            "/query",
            json={"query": "Test"}
        )

        assert response.status_code == 200
        data = response.json()

        # Required fields
        assert "answer" in data
        assert "model" in data
        assert "modality" in data
        assert "route" in data

        # Route structure
        route = data["route"]
        assert "chosen_model" in route
        assert "objectives" in route

    def test_candidates_is_list(self, client):
        """Candidates field should be a list."""
        test_client, mock_route = client

        response = test_client.post(
            "/query",
            json={"query": "Test"}
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data.get("candidates", []), list)
