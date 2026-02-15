# -*- coding: utf-8 -*-
"""
test_providers_ensure_ollama.py — Extensive tests for _ensure_ollama_model
--------------------------------------------------------------------------
Tests the synchronous Ollama model ensuring functionality including:
- Model existence checking
- Model pulling/downloading
- Error handling (timeouts, connection errors, etc.)
- Thread safety and event loop independence
"""

import asyncio
import concurrent.futures
from unittest.mock import MagicMock, patch

import pytest
import requests


class TestEnsureOllamaModelBasic:
    """Basic functionality tests for _ensure_ollama_model."""

    def test_model_already_exists_exact_match(self):
        """Should return True when model already exists with exact name."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [{"name": "gemma3:4b"}, {"name": "phi4:latest"}]}

        with patch("requests.get", return_value=mock_response) as mock_get:
            from app.providers_async import _ensure_ollama_model

            result = _ensure_ollama_model("gemma3:4b")

            assert result is True
            mock_get.assert_called_once()
            # Should not attempt to pull since model exists

    def test_model_already_exists_partial_match(self):
        """Should return True when model name is substring of existing model."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "gemma3:4b-instruct-q4"},
            ]
        }

        with patch("requests.get", return_value=mock_response):
            from app.providers_async import _ensure_ollama_model

            result = _ensure_ollama_model("gemma3:4b")

            assert result is True

    def test_model_not_found_pull_success(self):
        """Should pull model and return True when model doesn't exist."""
        mock_tags_response = MagicMock()
        mock_tags_response.status_code = 200
        mock_tags_response.json.return_value = {"models": []}

        mock_pull_response = MagicMock()
        mock_pull_response.status_code = 200
        mock_pull_response.iter_lines.return_value = iter([b'{"status": "downloading"}'])

        with patch("requests.get", return_value=mock_tags_response):
            with patch("requests.post", return_value=mock_pull_response) as mock_post:
                from app.providers_async import _ensure_ollama_model

                result = _ensure_ollama_model("new-model:latest")

                assert result is True
                mock_post.assert_called_once()

    def test_model_not_found_pull_failure(self):
        """Should return False when pull fails."""
        mock_tags_response = MagicMock()
        mock_tags_response.status_code = 200
        mock_tags_response.json.return_value = {"models": []}

        mock_pull_response = MagicMock()
        mock_pull_response.status_code = 404

        with patch("requests.get", return_value=mock_tags_response):
            with patch("requests.post", return_value=mock_pull_response):
                from app.providers_async import _ensure_ollama_model

                result = _ensure_ollama_model("nonexistent-model:latest")

                assert result is False

    def test_empty_models_list(self):
        """Should handle empty models list and attempt pull."""
        mock_tags_response = MagicMock()
        mock_tags_response.status_code = 200
        mock_tags_response.json.return_value = {"models": []}

        mock_pull_response = MagicMock()
        mock_pull_response.status_code = 200
        mock_pull_response.iter_lines.return_value = iter([])

        with patch("requests.get", return_value=mock_tags_response):
            with patch("requests.post", return_value=mock_pull_response):
                from app.providers_async import _ensure_ollama_model

                result = _ensure_ollama_model("model:tag")

                assert result is True


class TestEnsureOllamaModelErrorHandling:
    """Error handling tests for _ensure_ollama_model."""

    def test_tags_endpoint_timeout(self):
        """Should return False on tags endpoint timeout."""
        with patch("requests.get", side_effect=requests.exceptions.Timeout("Connection timed out")):
            from app.providers_async import _ensure_ollama_model

            result = _ensure_ollama_model("model:tag")

            assert result is False

    def test_tags_endpoint_connection_error(self):
        """Should return False on connection error."""
        with patch("requests.get", side_effect=requests.exceptions.ConnectionError("Connection refused")):
            from app.providers_async import _ensure_ollama_model

            result = _ensure_ollama_model("model:tag")

            assert result is False

    def test_pull_endpoint_timeout(self):
        """Should return False on pull endpoint timeout."""
        mock_tags_response = MagicMock()
        mock_tags_response.status_code = 200
        mock_tags_response.json.return_value = {"models": []}

        with patch("requests.get", return_value=mock_tags_response):
            with patch("requests.post", side_effect=requests.exceptions.Timeout("Pull timed out")):
                from app.providers_async import _ensure_ollama_model

                result = _ensure_ollama_model("model:tag")

                assert result is False

    def test_tags_endpoint_non_200_status(self):
        """Should attempt pull even if tags endpoint returns non-200."""
        mock_tags_response = MagicMock()
        mock_tags_response.status_code = 500

        mock_pull_response = MagicMock()
        mock_pull_response.status_code = 200
        mock_pull_response.iter_lines.return_value = iter([])

        with patch("requests.get", return_value=mock_tags_response):
            with patch("requests.post", return_value=mock_pull_response):
                from app.providers_async import _ensure_ollama_model

                result = _ensure_ollama_model("model:tag")

                assert result is True

    def test_malformed_json_response(self):
        """Should handle malformed JSON gracefully."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")

        with patch("requests.get", return_value=mock_response):
            from app.providers_async import _ensure_ollama_model

            result = _ensure_ollama_model("model:tag")

            assert result is False

    def test_missing_models_key_in_response(self):
        """Should handle missing 'models' key in response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}  # Missing 'models' key

        mock_pull_response = MagicMock()
        mock_pull_response.status_code = 200
        mock_pull_response.iter_lines.return_value = iter([])

        with patch("requests.get", return_value=mock_response):
            with patch("requests.post", return_value=mock_pull_response):
                from app.providers_async import _ensure_ollama_model

                result = _ensure_ollama_model("model:tag")

                assert result is True  # Should attempt pull

    def test_model_without_name_field(self):
        """Should handle models without 'name' field."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"size": 1234},  # Missing 'name'
                {"name": "other-model:tag"},
            ]
        }

        mock_pull_response = MagicMock()
        mock_pull_response.status_code = 200
        mock_pull_response.iter_lines.return_value = iter([])

        with patch("requests.get", return_value=mock_response):
            with patch("requests.post", return_value=mock_pull_response):
                from app.providers_async import _ensure_ollama_model

                result = _ensure_ollama_model("new-model:tag")

                assert result is True

    def test_generic_exception(self):
        """Should handle unexpected exceptions gracefully."""
        with patch("requests.get", side_effect=RuntimeError("Unexpected error")):
            from app.providers_async import _ensure_ollama_model

            result = _ensure_ollama_model("model:tag")

            assert result is False


class TestEnsureOllamaModelThreadSafety:
    """Thread safety and event loop independence tests."""

    def test_runs_in_thread_without_event_loop(self):
        """Should work correctly when called from a thread without event loop."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [{"name": "test-model:tag"}]}

        with patch("requests.get", return_value=mock_response):
            from app.providers_async import _ensure_ollama_model

            # Run in a ThreadPoolExecutor (same as asyncio.to_thread)
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_ensure_ollama_model, "test-model:tag")
                result = future.result(timeout=5)

            assert result is True

    def test_runs_in_multiple_threads_concurrently(self):
        """Should work correctly when called from multiple threads."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [{"name": "model:tag"}]}

        with patch("requests.get", return_value=mock_response):
            from app.providers_async import _ensure_ollama_model

            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(_ensure_ollama_model, f"model-{i}:tag") for i in range(5)]
                results = [f.result(timeout=5) for f in futures]

            # All should complete (model found or pull attempted)
            assert len(results) == 5

    @pytest.mark.asyncio
    async def test_works_with_asyncio_to_thread(self):
        """Should work correctly when called via asyncio.to_thread."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [{"name": "async-test:tag"}]}

        with patch("requests.get", return_value=mock_response):
            from app.providers_async import _ensure_ollama_model

            # This is exactly how it's called in router_core.py
            result = await asyncio.to_thread(_ensure_ollama_model, "async-test:tag")

            assert result is True

    @pytest.mark.asyncio
    async def test_fire_and_forget_pattern(self):
        """Should work with fire-and-forget pattern used in router_core.py."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [{"name": "fire-forget:tag"}]}

        with patch("requests.get", return_value=mock_response):
            from app.providers_async import _ensure_ollama_model

            # Simulate the pattern from router_core.py:517
            task = asyncio.create_task(asyncio.to_thread(_ensure_ollama_model, "fire-forget:tag"))

            # Wait for completion
            result = await task

            assert result is True


class TestEnsureOllamaModelIntegration:
    """Integration tests with router_core patterns."""

    @pytest.mark.asyncio
    async def test_ollama_model_prefix_handling(self):
        """Should handle model names with 'ollama/' prefix stripped."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [{"name": "gemma3:4b"}]}

        with patch("requests.get", return_value=mock_response):
            from app.providers_async import _ensure_ollama_model

            # In router_core.py, the prefix is stripped before calling
            chosen = "ollama/gemma3:4b"
            model_name = chosen.replace("ollama/", "")

            result = await asyncio.to_thread(_ensure_ollama_model, model_name)

            assert result is True

    @pytest.mark.asyncio
    async def test_multiple_concurrent_ensures(self):
        """Should handle multiple concurrent ensure calls."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [{"name": "shared-model:tag"}]}

        with patch("requests.get", return_value=mock_response):
            from app.providers_async import _ensure_ollama_model

            # Simulate multiple concurrent requests for same/different models
            tasks = [
                asyncio.create_task(asyncio.to_thread(_ensure_ollama_model, "model-a:tag")),
                asyncio.create_task(asyncio.to_thread(_ensure_ollama_model, "model-b:tag")),
                asyncio.create_task(asyncio.to_thread(_ensure_ollama_model, "shared-model:tag")),
            ]

            results = await asyncio.gather(*tasks)

            assert len(results) == 3


class TestEnsureOllamaModelLogging:
    """Tests for logging behavior."""

    def test_logs_download_on_pull(self):
        """Should log when downloading a new model."""
        mock_tags_response = MagicMock()
        mock_tags_response.status_code = 200
        mock_tags_response.json.return_value = {"models": []}

        mock_pull_response = MagicMock()
        mock_pull_response.status_code = 200
        mock_pull_response.iter_lines.return_value = iter([])

        with patch("requests.get", return_value=mock_tags_response):
            with patch("requests.post", return_value=mock_pull_response):
                with patch("app.providers_async.logger") as mock_logger:
                    from app.providers_async import _ensure_ollama_model

                    _ensure_ollama_model("new-model:tag")

                    # Verify download was logged
                    mock_logger.info.assert_any_call("[ensure_ollama_model] Downloading model: new-model:tag")

    def test_logs_warning_on_failure(self):
        """Should log warning when operation fails."""
        with patch("requests.get", side_effect=requests.exceptions.Timeout("timeout")):
            with patch("app.providers_async.logger") as mock_logger:
                from app.providers_async import _ensure_ollama_model

                _ensure_ollama_model("model:tag")

                # Verify warning was logged
                assert mock_logger.warning.called


class TestEnsureOllamaModelAsync:
    """Tests for the async version _ensure_ollama_model_async."""

    @pytest.mark.asyncio
    async def test_async_model_exists(self):
        """Async version should return True when model exists."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [{"name": "async-model:tag"}]}

        async def mock_get(*args, **kwargs):
            """Executa mock get."""
            return mock_response

        async def mock_aenter(self):
            """Executa mock aenter."""
            return self

        async def mock_aexit(self, *args):
            """Executa mock aexit."""
            pass

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get = mock_get
            mock_client.__aenter__ = mock_aenter
            mock_client.__aexit__ = mock_aexit
            mock_client_class.return_value = mock_client

            from app.providers_async import _ensure_ollama_model_async

            result = await _ensure_ollama_model_async("async-model:tag")

            assert result is True

    @pytest.mark.asyncio
    async def test_async_pull_model(self):
        """Async version should pull model when not found."""
        mock_tags_response = MagicMock()
        mock_tags_response.status_code = 200
        mock_tags_response.json.return_value = {"models": []}

        mock_pull_response = MagicMock()
        mock_pull_response.status_code = 200

        async def mock_get(*args, **kwargs):
            """Executa mock get."""
            return mock_tags_response

        async def mock_post(*args, **kwargs):
            """Executa mock post."""
            return mock_pull_response

        async def mock_aenter(self):
            """Executa mock aenter."""
            return self

        async def mock_aexit(self, *args):
            """Executa mock aexit."""
            pass

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get = mock_get
            mock_client.post = mock_post
            mock_client.__aenter__ = mock_aenter
            mock_client.__aexit__ = mock_aexit
            mock_client_class.return_value = mock_client

            from app.providers_async import _ensure_ollama_model_async

            result = await _ensure_ollama_model_async("new-async-model:tag")

            assert result is True
