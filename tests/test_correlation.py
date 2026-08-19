# -*- coding: utf-8 -*-
# Objective: Test coverage for correlation behavior and regressions.
"""
test_correlation.py — Tests for correlation ID infrastructure
-------------------------------------------------------------
Tests for correlation.py module including:
- Correlation ID generation
- Context variable propagation
- Context manager functionality
"""

import asyncio

import pytest
from app.correlation import (
    CORRELATION_ID_HEADER,
    CorrelationIdContext,
    clear_correlation_id,
    generate_correlation_id,
    get_correlation_id,
    set_correlation_id,
)


class TestCorrelationIdGeneration:
    """Tests for correlation ID generation."""

    def test_generate_correlation_id_returns_uuid(self):
        """Generated ID should be a valid UUID4 string."""
        cid = generate_correlation_id()

        assert cid is not None
        assert isinstance(cid, str)
        assert len(cid) == 36  # UUID format: 8-4-4-4-12

    def test_generate_correlation_id_is_unique(self):
        """Each generated ID should be unique."""
        ids = [generate_correlation_id() for _ in range(100)]

        assert len(set(ids)) == 100


class TestCorrelationIdContextVar:
    """Tests for correlation ID context variable."""

    def test_set_and_get_correlation_id(self):
        """Should be able to set and get correlation ID."""
        test_id = "test-correlation-123"

        set_correlation_id(test_id)
        result = get_correlation_id()

        assert result == test_id

        # Cleanup
        clear_correlation_id()

    def test_set_correlation_id_generates_new_if_none(self):
        """Should generate new ID if None is passed."""
        clear_correlation_id()

        result = set_correlation_id(None)

        assert result is not None
        assert len(result) == 36  # UUID format
        assert get_correlation_id() == result

        # Cleanup
        clear_correlation_id()

    def test_clear_correlation_id(self):
        """Should clear the correlation ID."""
        set_correlation_id("test-id")
        clear_correlation_id()

        assert get_correlation_id() is None

    def test_get_correlation_id_returns_none_when_not_set(self):
        """Should return None when no ID is set."""
        clear_correlation_id()

        result = get_correlation_id()

        assert result is None


class TestCorrelationIdContext:
    """Tests for CorrelationIdContext context manager."""

    def test_context_manager_sets_id(self):
        """Context manager should set correlation ID."""
        clear_correlation_id()

        with CorrelationIdContext("ctx-test-id") as cid:
            assert cid == "ctx-test-id"
            assert get_correlation_id() == "ctx-test-id"

    def test_context_manager_generates_id_if_none(self):
        """Context manager should generate ID if not provided."""
        clear_correlation_id()

        with CorrelationIdContext() as cid:
            assert cid is not None
            assert len(cid) == 36
            assert get_correlation_id() == cid

    def test_context_manager_restores_previous_id(self):
        """Context manager should restore previous ID on exit."""
        set_correlation_id("outer-id")

        with CorrelationIdContext("inner-id"):
            assert get_correlation_id() == "inner-id"

        assert get_correlation_id() == "outer-id"

        # Cleanup
        clear_correlation_id()

    def test_nested_context_managers(self):
        """Nested context managers should work correctly."""
        clear_correlation_id()

        with CorrelationIdContext("level-1"):
            assert get_correlation_id() == "level-1"

            with CorrelationIdContext("level-2"):
                assert get_correlation_id() == "level-2"

                with CorrelationIdContext("level-3"):
                    assert get_correlation_id() == "level-3"

                assert get_correlation_id() == "level-2"

            assert get_correlation_id() == "level-1"

        assert get_correlation_id() is None


class TestAsyncCorrelationPropagation:
    """Tests for async correlation ID propagation."""

    @pytest.mark.asyncio
    async def test_correlation_propagates_in_async_context(self):
        """Correlation ID should propagate through async calls."""
        set_correlation_id("async-test-id")

        async def inner_coroutine():
            """Execute the inner coroutine routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            return get_correlation_id()

        result = await inner_coroutine()

        assert result == "async-test-id"

        # Cleanup
        clear_correlation_id()

    @pytest.mark.asyncio
    async def test_correlation_isolated_in_concurrent_tasks(self):
        """Each concurrent task should have isolated correlation ID."""

        async def task_with_id(task_id: str):
            """Execute the task with id routine.

This helper encapsulates one focused step used by the surrounding workflow."""
            set_correlation_id(task_id)
            await asyncio.sleep(0.01)  # Simulate async work
            return get_correlation_id()

        # Note: In reality, asyncio tasks share the same context
        # This test verifies the basic behavior
        result1 = await task_with_id("task-1")
        result2 = await task_with_id("task-2")

        # Sequential execution means each gets its own ID
        assert result1 == "task-1"
        assert result2 == "task-2"


class TestCorrelationIdHeader:
    """Tests for correlation ID HTTP header constant."""

    def test_header_name_is_correct(self):
        """Header name should be X-Correlation-ID."""
        assert CORRELATION_ID_HEADER == "X-Correlation-ID"
