# -*- coding: utf-8 -*-
# Objective: Test coverage for ab testing behavior and regressions.
"""
Tests for A/B testing infrastructure.
"""

from unittest.mock import patch

import pytest


class TestVariant:
    """Test suite for Variant dataclass."""

    def test_variant_creation(self):
        """Test creating a variant."""
        from app.ab_testing import Variant

        variant = Variant(name="control", weight=0.5, config={"model": "gpt-4"})

        assert variant.name == "control"
        assert variant.weight == 0.5
        assert variant.config["model"] == "gpt-4"


class TestExperiment:
    """Test suite for Experiment dataclass."""

    def test_experiment_to_dict(self):
        """Test experiment serialization."""
        from app.ab_testing import Experiment, ExperimentStatus, Variant

        experiment = Experiment(
            id="exp_123",
            name="Test Experiment",
            description="A test",
            variants=[
                Variant(name="control", weight=0.5),
                Variant(name="treatment", weight=0.5),
            ],
            status=ExperimentStatus.RUNNING,
        )

        data = experiment.to_dict()

        assert data["id"] == "exp_123"
        assert data["name"] == "Test Experiment"
        assert len(data["variants"]) == 2
        assert data["status"] == "running"

    def test_experiment_from_dict(self):
        """Test experiment deserialization."""
        from app.ab_testing import Experiment, ExperimentStatus

        data = {
            "id": "exp_123",
            "name": "Test Experiment",
            "description": "A test",
            "variants": [
                {"name": "control", "weight": 0.5, "config": {}},
                {"name": "treatment", "weight": 0.5, "config": {}},
            ],
            "status": "running",
        }

        experiment = Experiment.from_dict(data)

        assert experiment.id == "exp_123"
        assert experiment.status == ExperimentStatus.RUNNING
        assert len(experiment.variants) == 2


class TestABTestManager:
    """Test suite for ABTestManager."""

    @pytest.fixture
    def ab_manager(self):
        """Create a fresh A/B test manager instance."""
        with patch("app.ab_testing.get_redis", return_value=None), patch("app.ab_testing.settings") as mock_settings:
            mock_settings.AB_TESTING_ENABLED = True

            from app.ab_testing import ABTestManager

            # Reset singleton before each test and keep patches active for the test lifecycle.
            ABTestManager._instance = None
            yield ABTestManager()
            ABTestManager._instance = None

    def test_hash_to_bucket_deterministic(self, ab_manager):
        """Test that hashing is deterministic."""
        bucket1 = ab_manager._hash_to_bucket("user123")
        bucket2 = ab_manager._hash_to_bucket("user123")

        assert bucket1 == bucket2

    def test_hash_to_bucket_distribution(self, ab_manager):
        """Test that hashing distributes across buckets."""
        buckets = [ab_manager._hash_to_bucket(f"user{i}") for i in range(1000)]

        # Should have reasonable distribution
        unique_buckets = len(set(buckets))
        assert unique_buckets > 100  # At least 10% of buckets used

    def test_create_experiment(self, ab_manager):
        """Test creating an experiment."""
        from app.ab_testing import ExperimentCreateRequest, ExperimentStatus

        request = ExperimentCreateRequest(
            name="Test Experiment",
            description="Testing",
            variants=[
                {"name": "control", "weight": 1.0},
                {"name": "treatment", "weight": 1.0},
            ],
        )

        experiment = ab_manager.create_experiment(request)

        assert experiment.name == "Test Experiment"
        assert experiment.status == ExperimentStatus.DRAFT
        assert len(experiment.variants) == 2
        # Weights should be normalized
        total_weight = sum(v.weight for v in experiment.variants)
        assert total_weight == pytest.approx(1.0, abs=0.01)

    def test_start_experiment(self, ab_manager):
        """Test starting an experiment."""
        from app.ab_testing import ExperimentCreateRequest, ExperimentStatus

        request = ExperimentCreateRequest(
            name="Test",
            variants=[
                {"name": "control", "weight": 0.5},
                {"name": "treatment", "weight": 0.5},
            ],
        )

        experiment = ab_manager.create_experiment(request)
        started = ab_manager.start_experiment(experiment.id)

        assert started.status == ExperimentStatus.RUNNING
        assert started.started_at is not None

    def test_pause_experiment(self, ab_manager):
        """Test pausing an experiment."""
        from app.ab_testing import ExperimentCreateRequest, ExperimentStatus

        request = ExperimentCreateRequest(
            name="Test",
            variants=[
                {"name": "control", "weight": 0.5},
                {"name": "treatment", "weight": 0.5},
            ],
        )

        experiment = ab_manager.create_experiment(request)
        ab_manager.start_experiment(experiment.id)
        paused = ab_manager.pause_experiment(experiment.id)

        assert paused.status == ExperimentStatus.PAUSED

    def test_complete_experiment(self, ab_manager):
        """Test completing an experiment."""
        from app.ab_testing import ExperimentCreateRequest, ExperimentStatus

        request = ExperimentCreateRequest(
            name="Test",
            variants=[
                {"name": "control", "weight": 0.5},
                {"name": "treatment", "weight": 0.5},
            ],
        )

        experiment = ab_manager.create_experiment(request)
        ab_manager.start_experiment(experiment.id)
        completed = ab_manager.complete_experiment(experiment.id)

        assert completed.status == ExperimentStatus.COMPLETED
        assert completed.ended_at is not None

    def test_get_assignment_not_running(self, ab_manager):
        """Test that assignment returns None for non-running experiment."""
        from app.ab_testing import ExperimentCreateRequest

        request = ExperimentCreateRequest(
            name="Test",
            variants=[
                {"name": "control", "weight": 0.5},
                {"name": "treatment", "weight": 0.5},
            ],
        )

        experiment = ab_manager.create_experiment(request)
        # Don't start it

        result = ab_manager.get_assignment(experiment.id, "user123")

        assert result is None

    def test_get_assignment_consistent(self, ab_manager):
        """Test that assignment is consistent for same user."""
        from app.ab_testing import ExperimentCreateRequest

        with patch("app.ab_testing.AB_EXPERIMENT_ASSIGNMENTS"):
            request = ExperimentCreateRequest(
                name="Test",
                variants=[
                    {"name": "control", "weight": 0.5},
                    {"name": "treatment", "weight": 0.5},
                ],
            )

            experiment = ab_manager.create_experiment(request)
            ab_manager.start_experiment(experiment.id)

            result1 = ab_manager.get_assignment(experiment.id, "user123")
            result2 = ab_manager.get_assignment(experiment.id, "user123")

            assert result1 == result2

    def test_list_experiments_filter_by_status(self, ab_manager):
        """Test listing experiments filtered by status."""
        from app.ab_testing import ExperimentCreateRequest, ExperimentStatus

        # Create multiple experiments
        for i in range(3):
            request = ExperimentCreateRequest(
                name=f"Test {i}",
                variants=[{"name": "a", "weight": 1.0}],
            )
            ab_manager.create_experiment(request)

        # Start one
        experiments = ab_manager.list_experiments()
        if experiments:
            ab_manager.start_experiment(experiments[0].id)

        draft_experiments = ab_manager.list_experiments(status=ExperimentStatus.DRAFT)
        running_experiments = ab_manager.list_experiments(status=ExperimentStatus.RUNNING)

        assert len(draft_experiments) >= 2
        assert len(running_experiments) >= 1

    def test_delete_experiment(self, ab_manager):
        """Test deleting an experiment."""
        from app.ab_testing import ExperimentCreateRequest

        request = ExperimentCreateRequest(
            name="Test",
            variants=[{"name": "a", "weight": 1.0}],
        )

        experiment = ab_manager.create_experiment(request)
        exp_id = experiment.id

        success = ab_manager.delete_experiment(exp_id)

        assert success is True
        assert ab_manager.get_experiment(exp_id) is None

    def test_delete_nonexistent_experiment(self, ab_manager):
        """Test deleting non-existent experiment returns False."""
        success = ab_manager.delete_experiment("nonexistent_id")
        assert success is False


class TestGetABTestManager:
    """Test the get_ab_test_manager factory function."""

    def test_returns_singleton(self):
        """Test that get_ab_test_manager returns singleton instance."""
        with patch("app.ab_testing.get_redis", return_value=None):
            with patch("app.ab_testing.settings") as mock_settings:
                mock_settings.AB_TESTING_ENABLED = True

                from app.ab_testing import ABTestManager, get_ab_test_manager
                ABTestManager._instance = None

                manager1 = get_ab_test_manager()
                manager2 = get_ab_test_manager()

                assert manager1 is manager2


class TestExperimentResults:
    """Results round trip through Redis sorted sets (fakeredis)."""

    @pytest.fixture
    def manager_with_redis(self):
        import fakeredis

        rds = fakeredis.FakeRedis()
        with patch("app.ab_testing.get_redis", return_value=rds), patch("app.ab_testing.settings") as mock_settings:
            mock_settings.AB_TESTING_ENABLED = True
            from app.ab_testing import ABTestManager, Experiment

            ABTestManager._instance = None
            manager = ABTestManager()
            manager._experiments["e1"] = Experiment.from_dict(
                {
                    "id": "e1",
                    "name": "rota",
                    "variants": [{"name": "control", "weight": 0.5}, {"name": "treatment", "weight": 0.5}],
                    "status": "running",
                }
            )
            yield manager, rds
            ABTestManager._instance = None

    def test_summarize_scores(self):
        from app.ab_testing import summarize_scores

        assert summarize_scores([]) == {"count": 0}
        assert summarize_scores([1.0, 2.0, 3.0, 4.0]) == {
            "count": 4,
            "mean": 2.5,
            "std": 1.118,
            "median": 2.5,
            "p95": 3.85,
        }

    def test_results_per_variant_and_metric(self, manager_with_redis):
        manager, _ = manager_with_redis
        for value in (7.0, 9.0):
            manager.record_result("e1", "treatment", "quality", value)
        manager.record_result("e1", "control", "latency", 1.5)

        out = manager.get_experiment_results("e1")
        assert out["experiment"]["id"] == "e1"
        assert out["variants"]["treatment"]["quality"]["mean"] == 8.0
        assert out["variants"]["treatment"]["latency"] == {"count": 0}
        assert out["variants"]["control"]["latency"]["count"] == 1
        assert manager.get_experiment_results("nope") == {"error": "Experiment not found"}

    def test_results_degrade_per_metric_on_redis_errors(self, manager_with_redis):
        manager, rds = manager_with_redis
        rds.set("ab:results:e1:control:quality", "not-a-zset")
        out = manager.get_experiment_results("e1")
        assert "error" in out["variants"]["control"]["quality"]
        assert out["variants"]["control"]["cost"] == {"count": 0}

        with patch.object(manager, "_get_redis", return_value=None):
            assert manager.get_experiment_results("e1")["variants"] == {}
