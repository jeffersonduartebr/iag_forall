# -*- coding: utf-8 -*-
"""
ab_testing.py — A/B Testing Infrastructure
-------------------------------------------
Provides A/B testing capabilities for the LLM router.

Features:
- Experiment definition with variants and weights
- Consistent user assignment via hashing
- Result tracking and aggregation
- Statistical significance calculation
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

from pydantic import BaseModel, Field

from .settings_dynamic import settings
from .utils.redis_client import get_redis
from .observability import AB_EXPERIMENT_ASSIGNMENTS

logger = logging.getLogger(__name__)

# Redis keys
REDIS_KEY_EXPERIMENTS = "ab:experiments"
REDIS_KEY_ASSIGNMENTS = "ab:assignments"
REDIS_KEY_RESULTS = "ab:results"


class ExperimentStatus(str, Enum):
    """Experiment status."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"


@dataclass
class Variant:
    """A/B test variant."""
    name: str
    weight: float  # Weight for assignment (0.0 - 1.0)
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Experiment:
    """A/B test experiment."""
    id: str
    name: str
    description: str
    variants: List[Variant]
    status: ExperimentStatus = ExperimentStatus.DRAFT
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    target_samples: int = 1000

    def to_dict(self) -> Dict[str, Any]:
        """Resumo do comportamento desta função.

        Returns:
            Valor retornado pela função.
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "variants": [{"name": v.name, "weight": v.weight, "config": v.config} for v in self.variants],
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "target_samples": self.target_samples,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Experiment":
        """Resumo do comportamento desta função.

        Args:
            data: Parâmetro de entrada.

        Returns:
            Valor retornado pela função.
        """
        variants = [
            Variant(name=v["name"], weight=v["weight"], config=v.get("config", {}))
            for v in data.get("variants", [])
        ]
        return Experiment(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            variants=variants,
            status=ExperimentStatus(data.get("status", "draft")),
            created_at=data.get("created_at", time.time()),
            started_at=data.get("started_at"),
            ended_at=data.get("ended_at"),
            target_samples=data.get("target_samples", 1000),
        )


class ExperimentCreateRequest(BaseModel):
    """Request to create a new experiment."""
    name: str = Field(..., description="Experiment name")
    description: str = Field(default="", description="Experiment description")
    variants: List[Dict[str, Any]] = Field(..., description="List of variants with name, weight, and config")
    target_samples: int = Field(default=1000, description="Target number of samples")


class ABTestManager:
    """
    Manages A/B testing experiments.

    Provides consistent variant assignment using hash-based bucketing.
    """

    _instance: Optional["ABTestManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ABTestManager":
        """Resumo do comportamento desta função.

        Returns:
            Valor retornado pela função.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Resumo do comportamento desta função.

        Returns:
            Valor retornado pela função.
        """
        if self._initialized:
            return

        self._experiments: Dict[str, Experiment] = {}
        self._load_experiments()
        self._initialized = True

    def _get_redis(self):
        """Resumo do comportamento desta função.

        Returns:
            Valor retornado pela função.
        """
        return get_redis()

    def _load_experiments(self) -> None:
        """Load experiments from Redis."""
        rds = self._get_redis()
        if not rds:
            return

        try:
            raw = rds.get(REDIS_KEY_EXPERIMENTS)
            if raw:
                data = json.loads(raw)
                for exp_data in data:
                    exp = Experiment.from_dict(exp_data)
                    self._experiments[exp.id] = exp
                logger.info(f"[ABTesting] Loaded {len(self._experiments)} experiments")
        except Exception as e:
            logger.warning(f"[ABTesting] Failed to load experiments: {e}")

    def _save_experiments(self) -> None:
        """Save experiments to Redis."""
        rds = self._get_redis()
        if not rds:
            return

        try:
            data = [exp.to_dict() for exp in self._experiments.values()]
            rds.set(REDIS_KEY_EXPERIMENTS, json.dumps(data))
        except Exception as e:
            logger.warning(f"[ABTesting] Failed to save experiments: {e}")

    def _hash_to_bucket(self, key: str, num_buckets: int = 1000) -> int:
        """Hash a key to a bucket number for consistent assignment."""
        hash_bytes = hashlib.md5(key.encode()).digest()
        hash_int = int.from_bytes(hash_bytes[:4], byteorder="big")
        return hash_int % num_buckets

    def _assign_variant(self, experiment: Experiment, user_key: str) -> Variant:
        """
        Assign a user to a variant using consistent hashing.

        Args:
            experiment: The experiment
            user_key: User identifier for consistent assignment

        Returns:
            Assigned variant
        """
        # Hash user key for consistent assignment
        bucket = self._hash_to_bucket(f"{experiment.id}:{user_key}")
        bucket_ratio = bucket / 1000.0

        # Assign based on cumulative weights
        cumulative = 0.0
        for variant in experiment.variants:
            cumulative += variant.weight
            if bucket_ratio < cumulative:
                return variant

        # Fallback to last variant
        return experiment.variants[-1] if experiment.variants else Variant(name="control", weight=1.0)

    def create_experiment(self, request: ExperimentCreateRequest) -> Experiment:
        """Create a new experiment."""
        if not settings.AB_TESTING_ENABLED:
            raise RuntimeError("A/B testing is disabled")

        # Generate ID
        exp_id = f"exp_{int(time.time())}_{len(self._experiments)}"

        # Normalize weights
        total_weight = sum(v.get("weight", 1.0) for v in request.variants)
        variants = [
            Variant(
                name=v["name"],
                weight=v.get("weight", 1.0) / total_weight,
                config=v.get("config", {}),
            )
            for v in request.variants
        ]

        experiment = Experiment(
            id=exp_id,
            name=request.name,
            description=request.description,
            variants=variants,
            status=ExperimentStatus.DRAFT,
            target_samples=request.target_samples,
        )

        self._experiments[exp_id] = experiment
        self._save_experiments()

        logger.info(f"[ABTesting] Created experiment: {exp_id}")
        return experiment

    def start_experiment(self, experiment_id: str) -> Experiment:
        """Start an experiment."""
        if experiment_id not in self._experiments:
            raise KeyError(f"Experiment not found: {experiment_id}")

        exp = self._experiments[experiment_id]
        exp.status = ExperimentStatus.RUNNING
        exp.started_at = time.time()

        self._save_experiments()
        logger.info(f"[ABTesting] Started experiment: {experiment_id}")
        return exp

    def pause_experiment(self, experiment_id: str) -> Experiment:
        """Pause an experiment."""
        if experiment_id not in self._experiments:
            raise KeyError(f"Experiment not found: {experiment_id}")

        exp = self._experiments[experiment_id]
        exp.status = ExperimentStatus.PAUSED

        self._save_experiments()
        logger.info(f"[ABTesting] Paused experiment: {experiment_id}")
        return exp

    def complete_experiment(self, experiment_id: str) -> Experiment:
        """Complete an experiment."""
        if experiment_id not in self._experiments:
            raise KeyError(f"Experiment not found: {experiment_id}")

        exp = self._experiments[experiment_id]
        exp.status = ExperimentStatus.COMPLETED
        exp.ended_at = time.time()

        self._save_experiments()
        logger.info(f"[ABTesting] Completed experiment: {experiment_id}")
        return exp

    def get_assignment(
        self,
        experiment_id: str,
        user_key: str,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Get variant assignment for a user.

        Args:
            experiment_id: Experiment ID
            user_key: User identifier for consistent assignment

        Returns:
            Tuple of (variant_name, variant_config) or None if experiment not running
        """
        if not settings.AB_TESTING_ENABLED:
            return None

        if experiment_id not in self._experiments:
            return None

        exp = self._experiments[experiment_id]
        if exp.status != ExperimentStatus.RUNNING:
            return None

        variant = self._assign_variant(exp, user_key)

        # Track assignment
        AB_EXPERIMENT_ASSIGNMENTS.labels(
            experiment_id=experiment_id,
            variant=variant.name,
        ).inc()

        return variant.name, variant.config

    def record_result(
        self,
        experiment_id: str,
        variant_name: str,
        metric_name: str,
        value: float,
    ) -> None:
        """
        Record a result for an experiment variant.

        Args:
            experiment_id: Experiment ID
            variant_name: Variant name
            metric_name: Metric name (e.g., "quality", "latency", "cost")
            value: Metric value
        """
        if not settings.AB_TESTING_ENABLED:
            return

        rds = self._get_redis()
        if not rds:
            return

        try:
            key = f"{REDIS_KEY_RESULTS}:{experiment_id}:{variant_name}:{metric_name}"
            # Use sorted set for percentile calculations
            rds.zadd(key, {f"{time.time()}:{value}": value})
            # Trim to last 10000 results
            rds.zremrangebyrank(key, 0, -10001)
        except Exception as e:
            logger.warning(f"[ABTesting] Failed to record result: {e}")

    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get aggregated results for an experiment.

        Returns statistics for each variant and metric.
        """
        if experiment_id not in self._experiments:
            return {"error": "Experiment not found"}

        exp = self._experiments[experiment_id]
        rds = self._get_redis()

        results = {
            "experiment": exp.to_dict(),
            "variants": {},
        }

        if not rds:
            return results

        metrics = ["quality", "latency", "cost"]

        for variant in exp.variants:
            variant_results = {}

            for metric in metrics:
                try:
                    key = f"{REDIS_KEY_RESULTS}:{experiment_id}:{variant.name}:{metric}"
                    values = rds.zrange(key, 0, -1, withscores=True)

                    if values:
                        scores = [float(v[1]) for v in values]
                        import numpy as np

                        variant_results[metric] = {
                            "count": len(scores),
                            "mean": round(float(np.mean(scores)), 4),
                            "std": round(float(np.std(scores)), 4),
                            "median": round(float(np.median(scores)), 4),
                            "p95": round(float(np.percentile(scores, 95)), 4),
                        }
                    else:
                        variant_results[metric] = {"count": 0}
                except Exception as e:
                    logger.warning(f"[ABTesting] Failed to get results for {variant.name}/{metric}: {e}")
                    variant_results[metric] = {"error": str(e)}

            results["variants"][variant.name] = variant_results

        return results

    def list_experiments(self, status: Optional[ExperimentStatus] = None) -> List[Experiment]:
        """List all experiments, optionally filtered by status."""
        experiments = list(self._experiments.values())

        if status:
            experiments = [e for e in experiments if e.status == status]

        return experiments

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get a specific experiment."""
        return self._experiments.get(experiment_id)

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment."""
        if experiment_id not in self._experiments:
            return False

        del self._experiments[experiment_id]
        self._save_experiments()

        # Clean up results
        rds = self._get_redis()
        if rds:
            try:
                # Delete all result keys for this experiment
                pattern = f"{REDIS_KEY_RESULTS}:{experiment_id}:*"
                for key in rds.scan_iter(match=pattern):
                    rds.delete(key)
            except Exception:
                pass

        logger.info(f"[ABTesting] Deleted experiment: {experiment_id}")
        return True


def get_ab_test_manager() -> ABTestManager:
    """Get the global A/B test manager instance."""
    return ABTestManager()
