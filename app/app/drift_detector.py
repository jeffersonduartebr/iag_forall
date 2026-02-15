# -*- coding: utf-8 -*-
"""
drift_detector.py — Query Distribution Drift Detection
-------------------------------------------------------
Detects when the distribution of incoming queries shifts
significantly from the historical baseline.

Features:
- Maintains baseline embedding centroid from historical queries
- Computes running centroid from recent queries
- Signals drift when cosine distance exceeds threshold
- Exposes Prometheus metrics for monitoring
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from typing import Optional, Dict, Any, List

import numpy as np

from .settings_dynamic import settings
from .utils.redis_client import get_redis
from .observability import QUERY_DRIFT_SCORE, QUERY_DRIFT_DETECTED

logger = logging.getLogger(__name__)

# Redis keys
REDIS_KEY_BASELINE_CENTROID = "drift:baseline_centroid"
REDIS_KEY_RECENT_EMBEDDINGS = "drift:recent_embeddings"
REDIS_KEY_DRIFT_STATS = "drift:stats"


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine distance between two vectors.

    Returns a value in [0, 2] where:
    - 0 = identical direction
    - 1 = orthogonal
    - 2 = opposite direction
    """
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0 or b_norm == 0:
        return 1.0  # Undefined, return neutral

    similarity = np.dot(a, b) / (a_norm * b_norm)
    return 1.0 - similarity


class QueryDriftDetector:
    """
    Detects drift in query distribution using embedding centroids.

    Maintains a rolling window of recent query embeddings and compares
    their centroid to the historical baseline.
    """

    _instance: Optional["QueryDriftDetector"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "QueryDriftDetector":
        """Executa new."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Inicializa estado interno necessário para uso da classe."""
        if self._initialized:
            return

        self._window_size = settings.DRIFT_WINDOW_SIZE
        self._threshold = settings.DRIFT_THRESHOLD

        # In-memory storage for recent embeddings
        self._recent_embeddings: deque = deque(maxlen=self._window_size)

        # Baseline centroid (loaded from Redis or computed)
        self._baseline_centroid: Optional[np.ndarray] = None
        self._baseline_sample_count = 0

        # Statistics
        self._total_queries = 0
        self._drift_events = 0
        self._last_drift_score = 0.0

        # Load persisted state
        self._load_state()
        self._initialized = True

    def _get_redis(self):
        """Executa get redis."""
        return get_redis()

    def _load_state(self) -> None:
        """Load persisted baseline centroid from Redis."""
        rds = self._get_redis()
        if not rds:
            return

        try:
            # Load baseline centroid
            raw_centroid = rds.get(REDIS_KEY_BASELINE_CENTROID)
            if raw_centroid:
                data = json.loads(raw_centroid)
                self._baseline_centroid = np.array(data["centroid"])
                self._baseline_sample_count = data.get("sample_count", 0)
                logger.info(
                    f"[DriftDetector] Loaded baseline centroid "
                    f"(dim={len(self._baseline_centroid)}, samples={self._baseline_sample_count})"
                )

            # Load stats
            raw_stats = rds.get(REDIS_KEY_DRIFT_STATS)
            if raw_stats:
                stats = json.loads(raw_stats)
                self._total_queries = stats.get("total_queries", 0)
                self._drift_events = stats.get("drift_events", 0)
        except Exception as e:
            logger.warning(f"[DriftDetector] Failed to load state: {e}")

    def _save_baseline(self) -> None:
        """Persist baseline centroid to Redis."""
        rds = self._get_redis()
        if not rds or self._baseline_centroid is None:
            return

        try:
            data = {
                "centroid": self._baseline_centroid.tolist(),
                "sample_count": self._baseline_sample_count,
                "updated_at": time.time(),
            }
            rds.set(REDIS_KEY_BASELINE_CENTROID, json.dumps(data))
        except Exception as e:
            logger.warning(f"[DriftDetector] Failed to save baseline: {e}")

    def _save_stats(self) -> None:
        """Persist drift statistics to Redis."""
        rds = self._get_redis()
        if not rds:
            return

        try:
            stats = {
                "total_queries": self._total_queries,
                "drift_events": self._drift_events,
                "last_drift_score": self._last_drift_score,
                "updated_at": time.time(),
            }
            rds.set(REDIS_KEY_DRIFT_STATS, json.dumps(stats))
        except Exception as e:
            logger.warning(f"[DriftDetector] Failed to save stats: {e}")

    def _compute_centroid(self, embeddings: List[np.ndarray]) -> Optional[np.ndarray]:
        """Compute centroid of a list of embeddings."""
        if not embeddings:
            return None
        return np.mean(embeddings, axis=0)

    def record_query(self, embedding: List[float]) -> Dict[str, Any]:
        """
        Record a query embedding and check for drift.

        Args:
            embedding: Query embedding vector

        Returns:
            Dict with drift detection results
        """
        emb_array = np.array(embedding)
        self._recent_embeddings.append(emb_array)
        self._total_queries += 1

        # Refresh settings in case they changed
        self._threshold = settings.DRIFT_THRESHOLD
        self._window_size = settings.DRIFT_WINDOW_SIZE

        # Update baseline if we don't have one or it's very small
        if self._baseline_centroid is None or self._baseline_sample_count < 50:
            self._update_baseline(emb_array)
            return {
                "drift_detected": False,
                "drift_score": 0.0,
                "message": "Building baseline",
                "baseline_samples": self._baseline_sample_count,
            }

        # Check for drift only when we have enough recent samples
        if len(self._recent_embeddings) < min(10, self._window_size // 2):
            return {
                "drift_detected": False,
                "drift_score": 0.0,
                "message": "Collecting samples",
                "recent_samples": len(self._recent_embeddings),
            }

        # Compute drift
        result = self._check_drift()

        # Periodically save stats
        if self._total_queries % 50 == 0:
            self._save_stats()

        return result

    def _update_baseline(self, new_embedding: np.ndarray) -> None:
        """Update baseline centroid with a new embedding using running mean."""
        if self._baseline_centroid is None:
            self._baseline_centroid = new_embedding.copy()
            self._baseline_sample_count = 1
        else:
            # Running mean update
            n = self._baseline_sample_count
            self._baseline_centroid = (self._baseline_centroid * n + new_embedding) / (n + 1)
            self._baseline_sample_count += 1

        # Periodically save baseline
        if self._baseline_sample_count % 100 == 0:
            self._save_baseline()

    def _check_drift(self) -> Dict[str, Any]:
        """
        Check if current query distribution has drifted from baseline.

        Returns dict with drift detection results.
        """
        if self._baseline_centroid is None:
            return {"drift_detected": False, "drift_score": 0.0, "message": "No baseline"}

        # Compute recent centroid
        recent_centroid = self._compute_centroid(list(self._recent_embeddings))
        if recent_centroid is None:
            return {"drift_detected": False, "drift_score": 0.0, "message": "No recent data"}

        # Compute drift score
        drift_score = cosine_distance(self._baseline_centroid, recent_centroid)
        self._last_drift_score = drift_score

        # Update Prometheus metric
        QUERY_DRIFT_SCORE.set(drift_score)

        # Check if drift exceeds threshold
        drift_detected = drift_score > self._threshold

        if drift_detected:
            self._drift_events += 1
            QUERY_DRIFT_DETECTED.inc()
            logger.warning(
                f"[DriftDetector] Query drift detected! "
                f"Score: {drift_score:.4f} (threshold: {self._threshold})"
            )

        return {
            "drift_detected": drift_detected,
            "drift_score": round(drift_score, 4),
            "threshold": self._threshold,
            "recent_samples": len(self._recent_embeddings),
            "baseline_samples": self._baseline_sample_count,
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current drift detector status."""
        return {
            "total_queries": self._total_queries,
            "drift_events": self._drift_events,
            "last_drift_score": round(self._last_drift_score, 4),
            "threshold": self._threshold,
            "window_size": self._window_size,
            "recent_samples": len(self._recent_embeddings),
            "baseline_samples": self._baseline_sample_count,
            "baseline_ready": self._baseline_centroid is not None and self._baseline_sample_count >= 50,
            "drift_rate": (
                round(self._drift_events / self._total_queries, 4)
                if self._total_queries > 0
                else 0.0
            ),
        }

    def reset_baseline(self) -> None:
        """Reset baseline centroid to rebuild from scratch."""
        self._baseline_centroid = None
        self._baseline_sample_count = 0
        self._recent_embeddings.clear()

        rds = self._get_redis()
        if rds:
            try:
                rds.delete(REDIS_KEY_BASELINE_CENTROID)
            except Exception:
                pass

        logger.info("[DriftDetector] Baseline reset")

    def force_baseline_update(self) -> Dict[str, Any]:
        """
        Force update baseline from current recent embeddings.

        Useful when drift is expected (e.g., new product launch).
        """
        if len(self._recent_embeddings) < 10:
            return {"success": False, "message": "Not enough recent samples"}

        new_baseline = self._compute_centroid(list(self._recent_embeddings))
        if new_baseline is not None:
            self._baseline_centroid = new_baseline
            self._baseline_sample_count = len(self._recent_embeddings)
            self._save_baseline()
            logger.info(f"[DriftDetector] Baseline updated from {self._baseline_sample_count} recent samples")
            return {"success": True, "baseline_samples": self._baseline_sample_count}

        return {"success": False, "message": "Failed to compute new baseline"}


def get_drift_detector() -> QueryDriftDetector:
    """Get the global drift detector instance."""
    return QueryDriftDetector()
