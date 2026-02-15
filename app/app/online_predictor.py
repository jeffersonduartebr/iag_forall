# -*- coding: utf-8 -*-
"""
online_predictor.py — Real-Time Error Prediction Module (Logistic Regression SGD)
---------------------------------------------------------------------------------
Implements an Online Machine Learning predictor using River.
It estimates the probability of model failure based on query embeddings.

Algorithm:
- Logistic Regression with Stochastic Gradient Descent (SGD).
- Chosen for its O(d) complexity, making it ideal for high-dimensional
  embedding spaces (768d+) with minimal latency overhead.

Validation & Calibration (Phase 5 - Autonomous Behavior):
- Tracks prediction vs outcome for Brier score calculation
- Temperature scaling for probability calibration
- Prometheus metrics for monitoring prediction quality

Used by the Router to decide whether to trigger expensive LLM-as-a-Judge evaluation
(Active Learning / Importance Sampling).
"""

from __future__ import annotations

import os
import pickle
import logging
import time
from typing import List, Dict, Tuple, Optional

# River Imports (Online ML Library)
try:
    from river import linear_model
    from river import optim
    from river import preprocessing
    from river import compose
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False

logger = logging.getLogger(__name__)

# Diretório para persistência do estado dos modelos
STATE_DIR = "/app/state"
os.makedirs(STATE_DIR, exist_ok=True)

# Validation log max size (configurable via settings)
DEFAULT_VALIDATION_WINDOW = 1000

class OnlineErrorPredictor:
    def __init__(self, model_name: str):
        """
        Inicializa o preditor de erro online usando Regressão Logística.

        Args:
            model_name: Nome do modelo sendo monitorado (ex: 'ollama/gemma3:4b').
        """
        if not RIVER_AVAILABLE:
            logger.warning("River library not installed. Online prediction disabled.")
            return

        self.model_name = model_name

        # Sanitiza o nome do arquivo para persistência
        safe_name = model_name.replace("/", "_").replace(":", "_")
        self.save_path = os.path.join(STATE_DIR, f"predictor_{safe_name}_logistic.pkl")
        self.validation_path = os.path.join(STATE_DIR, f"predictor_{safe_name}_validation.pkl")

        self.pipeline = None

        # Phase 5: Validation tracking for calibration
        # Stores tuples of (predicted_prob, actual_error, timestamp)
        self.prediction_log: List[Tuple[float, bool, float]] = []

        # Temperature scaling for probability calibration (Platt scaling)
        # Temperature > 1 softens probabilities, < 1 sharpens them
        self.calibration_temp: float = 1.0

        # Metrics
        self._total_predictions: int = 0
        self._correct_predictions: int = 0

        self._init_model()
        self._load()  # Tenta carregar estado anterior do disco
        self._load_validation()  # Load validation history

    def _init_model(self):
        """
        Define o pipeline: StandardScaler -> Logistic Regression (SGD).
        O StandardScaler é vital para a convergência do SGD em vetores de embedding.
        """
        self.pipeline = compose.Pipeline(
            preprocessing.StandardScaler(),
            linear_model.LogisticRegression(
                optimizer=optim.SGD(lr=0.01)
            )
        )

    def _embed_to_dict(self, embedding: List[float]) -> Dict[str, float]:
        """Converte lista de floats para dicionário (formato exigido pelo River)."""
        return {f"d{i}": val for i, val in enumerate(embedding)}

    def predict_error_probability(self, embedding: List[float]) -> float:
        """
        Estima a probabilidade de o modelo ERRAR a resposta para este embedding.

        Returns:
            float: Probabilidade entre 0.0 (Acerto provável) e 1.0 (Erro provável).
        """
        if not self.pipeline:
            return 0.5  # Incerteza máxima (50/50) se não houver modelo

        try:
            x = self._embed_to_dict(embedding)

            # predict_proba_one retorna um dict com as probabilidades das classes.
            # Ex: {0: 0.8, 1: 0.2}
            probs = self.pipeline.predict_proba_one(x)

            # Retorna a probabilidade da classe "1" (Erro).
            # Se a classe 1 ainda não foi vista, retorna 0.0.
            raw_prob = probs.get(1, 0.0)

            # Apply temperature scaling for calibration
            # Temperature scaling: calibrated_prob = sigmoid(logit(raw_prob) / temperature)
            if self.calibration_temp != 1.0 and 0.0 < raw_prob < 1.0:
                import math
                logit = math.log(raw_prob / (1 - raw_prob))
                scaled_logit = logit / self.calibration_temp
                raw_prob = 1.0 / (1.0 + math.exp(-scaled_logit))

            self._total_predictions += 1
            return raw_prob

        except Exception as e:
            logger.error(f"[OnlinePredictor] Prediction failed: {e}")
            return 0.5

    def learn(self, embedding: List[float], is_correct: bool):
        """
        Atualiza os pesos do modelo com um novo exemplo rotulado (Feedback Loop).

        Args:
            embedding: Vetor da query.
            is_correct: True se o modelo acertou, False se errou.
        """
        if not self.pipeline:
            return

        try:
            x = self._embed_to_dict(embedding)

            # Definimos o alvo (Target):
            # 1 = ERRO (Queremos prever falhas para acionar o juiz)
            # 0 = ACERTO
            y = 0 if is_correct else 1

            self.pipeline.learn_one(x, y)

        except Exception as e:
            logger.error(f"[OnlinePredictor] Learning failed: {e}")

    def record_outcome(self, predicted_prob: float, actual_error: bool) -> None:
        """
        Record a prediction vs actual outcome for calibration analysis.

        Args:
            predicted_prob: The probability predicted for error (0.0 to 1.0)
            actual_error: True if the model actually made an error
        """
        timestamp = time.time()
        self.prediction_log.append((predicted_prob, actual_error, timestamp))

        # Update accuracy tracking
        predicted_error = predicted_prob >= 0.5
        if predicted_error == actual_error:
            self._correct_predictions += 1

        # Trim to max window size
        max_window = DEFAULT_VALIDATION_WINDOW
        try:
            from .settings_dynamic import settings
            max_window = settings.PREDICTOR_CALIBRATION_WINDOW
        except Exception:
            pass

        if len(self.prediction_log) > max_window:
            self.prediction_log = self.prediction_log[-max_window:]

    def compute_brier_score(self) -> float:
        """
        Compute Brier score for calibration assessment.

        Lower is better:
        - 0.0 = Perfect calibration
        - 0.25 = Random guessing (uninformative)
        - > 0.25 = Worse than random

        Returns:
            float: Brier score (0.0 to 1.0)
        """
        if len(self.prediction_log) < 50:
            return 0.25  # Not enough data, return baseline

        total = 0.0
        for predicted_prob, actual_error, _ in self.prediction_log:
            actual_value = 1.0 if actual_error else 0.0
            total += (predicted_prob - actual_value) ** 2

        return total / len(self.prediction_log)

    def compute_accuracy(self) -> float:
        """
        Compute binary classification accuracy.

        Returns:
            float: Accuracy (0.0 to 1.0)
        """
        if self._total_predictions == 0:
            return 0.5  # No predictions yet

        return self._correct_predictions / self._total_predictions

    def get_calibration_metrics(self) -> Dict[str, float]:
        """
        Get all calibration metrics for monitoring.

        Returns:
            Dict with brier_score, accuracy, total_predictions, calibration_temp
        """
        return {
            "brier_score": self.compute_brier_score(),
            "accuracy": self.compute_accuracy(),
            "total_predictions": self._total_predictions,
            "calibration_temp": self.calibration_temp,
            "validation_log_size": len(self.prediction_log),
        }

    def auto_calibrate_temperature(self) -> None:
        """
        Auto-calibrate temperature using Platt scaling on validation log.

        This adjusts the calibration_temp to minimize Brier score.
        Should be called periodically (e.g., in background loop).
        """
        if len(self.prediction_log) < 100:
            return  # Not enough data

        import math

        # Simple grid search for optimal temperature
        best_temp = 1.0
        best_score = float("inf")

        for temp in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]:
            score = self._compute_brier_with_temp(temp)
            if score < best_score:
                best_score = score
                best_temp = temp

        if best_temp != self.calibration_temp:
            logger.info(
                f"[OnlinePredictor] Calibrating {self.model_name}: "
                f"temp {self.calibration_temp:.2f} -> {best_temp:.2f} "
                f"(brier {best_score:.4f})"
            )
            self.calibration_temp = best_temp

    def _compute_brier_with_temp(self, temp: float) -> float:
        """Compute Brier score with a specific temperature value."""
        import math

        total = 0.0
        for raw_prob, actual_error, _ in self.prediction_log:
            # Apply temperature scaling
            if 0.0 < raw_prob < 1.0:
                logit = math.log(raw_prob / (1 - raw_prob))
                scaled_logit = logit / temp
                calibrated_prob = 1.0 / (1.0 + math.exp(-scaled_logit))
            else:
                calibrated_prob = raw_prob

            actual_value = 1.0 if actual_error else 0.0
            total += (calibrated_prob - actual_value) ** 2

        return total / len(self.prediction_log)

    def save(self):
        """Persiste o modelo treinado em disco."""
        if not self.pipeline:
            return
        try:
            with open(self.save_path, "wb") as f:
                pickle.dump(self.pipeline, f)
            # logger.debug(f"[OnlinePredictor] Saved state to {self.save_path}")
        except Exception as e:
            logger.warning(f"[OnlinePredictor] Save failed: {e}")

        # Also save validation data
        self._save_validation()

    def _save_validation(self):
        """Save validation log and calibration parameters."""
        try:
            validation_data = {
                "prediction_log": self.prediction_log,
                "calibration_temp": self.calibration_temp,
                "total_predictions": self._total_predictions,
                "correct_predictions": self._correct_predictions,
            }
            with open(self.validation_path, "wb") as f:
                pickle.dump(validation_data, f)
        except Exception as e:
            logger.warning(f"[OnlinePredictor] Validation save failed: {e}")

    def _load(self):
        """Carrega o modelo do disco se existir."""
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, "rb") as f:
                    self.pipeline = pickle.load(f)
                logger.info(f"[OnlinePredictor] Loaded state for {self.model_name}")
            except Exception as e:
                logger.warning(f"[OnlinePredictor] Load failed, starting fresh: {e}")

    def _load_validation(self):
        """Load validation log and calibration parameters."""
        if os.path.exists(self.validation_path):
            try:
                with open(self.validation_path, "rb") as f:
                    data = pickle.load(f)
                self.prediction_log = data.get("prediction_log", [])
                self.calibration_temp = data.get("calibration_temp", 1.0)
                self._total_predictions = data.get("total_predictions", 0)
                self._correct_predictions = data.get("correct_predictions", 0)
                logger.info(
                    f"[OnlinePredictor] Loaded validation for {self.model_name}: "
                    f"{len(self.prediction_log)} records, temp={self.calibration_temp:.2f}"
                )
            except Exception as e:
                logger.warning(f"[OnlinePredictor] Validation load failed: {e}")

# --- Singleton Factory ---
# Gerencia instâncias únicas para cada modelo LLM monitorado
_predictors: Dict[str, OnlineErrorPredictor] = {}


def get_predictor(model_name: str) -> OnlineErrorPredictor:
    if model_name not in _predictors:
        _predictors[model_name] = OnlineErrorPredictor(model_name)
    return _predictors[model_name]


def get_all_predictor_metrics() -> Dict[str, Dict[str, float]]:
    """
    Get calibration metrics for all active predictors.

    Used for Prometheus metric export and monitoring.

    Returns:
        Dict mapping model_name to calibration metrics
    """
    return {name: pred.get_calibration_metrics() for name, pred in _predictors.items()}


def calibrate_all_predictors() -> None:
    """
    Run auto-calibration on all active predictors.

    Should be called periodically from the background loop.
    """
    for name, pred in _predictors.items():
        try:
            pred.auto_calibrate_temperature()
        except Exception as e:
            logger.warning(f"[OnlinePredictor] Calibration failed for {name}: {e}")
