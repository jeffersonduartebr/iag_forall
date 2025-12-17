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

Used by the Router to decide whether to trigger expensive LLM-as-a-Judge evaluation
(Active Learning / Importance Sampling).
"""

from __future__ import annotations

import os
import pickle
import logging
from typing import List, Dict

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
        
        self.pipeline = None
        self._init_model()
        self._load() # Tenta carregar estado anterior do disco

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
            return 0.5 # Incerteza máxima (50/50) se não houver modelo

        try:
            x = self._embed_to_dict(embedding)
            
            # predict_proba_one retorna um dict com as probabilidades das classes.
            # Ex: {0: 0.8, 1: 0.2}
            probs = self.pipeline.predict_proba_one(x)
            
            # Retorna a probabilidade da classe "1" (Erro).
            # Se a classe 1 ainda não foi vista, retorna 0.0.
            return probs.get(1, 0.0)
            
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

    def save(self):
        """Persiste o modelo treinado em disco."""
        if not self.pipeline: return
        try:
            with open(self.save_path, "wb") as f:
                pickle.dump(self.pipeline, f)
            # logger.debug(f"[OnlinePredictor] Saved state to {self.save_path}")
        except Exception as e:
            logger.warning(f"[OnlinePredictor] Save failed: {e}")

    def _load(self):
        """Carrega o modelo do disco se existir."""
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, "rb") as f:
                    self.pipeline = pickle.load(f)
                logger.info(f"[OnlinePredictor] Loaded state for {self.model_name}")
            except Exception as e:
                logger.warning(f"[OnlinePredictor] Load failed, starting fresh: {e}")

# --- Singleton Factory ---
# Gerencia instâncias únicas para cada modelo LLM monitorado
_predictors: Dict[str, OnlineErrorPredictor] = {}

def get_predictor(model_name: str) -> OnlineErrorPredictor:
    if model_name not in _predictors:
        _predictors[model_name] = OnlineErrorPredictor(model_name)
    return _predictors[model_name]
