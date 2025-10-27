from typing import Dict, Any, ClassVar
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import json
import os
import logging


class Settings(BaseSettings):
    """Configurações gerais da aplicação de roteamento de LLMs."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # ------------------------------------------------------
    # Modelos
    # ------------------------------------------------------
    OLLAMA_MODEL: str = Field(default="phi4:latest", description="Modelo local via Ollama")
    COMMERCIAL_MODEL_1: str = Field(default="gemini-2.0-flash", description="Modelo comercial 1")
    COMMERCIAL_MODEL_2: str = Field(default="gemini-2.5-flash", description="Modelo comercial 2")

    # ------------------------------------------------------
    # Bases de API
    # ------------------------------------------------------
    OLLAMA_BASE_URL: str = Field(default="http://ollama:11434", description="URL da API do Ollama")
    LITELLM_BASE_URL: str = Field(default="", description="Base URL para LiteLLM, se necessário")

    # ------------------------------------------------------
    # Parâmetros de controle
    # ------------------------------------------------------
    QUALITY_MIN: float = Field(default=6.5, description="Qualidade mínima para considerar um modelo bom")
    MAX_TOKENS: int = Field(default=512, description="Limite máximo de tokens por requisição")
    ENABLE_RAG_FOR_JUDGES: bool = Field(default=True, description="Ativa o uso de contexto RAG nos juízes")
    # ------------------------------------------------------
    # Juízes / avaliação de qualidade
    # ------------------------------------------------------
    JUDGES_MODE: str = Field(
        default="auto",
        description="Modo de operação dos juízes (auto, disabled, manual)"
    )
    # ------------------------------------------------------
    # Juízes de avaliação LLM
    # ------------------------------------------------------
    JUDGE_LLM_MODEL: str = Field(
        default="phi4:latest",
        description="Modelo usado para julgamento de qualidade de resposta"
    )
    # ------------------------------------------------------
    # Configuração de quantidade de juízes LLM
    # ------------------------------------------------------
    JUDGE_LLM_N: int = Field(
        default=3,
        description="Número de instâncias LLM usadas para julgamento paralelo"
    )


    # ------------------------------------------------------
    # Custos por 1K tokens
    # ------------------------------------------------------
    COSTS_USD_PER_1K_DEFAULT: ClassVar[Dict[str, float]] = {
        "phi4:latest": 0.001,
        "gemini-2.0-flash": 0.15,
        "gemini-2.5-flash": 0.25,
    }

    COSTS_USD_PER_1K: Dict[str, float] = Field(default_factory=dict)

    # ------------------------------------------------------
    # Algoritmo de decisão
    # ------------------------------------------------------
    ALGORITHM: str = Field(default="nsga-ii", description="Algoritmo multiobjetivo usado")

    # ------------------------------------------------------
    # Observabilidade
    # ------------------------------------------------------
    LOG_LEVEL: str = Field(default="INFO", description="Nível de log da aplicação")

    # ------------------------------------------------------
    # Métodos auxiliares
    # ------------------------------------------------------
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Define custos (env > default)
        self.COSTS_USD_PER_1K = self._load_costs()

        # Configura logging
        logging.basicConfig(
            level=getattr(logging, self.LOG_LEVEL.upper(), logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    def _load_costs(self) -> Dict[str, float]:
        """Carrega custos de variáveis de ambiente ou usa os padrões."""
        result = self.COSTS_USD_PER_1K_DEFAULT.copy()
        try:
            for key in result.keys():
                env_key = f"COST_{key.replace('-', '_').replace(':', '_').upper()}"
                env_val = os.getenv(env_key)
                if env_val:
                    result[key] = float(env_val)
        except Exception as e:
            logging.warning(f"[settings] Falha ao carregar custos customizados: {e}")
        return result


# ------------------------------------------------------
# Instância global de configuração
# ------------------------------------------------------
settings = Settings()
