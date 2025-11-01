from typing import Dict, Any, ClassVar, List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import os
import logging


class Settings(BaseSettings):
    """
    Configurações gerais da aplicação de roteamento de LLMs.
    Compatível com variáveis de ambiente definidas no arquivo `.env`.
    """

    # ------------------------------------------------------
    # Fonte de configuração (.env)
    # ------------------------------------------------------
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )

    # ------------------------------------------------------
    # Modelos
    # ------------------------------------------------------
    CANDIDATE_MODELS_LIST: List[str] = Field(
        default_factory=lambda: [
            "ollama/deepseek-r1:1.5b",
            "gemini/gemini-2.0-flash",
        ],
        description=(
            "Lista de todos os modelos candidatos. "
            "Pode ser definida no .env como lista JSON "
            '["a","b","c"] ou string separada por vírgulas "a,b,c".'
        ),
    )

    # ✅ Validador para aceitar formato JSON ou separado por vírgulas
    @field_validator("CANDIDATE_MODELS_LIST", mode="before")
    def split_candidate_models(cls, v):
        if isinstance(v, str):
            # Remove aspas externas e divide por vírgula
            v = v.strip().strip('"').strip("'")
            return [x.strip() for x in v.split(",") if x.strip()]
        return v

    # ------------------------------------------------------
    # Bases de API
    # ------------------------------------------------------
    OLLAMA_BASE_URL: str = Field(
        default="http://ollama:11434",
        description="URL base da API Ollama local"
    )
    LITELLM_BASE_URL: str = Field(
        default="",
        description="Base URL para LiteLLM, se necessário"
    )

    # ------------------------------------------------------
    # Parâmetros de controle
    # ------------------------------------------------------
    QUALITY_MIN: float = Field(
        default=6.5,
        description="Qualidade mínima para considerar um modelo adequado"
    )
    MAX_TOKENS: int = Field(
        default=4096,
        description="Limite máximo de tokens por requisição"
    )
    ENABLE_RAG_FOR_JUDGES: bool = Field(
        default=True,
        description="Ativa o uso de contexto RAG pelos juízes"
    )

    # ------------------------------------------------------
    # Juízes / Avaliação de qualidade
    # ------------------------------------------------------
    JUDGES_MODE: str = Field(
        default="auto",
        description="Modo de operação dos juízes (auto, disabled, manual)"
    )

    JUDGE_MODELS: List[str] = Field(
        default_factory=lambda: [
            "phi4:latest",
            "ollama/deepseek-r1:1.5b",
            "gemini-2.0-flash",
        ],
        description=(
            "Lista de modelos LLM usados como juízes. "
            "Pode ser definida no .env como JSON ou lista separada por vírgulas."
        ),
    )

    @field_validator("JUDGE_MODELS", mode="before")
    def split_judge_models(cls, v):
        if isinstance(v, str):
            v = v.strip().strip('"').strip("'")
            return [x.strip() for x in v.split(",") if x.strip()]
        return v

    # ------------------------------------------------------
    # Custos por 1K tokens
    # ------------------------------------------------------
    COSTS_USD_PER_1K_DEFAULT: ClassVar[Dict[str, float]] = {
        "phi4:latest": 0.001,
        "gemini-2.5-flash-lite": 0.15,
        "gemini-2.0-flash": 0.15,
        "deepseek-r1:1.5b": 0.002,
        "openai/gpt-5-nano": 0.1,
    }

    COSTS_USD_PER_1K: Dict[str, float] = Field(default_factory=dict)

    # ------------------------------------------------------
    # Algoritmo de decisão
    # ------------------------------------------------------
    ALGORITHM: str = Field(
        default="nsga-ii",
        description="Algoritmo multiobjetivo utilizado para otimização"
    )

    # ------------------------------------------------------
    # Observabilidade e Logging
    # ------------------------------------------------------
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Nível de log global da aplicação"
    )

    # ------------------------------------------------------
    # Inicialização personalizada
    # ------------------------------------------------------
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Define custos (env > default)
        self.COSTS_USD_PER_1K = self._load_costs()

        # Configura logging global
        logging.basicConfig(
            level=getattr(logging, self.LOG_LEVEL.upper(), logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    def _load_costs(self) -> Dict[str, float]:
        """Carrega custos personalizados a partir do ambiente, com fallback seguro."""
        result = Settings.COSTS_USD_PER_1K_DEFAULT.copy()
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
    # Segurança / Administração
    # ------------------------------------------------------
    ADMIN_TOKEN: str = Field(
        default="admin123",
        description="Token administrativo para rotas protegidas"
    )

    # ------------------------------------------------------
    # RAG (Retrieval-Augmented Generation)
    # ------------------------------------------------------
    RAG_ENABLED: bool = Field(default=True)
    RAG_SIM_THRESHOLD: float = Field(default=0.75)
    RAG_TOP_K: int = Field(default=4)

    # ------------------------------------------------------
    # Cache semântico
    # ------------------------------------------------------
    SEM_CACHE_ENABLED: bool = Field(default=True)
    CACHE_SIM_THRESHOLD: float = Field(default=0.86)
    CACHE_TTL_SECONDS: int = Field(default=86400)
    CACHE_TOP_K: int = Field(default=3)

    # ------------------------------------------------------
    # Celery / Redis
    # ------------------------------------------------------
    CELERY_BROKER_URL: str = Field(default="redis://redis:6379/0")
    CELERY_RESULT_BACKEND: str = Field(default="redis://redis:6379/1")


# ------------------------------------------------------
# Instância global de configuração
# ------------------------------------------------------
settings = Settings()
