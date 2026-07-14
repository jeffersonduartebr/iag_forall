# Objective: Typed settings property accessors extracted from DynamicSettings (roadmap #19).
"""Typed @property accessors for DynamicSettings, split out as a mixin.

DynamicSettings inherits this mixin; the properties call ``self._get_*`` / ``self.get``
(resolved on DynamicSettings via the MRO), so this module only needs the shared DB
env defaults and typing imports."""

from __future__ import annotations

from typing import List

from .settings_env import DB_HOST_ENV, DB_NAME_ENV, DB_PASS_ENV, DB_PORT_ENV, DB_USER_ENV


class TypedSettingsMixin:
    """Typed, cached accessors for known settings, mixed into DynamicSettings."""

    # -------------------------
    # Propriedades Tipadas
    # -------------------------
    @property
    def CANDIDATE_MODELS_LIST(self) -> List[str]:
        """Executa a responsabilidade descrita por este método.

        Returns:
            Valor produzido pela execução.
        """
        return self._get_list("CANDIDATE_MODELS_LIST")

    @property
    def CANDIDATE_VISION_MODELS_LIST(self) -> List[str]:
        """Executa a responsabilidade descrita por este método.

        Returns:
            Valor produzido pela execução.
        """
        return self._get_list("CANDIDATE_VISION_MODELS_LIST")

    @property
    def CANDIDATE_MULTIMODAL_MODELS_LIST(self) -> List[str]:
        """Executa a responsabilidade descrita por este método.

        Returns:
            Valor produzido pela execução.
        """
        return self._get_list("CANDIDATE_MULTIMODAL_MODELS_LIST")

    @property
    def CANDIDATE_TOOL_MODELS_LIST(self) -> List[str]:
        """Lista de modelos habilitados para tool/function calling (opt-in explícito).

        Quando não vazia, restringe o roteamento de requisições com ``tools`` a
        esses modelos. Quando vazia, o roteador infere a capacidade por modelo.

        Returns:
            Lista de nomes de modelos com suporte a tools.
        """
        return self._get_list("CANDIDATE_TOOL_MODELS_LIST")

    @property
    def VLM_OLLAMA_MODELS(self) -> List[str]:
        """Executa a responsabilidade descrita por este método.

        Returns:
            Valor produzido pela execução.
        """
        return self._get_list("VLM_OLLAMA_MODELS")

    @property
    def JUDGE_MODELS(self) -> List[str]:
        """Executa a responsabilidade descrita por este método.

        Returns:
            Valor produzido pela execução.
        """
        return self._get_list("JUDGE_MODELS")

    # Embeddings
    @property
    def EMBED_TEXT_MODEL(self) -> str:
        """Executa a responsabilidade descrita por este método.

        Returns:
            Valor produzido pela execução.
        """
        return self._get_str("EMBED_TEXT_MODEL", "nomic-embed-text")

    @property
    def TEXT_EMBEDDING_MODEL(self) -> str:
        """Obtém o valor da configuração `TEXT_EMBEDDING_MODEL`."""
        return self.get("TEXT_EMBEDDING_MODEL")

    @property
    def IMAGE_EMBEDDING_MODEL(self) -> str:
        """Obtém o valor da configuração `IMAGE_EMBEDDING_MODEL`."""
        return self.get("IMAGE_EMBEDDING_MODEL")

    @property
    def MULTIMODAL_EMBEDDING_MODEL(self) -> str:
        """Obtém o valor da configuração `MULTIMODAL_EMBEDDING_MODEL`."""
        return self.get("MULTIMODAL_EMBEDDING_MODEL")

    @property
    def EMBED_MODEL(self) -> str:
        """Obtém o valor da configuração `EMBED_MODEL`."""
        return self.get("EMBED_MODEL")

    @property
    def EMBED_PROVIDER(self) -> str:
        """Obtém o valor da configuração `EMBED_PROVIDER`."""
        return self.get("EMBED_PROVIDER")

    @property
    def EMBED_DEVICE(self) -> str:
        """Obtém o valor da configuração `EMBED_DEVICE`."""
        return self.get("EMBED_DEVICE")

    # Router Params
    @property
    def MAX_TOKENS_DEFAULT(self) -> int:
        """Obtém o valor da configuração `MAX_TOKENS_DEFAULT`."""
        return self._get_int("MAX_TOKENS_DEFAULT", 2000)

    @property
    def TEMPERATURE_DEFAULT(self) -> float:
        """Obtém o valor da configuração `TEMPERATURE_DEFAULT`."""
        return self._get_float("TEMPERATURE_DEFAULT", 0.5)

    @property
    def BANDIT_EPSILON(self) -> float:
        """Obtém o valor da configuração `BANDIT_EPSILON`."""
        return self._get_float("BANDIT_EPSILON", 0.12)

    @property
    def QUERY_LOG_RETENTION_DAYS(self) -> int:
        """Obtém o valor da configuração `QUERY_LOG_RETENTION_DAYS`."""
        return self._get_int("QUERY_LOG_RETENTION_DAYS", 7)

    # Banco / Infra
    @property
    def REDIS_HOST(self) -> str:
        """Obtém o valor da configuração `REDIS_HOST`."""
        return self.get("REDIS_HOST")

    @property
    def REDIS_PORT(self) -> int:
        """Obtém o valor da configuração `REDIS_PORT`."""
        return self._get_int("REDIS_PORT", 6379)

    @property
    def REDIS_DB(self) -> int:
        """Obtém o valor da configuração `REDIS_DB`."""
        return self._get_int("REDIS_DB", 0)

    @property
    def REDIS_PASSWORD(self) -> str:
        """Obtém o valor da configuração `REDIS_PASSWORD`."""
        return self.get("REDIS_PASSWORD")

    @property
    def DB_HOST(self) -> str:
        """Obtém o valor da configuração `DB_HOST`."""
        return self.get("DB_HOST", DB_HOST_ENV)

    @property
    def DB_PORT(self) -> int:
        """Obtém o valor da configuração `DB_PORT`."""
        return self._get_int("DB_PORT", DB_PORT_ENV)

    @property
    def DB_USER(self) -> str:
        """Obtém o valor da configuração `DB_USER`."""
        return self.get("DB_USER", DB_USER_ENV)

    @property
    def DB_PASS(self) -> str:
        """Obtém o valor da configuração `DB_PASS`."""
        return self.get("DB_PASS", DB_PASS_ENV)

    @property
    def DB_NAME(self) -> str:
        """Obtém o valor da configuração `DB_NAME`."""
        return self.get("DB_NAME", DB_NAME_ENV)

    @property
    def ADMIN_TOKEN(self) -> str:
        """Obtém o valor da configuração `ADMIN_TOKEN`."""
        return self.get("ADMIN_TOKEN", "")

    @property
    def ADMIN_TOKEN_PREVIOUS(self) -> str:
        """Obtém o valor da configuração `ADMIN_TOKEN_PREVIOUS`."""
        return self.get("ADMIN_TOKEN_PREVIOUS", "")

    @property
    def REQUIRE_API_AUTH(self) -> bool:
        """Whether public API endpoints require authentication."""
        return self._get_bool("REQUIRE_API_AUTH", False)

    # Judges
    @property
    def JUDGES_ENABLED(self) -> bool:
        """Obtém o valor da configuração `JUDGES_ENABLED`."""
        return self._get_bool("JUDGES_ENABLED", True)

    @property
    def JUDGES_MODE(self) -> str:
        """Obtém o valor da configuração `JUDGES_MODE`."""
        return self.get("JUDGES_MODE")

    @property
    def JUDGES_LOCAL_MODEL(self) -> str:
        """Obtém o valor da configuração `JUDGES_LOCAL_MODEL`."""
        return self.get("JUDGES_LOCAL_MODEL")

    @property
    def JUDGES_REMOTE_MODEL(self) -> str:
        """Obtém o valor da configuração `JUDGES_REMOTE_MODEL`."""
        return self.get("JUDGES_REMOTE_MODEL")

    @property
    def JUDGES_TIMEOUT_S(self) -> int:
        """Obtém o valor da configuração `JUDGES_TIMEOUT_S`."""
        return self._get_int("JUDGES_TIMEOUT_S", 15)

    @property
    def JUDGE_MIN_SAMPLE_RATE(self) -> float:
        """Obtém o valor da configuração `JUDGE_MIN_SAMPLE_RATE`."""
        return self._get_float("JUDGE_MIN_SAMPLE_RATE", 0.05)

    # Ollama
    @property
    def OLLAMA_HOST(self) -> str:
        """Obtém o valor da configuração `OLLAMA_HOST`."""
        return self.get("OLLAMA_HOST") or "http://ollama:11434"

    @property
    def OLLAMA_BASE_URL(self) -> str:
        """Obtém o valor da configuração `OLLAMA_BASE_URL`."""
        return self.get("OLLAMA_BASE_URL") or self.OLLAMA_HOST

    # Centroids
    @property
    def CENTROIDS_DIM(self) -> int:
        """Obtém o valor da configuração `CENTROIDS_DIM`."""
        return int(self.get("CENTROIDS_DIM"))

    @property
    def CENTROIDS_K(self) -> int:
        """Obtém o valor da configuração `CENTROIDS_K`."""
        return int(self.get("CENTROIDS_K"))

    @property
    def CENTROIDS_MIN_SIM_CREATE(self) -> float:
        """Obtém o valor da configuração `CENTROIDS_MIN_SIM_CREATE`."""
        return float(self.get("CENTROIDS_MIN_SIM_CREATE"))

    @property
    def CENTROIDS_ENABLE_ONLINE(self) -> bool:
        """Obtém o valor da configuração `CENTROIDS_ENABLE_ONLINE`."""
        return self._get_bool("CENTROIDS_ENABLE_ONLINE", True)

    @property
    def CENTROIDS_UPDATE_INTERVAL_S(self) -> int:
        """Obtém o valor da configuração `CENTROIDS_UPDATE_INTERVAL_S`."""
        return int(self.get("CENTROIDS_UPDATE_INTERVAL_S"))

    @property
    def CENTROIDS_MIN_RECORDS_FOR_TRAIN(self) -> int:
        """Obtém o valor da configuração `CENTROIDS_MIN_RECORDS_FOR_TRAIN`."""
        return int(self.get("CENTROIDS_MIN_RECORDS_FOR_TRAIN"))

    @property
    def CENTROIDS_MAX_HISTORY(self) -> int:
        """Obtém o valor da configuração `CENTROIDS_MAX_HISTORY`."""
        return int(self.get("CENTROIDS_MAX_HISTORY"))

    @property
    def CENTROIDS_HOURLY_REFRESH_ENABLED(self) -> bool:
        """Obtém o valor da configuração `CENTROIDS_HOURLY_REFRESH_ENABLED`."""
        return str(self.get("CENTROIDS_HOURLY_REFRESH_ENABLED")).strip() in ("1", "true", "True")

    @property
    def CENTROIDS_MIN_LOG_ROWS_FOR_REFRESH(self) -> int:
        """Obtém o valor da configuração `CENTROIDS_MIN_LOG_ROWS_FOR_REFRESH`."""
        return int(self.get("CENTROIDS_MIN_LOG_ROWS_FOR_REFRESH"))

    # NSGA / Meta
    @property
    def NSGA_UPDATE_INTERVAL_S(self) -> int:
        """Obtém o valor da configuração `NSGA_UPDATE_INTERVAL_S`."""
        return int(self.get("NSGA_UPDATE_INTERVAL_S", "300"))

    @property
    def NSGA_LOOKBACK_MINUTES(self) -> int:
        """Obtém o valor da configuração `NSGA_LOOKBACK_MINUTES`."""
        return int(self.get("NSGA_LOOKBACK_MINUTES", "180"))

    @property
    def NSGA_LOOKBACK_MAXROWS(self) -> int:
        """Obtém o valor da configuração `NSGA_LOOKBACK_MAXROWS`."""
        return int(self.get("NSGA_LOOKBACK_MAXROWS", "2000"))

    @property
    def METAOPT_REPS(self) -> int:
        """Obtém o valor da configuração `METAOPT_REPS`."""
        return int(self.get("METAOPT_REPS", "5"))

    @property
    def METAOPT_TRIALS(self) -> int:
        """Obtém o valor da configuração `METAOPT_TRIALS`."""
        return int(self.get("METAOPT_TRIALS", "100"))

    # Propriedades de Pesos NSGA-II
    @property
    def NSGA_W_QUALITY(self) -> float:
        """Obtém o valor da configuração `NSGA_W_QUALITY`."""
        return float(self.get("NSGA_W_QUALITY", 1.0))

    @property
    def NSGA_W_LATENCY(self) -> float:
        """Obtém o valor da configuração `NSGA_W_LATENCY`."""
        return float(self.get("NSGA_W_LATENCY", 0.5))

    @property
    def NSGA_W_COST(self) -> float:
        """Obtém o valor da configuração `NSGA_W_COST`."""
        return float(self.get("NSGA_W_COST", 50.0))

    @property
    def NSGA_W_ALIGNMENT(self) -> float:
        """Obtém o valor da configuração `NSGA_W_ALIGNMENT`."""
        return float(self.get("NSGA_W_ALIGNMENT", 1.0))

    # Phase 1: Monitoring Properties
    @property
    def NSGA_CONVERGENCE_HISTORY_SIZE(self) -> int:
        """Obtém o valor da configuração `NSGA_CONVERGENCE_HISTORY_SIZE`."""
        return int(self.get("NSGA_CONVERGENCE_HISTORY_SIZE", 20))

    @property
    def CASCADE_WARNING_THRESHOLD(self) -> float:
        """Obtém o valor da configuração `CASCADE_WARNING_THRESHOLD`."""
        return float(self.get("CASCADE_WARNING_THRESHOLD", 0.3))

    @property
    def CASCADE_CRITICAL_THRESHOLD(self) -> float:
        """Obtém o valor da configuração `CASCADE_CRITICAL_THRESHOLD`."""
        return float(self.get("CASCADE_CRITICAL_THRESHOLD", 0.5))

    # Phase 2: Self-Tuning Properties
    @property
    def RISK_FACTOR_SOTA_HIGH_UQ(self) -> float:
        """Obtém o valor da configuração `RISK_FACTOR_SOTA_HIGH_UQ`."""
        return float(self.get("RISK_FACTOR_SOTA_HIGH_UQ", 1.3))

    @property
    def RISK_FACTOR_LOCAL_HIGH_UQ(self) -> float:
        """Obtém o valor da configuração `RISK_FACTOR_LOCAL_HIGH_UQ`."""
        return float(self.get("RISK_FACTOR_LOCAL_HIGH_UQ", 0.6))

    @property
    def RISK_FACTOR_LOCAL_LOW_UQ(self) -> float:
        """Obtém o valor da configuração `RISK_FACTOR_LOCAL_LOW_UQ`."""
        return float(self.get("RISK_FACTOR_LOCAL_LOW_UQ", 1.1))

    @property
    def RISK_FACTOR_ADAPT_ENABLED(self) -> bool:
        """Obtém o valor da configuração `RISK_FACTOR_ADAPT_ENABLED`."""
        return self._get_bool("RISK_FACTOR_ADAPT_ENABLED", False)

    @property
    def RISK_FACTOR_ADAPT_RATE(self) -> float:
        """Obtém o valor da configuração `RISK_FACTOR_ADAPT_RATE`."""
        return float(self.get("RISK_FACTOR_ADAPT_RATE", 0.02))

    @property
    def ADAPTIVE_TIMEOUT_ENABLED(self) -> bool:
        """Obtém o valor da configuração `ADAPTIVE_TIMEOUT_ENABLED`."""
        return self._get_bool("ADAPTIVE_TIMEOUT_ENABLED", False)

    @property
    def ADAPTIVE_TIMEOUT_MULTIPLIER(self) -> float:
        """Obtém o valor da configuração `ADAPTIVE_TIMEOUT_MULTIPLIER`."""
        return float(self.get("ADAPTIVE_TIMEOUT_MULTIPLIER", 2.0))

    @property
    def ADAPTIVE_TIMEOUT_REASONING_MULTIPLIER(self) -> float:
        """Obtém o valor da configuração `ADAPTIVE_TIMEOUT_REASONING_MULTIPLIER`."""
        return float(self.get("ADAPTIVE_TIMEOUT_REASONING_MULTIPLIER", 3.0))

    @property
    def MIN_TIMEOUT(self) -> int:
        """Obtém o valor da configuração `MIN_TIMEOUT`."""
        return int(self.get("MIN_TIMEOUT", 30))

    @property
    def MAX_TIMEOUT(self) -> int:
        """Obtém o valor da configuração `MAX_TIMEOUT`."""
        return int(self.get("MAX_TIMEOUT", 1200))

    @property
    def META_OPT_ENABLED(self) -> bool:
        """Obtém o valor da configuração `META_OPT_ENABLED`."""
        return self._get_bool("META_OPT_ENABLED", False)

    @property
    def META_OPT_SCHEDULE_HOUR(self) -> int:
        """Obtém o valor da configuração `META_OPT_SCHEDULE_HOUR`."""
        return int(self.get("META_OPT_SCHEDULE_HOUR", 3))

    @property
    def META_OPT_SCHEDULED_TRIALS(self) -> int:
        """Obtém o valor da configuração `META_OPT_SCHEDULED_TRIALS`."""
        return int(self.get("META_OPT_SCHEDULED_TRIALS", 20))

    # Phase 3: Feedback Properties
    @property
    def DRIFT_THRESHOLD(self) -> float:
        """Obtém o valor da configuração `DRIFT_THRESHOLD`."""
        return float(self.get("DRIFT_THRESHOLD", 0.15))

    @property
    def DRIFT_WINDOW_SIZE(self) -> int:
        """Obtém o valor da configuração `DRIFT_WINDOW_SIZE`."""
        return int(self.get("DRIFT_WINDOW_SIZE", 100))

    @property
    def USER_FEEDBACK_WEIGHT(self) -> float:
        """Obtém o valor da configuração `USER_FEEDBACK_WEIGHT`."""
        return float(self.get("USER_FEEDBACK_WEIGHT", 0.7))

    # Phase 4: A/B Testing Properties
    @property
    def AB_TESTING_ENABLED(self) -> bool:
        """Obtém o valor da configuração `AB_TESTING_ENABLED`."""
        return self._get_bool("AB_TESTING_ENABLED", False)

    # Phase 5: Autonomous Behavior - Adaptive Cache Properties
    @property
    def CACHE_THRESHOLD_MIN(self) -> float:
        """Obtém o valor da configuração `CACHE_THRESHOLD_MIN`."""
        return float(self.get("CACHE_THRESHOLD_MIN", 0.85))

    @property
    def CACHE_THRESHOLD_MAX(self) -> float:
        """Obtém o valor da configuração `CACHE_THRESHOLD_MAX`."""
        return float(self.get("CACHE_THRESHOLD_MAX", 0.98))

    @property
    def CACHE_HIT_RATE_TARGET(self) -> float:
        """Obtém o valor da configuração `CACHE_HIT_RATE_TARGET`."""
        return float(self.get("CACHE_HIT_RATE_TARGET", 0.20))

    @property
    def CACHE_THRESHOLD_ADAPT_ENABLED(self) -> bool:
        """Obtém o valor da configuração `CACHE_THRESHOLD_ADAPT_ENABLED`."""
        return self._get_bool("CACHE_THRESHOLD_ADAPT_ENABLED", False)

    # Phase 5: Autonomous Behavior - Predictor Validation Properties
    @property
    def PREDICTOR_VALIDATION_ENABLED(self) -> bool:
        """Obtém o valor da configuração `PREDICTOR_VALIDATION_ENABLED`."""
        return self._get_bool("PREDICTOR_VALIDATION_ENABLED", True)

    @property
    def PREDICTOR_BRIER_SCORE_THRESHOLD(self) -> float:
        """Obtém o valor da configuração `PREDICTOR_BRIER_SCORE_THRESHOLD`."""
        return float(self.get("PREDICTOR_BRIER_SCORE_THRESHOLD", 0.25))

    @property
    def PREDICTOR_CALIBRATION_WINDOW(self) -> int:
        """Obtém o valor da configuração `PREDICTOR_CALIBRATION_WINDOW`."""
        return int(self.get("PREDICTOR_CALIBRATION_WINDOW", 1000))

    # Phase 5: Autonomous Behavior - UQ Calibration Properties
    @property
    def UQ_CALIBRATION_ENABLED(self) -> bool:
        """Obtém o valor da configuração `UQ_CALIBRATION_ENABLED`."""
        return self._get_bool("UQ_CALIBRATION_ENABLED", True)

    @property
    def UQ_QUALITY_GAP_RELAX(self) -> float:
        """Obtém o valor da configuração `UQ_QUALITY_GAP_RELAX`."""
        return float(self.get("UQ_QUALITY_GAP_RELAX", 0.5))

    @property
    def UQ_QUALITY_GAP_TIGHTEN(self) -> float:
        """Obtém o valor da configuração `UQ_QUALITY_GAP_TIGHTEN`."""
        return float(self.get("UQ_QUALITY_GAP_TIGHTEN", 2.0))

    # Phase 5: Autonomous Behavior - Judge Calibration Properties
    @property
    def JUDGE_CALIBRATION_ENABLED(self) -> bool:
        """Obtém o valor da configuração `JUDGE_CALIBRATION_ENABLED`."""
        return self._get_bool("JUDGE_CALIBRATION_ENABLED", True)

    @property
    def JUDGE_CACHE_AGREEMENT_TARGET(self) -> float:
        """Obtém o valor da configuração `JUDGE_CACHE_AGREEMENT_TARGET`."""
        return float(self.get("JUDGE_CACHE_AGREEMENT_TARGET", 0.7))

    # Circuit Breaker Properties
    @property
    def CIRCUIT_BREAKER_FAIL_MAX(self) -> int:
        """Obtém o valor da configuração `CIRCUIT_BREAKER_FAIL_MAX`."""
        return int(self.get("CIRCUIT_BREAKER_FAIL_MAX", 5))

    @property
    def CIRCUIT_BREAKER_RESET_TIMEOUT(self) -> int:
        """Obtém o valor da configuração `CIRCUIT_BREAKER_RESET_TIMEOUT`."""
        return int(self.get("CIRCUIT_BREAKER_RESET_TIMEOUT", 60))

    @property
    def CIRCUIT_BREAKER_LOCAL_FAIL_MAX(self) -> int:
        """Obtém o valor da configuração `CIRCUIT_BREAKER_LOCAL_FAIL_MAX`."""
        return int(self.get("CIRCUIT_BREAKER_LOCAL_FAIL_MAX", 3))

    @property
    def CIRCUIT_BREAKER_LOCAL_RESET_TIMEOUT(self) -> int:
        """Obtém o valor da configuração `CIRCUIT_BREAKER_LOCAL_RESET_TIMEOUT`."""
        return int(self.get("CIRCUIT_BREAKER_LOCAL_RESET_TIMEOUT", 30))

    # Backpressure / Concurrency Control
    @property
    def MAX_CONCURRENT_REQUESTS(self) -> int:
        """Obtém o valor da configuração `MAX_CONCURRENT_REQUESTS`."""
        return int(self.get("MAX_CONCURRENT_REQUESTS", 500))

    @property
    def BACKPRESSURE_ENABLED(self) -> bool:
        """Obtém o valor da configuração `BACKPRESSURE_ENABLED`."""
        return self._get_bool("BACKPRESSURE_ENABLED", True)

    # Emergency Fallback Models
    @property
    def EMERGENCY_FALLBACK_MODELS(self) -> List[str]:
        """Executa a responsabilidade descrita por este método.

        Returns:
            Valor produzido pela execução.
        """
        return self._get_list("EMERGENCY_FALLBACK_MODELS")
