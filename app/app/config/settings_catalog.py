# Objective: Configuration support code for settings catalog.
"""Catalog of dynamic settings grouped by domain."""

from __future__ import annotations

import json
from typing import Dict, List

from .constants import DEFAULT_UNCERTAINTY_THRESHOLD

SETTINGS_BY_DOMAIN: Dict[str, Dict[str, str]] = {
    "auth": {
        "REQUIRE_API_AUTH": "0",
        # Origens que o browser pode usar para chamar esta API. Resolvida por
        # pedido (middleware/cors.py), por isso muda em runtime como qualquer
        # outra chave operacional. Era lida com `os.getenv` no import: fora do
        # catálogo, fora do /admin/settings, e impossível de mudar sem
        # reiniciar o container.
        "ADMIN_UI_CORS_ORIGINS": "http://localhost:5173,http://localhost:8082,http://127.0.0.1:8082",
        "API_KEYS": "",
        "JWT_SECRET": "",
        "JWT_ALGORITHM": "HS256",
        "ENV": "development",
        "METRICS_TOKEN": "",
        "TRUSTED_PROXY_IPS": "",
        "TRUST_HEADER_ROLES": "0",
        "ENFORCE_TENANT_BINDING": "1",
        "ROADMAP_AUTO_DDL": "1",
    },
    "runtime": {
        "MAX_TOKENS_DEFAULT": "4096",
        "ROUTER_SIMPLE_QUERY_MAX_TOKENS": "1024",
        "ROUTER_SIMPLE_TEXT_MAX_FALLBACKS": "1",
        "TEMPERATURE_DEFAULT": "0.55",
        "QUERY_LOG_RETENTION_DAYS": "7",
        "REQUEST_TIMEOUT_SECONDS": "120",
        "REQUEST_DEDUP_ENABLED": "1",
        "MAX_CONCURRENT_REQUESTS": "500",
        "BACKPRESSURE_ENABLED": "1",
        "BACKPRESSURE_REDIS_ENABLED": "0",
        "REDIS_REQUIRED_IN_PRODUCTION": "1",
        "REDIS_DEDUP_ENABLED": "1",
        "ADAPTIVE_LIMITER_ENABLED": "1",
        "ADAPTIVE_LIMITER_WINDOW_SECONDS": "15",
        "ADAPTIVE_LIMITER_HYSTERESIS_WINDOWS": "3",
        "ADAPTIVE_LIMITER_ELEVATED_UTILIZATION": "0.80",
        "ADAPTIVE_LIMITER_CONGESTED_UTILIZATION": "1.00",
        "ADAPTIVE_LIMITER_ELEVATED_QUEUE_WAIT_P95_MS": "500",
        "ADAPTIVE_LIMITER_CONGESTED_QUEUE_WAIT_P95_MS": "1000",
        "ADAPTIVE_LIMITER_SYNC_QUEUE_WAIT_MS": "250",
        "ADAPTIVE_LIMITER_INTERACTIVE_PER_SLOT_ELEVATED": "12",
        "ADAPTIVE_LIMITER_INTERACTIVE_PER_SLOT_CONGESTED": "6",
        "ADAPTIVE_LIMITER_ADMIN_PER_SLOT_ELEVATED": "3",
        "ADAPTIVE_LIMITER_ADMIN_PER_SLOT_CONGESTED": "1",
        "QUERY_JOB_TTL_SECONDS": "3600",
        "QUERY_JOB_MAX_PENDING_PER_TENANT": "500",
        "QUERY_JOB_MAX_PENDING_PER_IP": "250",
        "SYNC_DEADLINE_SIMPLE_SECONDS": "25",
        "SYNC_DEADLINE_KNOWLEDGE_SECONDS": "40",
        "SYNC_DEADLINE_REASONING_SECONDS": "70",
        "SYNC_DEADLINE_VISION_SECONDS": "100",
        "PROVIDER_TIMEOUT_SIMPLE_SECONDS": "20",
        "PROVIDER_TIMEOUT_KNOWLEDGE_SECONDS": "35",
        "PROVIDER_TIMEOUT_REASONING_SECONDS": "60",
        "PROVIDER_TIMEOUT_VISION_SECONDS": "90",
        "ROUTER_PERF_MODE": "0",
        "ROUTER_QUERY_CLASSIFIER_ENABLED": "1",
    },
    "redis": {
        "REDIS_HOST": "redis",
        "REDIS_PORT": "6379",
        "REDIS_DB": "0",
        "REDIS_PASSWORD": "",
        "TENANT_RATE_LIMIT_ENABLED": "1",
        "TENANT_RATE_LIMIT_RPM": "120",
        "TENANT_RATE_LIMIT_WINDOW_S": "60",
    },
    "embeddings": {
        "EMBED_MODEL": "nomic-embed-text",
        "EMBED_PROVIDER": "ollama",
        "EMBED_DEVICE": "cpu",
        "EMBED_TEXT_MODEL": "nomic-embed-text",
        "TEXT_EMBEDDING_MODEL": "nomic-embed-text",
        "IMAGE_EMBEDDING_MODEL": "clip-vit-large-patch14",
        "MULTIMODAL_EMBEDDING_MODEL": "clip-vit-large-patch14",
    },
    "centroids": {
        "CENTROIDS_DIM": "768",
        "CENTROIDS_K": "20",
        "CENTROIDS_MIN_SIM_CREATE": "0.35",
        "CENTROIDS_ENABLE_ONLINE": "1",
        "CENTROIDS_UPDATE_INTERVAL_S": "1800",
        "CENTROIDS_MIN_RECORDS_FOR_TRAIN": "50",
        "CENTROIDS_MAX_HISTORY": "50000",
        "CENTROIDS_HOURLY_REFRESH_ENABLED": "1",
        "CENTROIDS_MIN_LOG_ROWS_FOR_REFRESH": "50",
    },
    "judges": {
        "JUDGES_ENABLED": "1",
        "JUDGES_MODE": "llm",
        "JUDGES_LOCAL_MODEL": "ollama/phi4:latest",
        "JUDGES_REMOTE_MODEL": "ollama/phi4:latest",
        "JUDGES_TIMEOUT_S": "15",
        "JUDGE_MIN_SAMPLE_RATE": "0.05",
        "JUDGE_CALIBRATION_ENABLED": "1",
        "JUDGE_CACHE_AGREEMENT_TARGET": "0.7",
        "JUDGE_MODELS": "[]",
        # "rubric": 3 dimensões (Cap. 5, §5.3.5) por 2 juízes; "binary": CORRECT/INCORRECT.
        "JUDGE_SCORING_MODE": "rubric",
        "JUDGE_RUBRIC_WEIGHTS": json.dumps({"clareza": 0.3, "acuracia": 0.5, "alinhamento": 0.2}),
        # Diferença em Q (0-10) entre os dois juízes a partir da qual o meta-juiz desempata.
        "JUDGE_RUBRIC_DISAGREEMENT": "3.0",
        # Juiz de usurpação (p_entrega). Desligado por omissão: acrescenta uma
        # terceira chamada de juiz por julgamento e muda a semântica da nota
        # calibrada, por isso só deve ser ligado com o resto da migração.
        "JUDGE_USURPATION_ENABLED": "0",
        "JUDGE_USURPATION_DISAGREEMENT": "0.5",
    },
    "providers": {
        "OPENROUTER_API_KEY": "",
        "OLLAMA_BASE_URL": "http://ollama:11434",
        "OLLAMA_HOST": "http://ollama:11434",
        "OLLAMA_CONCURRENCY_LIMIT": "5",
        "OLLAMA_DYNAMIC_CONCURRENCY_ENABLED": "0",
        "OLLAMA_CONCURRENCY_MIN": "1",
        "OLLAMA_CONCURRENCY_MAX": "5",
        "OLLAMA_VRAM_TARGET_UTILIZATION": "0.72",
        "OLLAMA_VRAM_HIGH_WATERMARK": "0.82",
        "OLLAMA_VRAM_LOW_WATERMARK": "0.55",
        "OLLAMA_VRAM_POLL_INTERVAL_SECONDS": "5",
        "OLLAMA_CONCURRENCY_STEP_UP": "1",
        "OLLAMA_CONCURRENCY_STEP_DOWN": "1",
        "OLLAMA_CONCURRENCY_STABLE_WINDOWS": "3",
        "OLLAMA_GPU_INDEX": "0",
        "OLLAMA_WARM_MODELS": "[]",
        "OLLAMA_INTERACTIVE_WARM_MODELS": "[]",
        "OLLAMA_INTERACTIVE_MODEL_SET": "[]",
        "OLLAMA_WARMUP_GENERATE_ENABLED": "1",
        "OLLAMA_LOAD_PENALTY_ENABLED": "1",
        "OLLAMA_BACKGROUND_LOAD_SHEDDING_ENABLED": "1",
        "OLLAMA_ROUTE_CANDIDATE_LIMIT": "3",
        "PROVIDER_UNAVAILABLE_NEGATIVE_CACHE_TTL_SECONDS": "30",
        "MODEL_UNAVAILABLE_NEGATIVE_CACHE_TTL_SECONDS": "30",
        "CANDIDATE_MODELS_LIST": "[]",
        "CANDIDATE_VISION_MODELS_LIST": "[]",
        "CANDIDATE_MULTIMODAL_MODELS_LIST": "[]",
        "CANDIDATE_TOOL_MODELS_LIST": "[]",
        "VLM_OLLAMA_MODELS": json.dumps(
            [
                "qwen3-vl:8b",
                "gemma3:4b",
                "llama3.2:3b",
                "llama3:8b",
                "llava:7b",
                "llama3.2-vision:11b",
                "llava-llama3:8b",
                "granite3.2-vision:2b",
            ]
        ),
        "EMERGENCY_FALLBACK_MODELS": json.dumps(
            [
                "ollama/phi4:latest",
                "ollama/gemma3:4b",
                "ollama/llama3:8b",
            ]
        ),
    },
    "rag_cache": {
        "CACHE_TTL_DAYS": "7",
        "CACHE_THRESHOLD": "0.92",
        "CACHE_THRESHOLD_MIN": "0.85",
        "CACHE_THRESHOLD_MAX": "0.98",
        "CACHE_HIT_RATE_TARGET": "0.20",
        "CACHE_THRESHOLD_ADAPT_ENABLED": "0",
        # Canonicalize queries (casefold + whitespace collapse) before hashing/
        # embedding so trivial surface variants share a cache entry (perf #24).
        "SEMANTIC_CACHE_NORMALIZE_ENABLED": "1",
        "UNCERTAINTY_THRESHOLD": str(DEFAULT_UNCERTAINTY_THRESHOLD),
        "RAG_SIMPLE_QUERY_BYPASS_ENABLED": "1",
        "RAG_LIGHT_TOP_K": "2",
        "RAG_LIGHT_VECTOR_TOP_K": "6",
        "RAG_LIGHT_SPARSE_TOP_K": "6",
        "RAG_RERANK_MIN_CANDIDATES": "3",
        "RAG_LIGHT_CONTEXT_TOKEN_BUDGET": "480",
        "RAG_FULL_CONTEXT_TOKEN_BUDGET": "1200",
        "RAG_CONTEXT_TOKEN_BUDGET": "1200",
        "RAG_CONTEXT_QUALITY_MIN_DOCS": "2",
        "RERANK_ENABLED_FOR_LIGHT_RETRIEVAL": "0",
        "RERANK_MODEL": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        "RERANK_ENABLED": "1",
        "RAG_DATA_DIR": "/app/data",
        "CHROMA_HOST": "",
        "CHROMA_PORT": "8000",
        "CHROMA_PATH": "/data/chroma",
    },
    "routing": {
        "BANDIT_EPSILON": "0.12",
        "OPENROUTER_EXPLORATION_ENABLED": "0",
        "OPENROUTER_EXPLORATION_MODE": "balanced",
        "OPENROUTER_EXPLORATION_RATE": "0.10",
        "OPENROUTER_EXPLORATION_MAX_PER_DAY": "100",
        "OPENROUTER_EXPLORATION_MAX_USD_PER_DAY": "3.0",
        "OPENROUTER_EXPLORATION_MAX_PRICE_PROMPT_1K": "0.01",
        "OPENROUTER_EXPLORATION_MAX_PRICE_COMPLETION_1K": "0.03",
        "OPENROUTER_EXPLORATION_PROMOTE_MIN_SAMPLES": "15",
        "OPENROUTER_EXPLORATION_PROMOTE_MIN_REWARD": "0.72",
        "OPENROUTER_EXPLORATION_PROMOTE_MAX_LATENCY_S": "30.0",
        "OPENROUTER_EXPLORATION_PROMOTE_MAX_COST_USD_PER_1K": "0.02",
        "OPENROUTER_EXPLORATION_PROMOTE_MAX_FAILURE_RATE": "0.20",
        "OPENROUTER_EXPLORATION_AUTO_PROMOTE_ENABLED": "1",
        "OPENROUTER_EXPLORATION_ADAPTIVE_RATE_ENABLED": "1",
        "OPENROUTER_EXPLORATION_POOL_CACHE_TTL_S": "600",
        "OPENROUTER_EXPLORATION_SHADOW_COMPARE_RATE": "0.05",
        "OPENROUTER_EXPLORATION_CONSECUTIVE_FAILURE_BLOCK": "3",
        "OPENROUTER_EXPLORATION_PROVIDER_ALLOWLIST": json.dumps(
            ["anthropic", "openai", "google", "meta-llama", "mistralai"]
        ),
        "OPENROUTER_EXPLORATION_POOL_SIZE": "40",
        "NSGA_W_QUALITY": "1.0",
        "NSGA_W_LATENCY": "0.5",
        "NSGA_W_COST": "100.0",
        "NSGA_W_ALIGNMENT": "1.0",
        # Inclinação da transferência logística de latência.
        "REWARD_LATENCY_K": "0.12",
        # Limiar logístico dinâmico: x0(N) = TTFT + N/taxa, o prazo justo de uma
        # resposta daquele comprimento. Desligado por omissão: ligá-lo desloca a
        # distribuição da recompensa e obriga a recalibrar o gate de promoção
        # por quantil, não por valor absoluto.
        "REWARD_DYNAMIC_LATENCY_ENABLED": "0",
        "REWARD_LATENCY_TTFT_BUDGET_S": "5.0",
        "REWARD_LATENCY_TOKENS_PER_S": "25.0",
        "REWARD_DEFAULT_COMPLETION_TOKENS": "375",
        # C_base da recompensa (USD/1k tokens); derivação em app.services.reward.
        "REWARD_COST_BASELINE_PER_1K": "0.007",
        # Parcela mínima de cada objetivo nos pesos da recompensa publicados pelo NSGA-II.
        "REWARD_WEIGHT_MIN_SHARE": "0.05",
        "NSGA_CONVERGENCE_HISTORY_SIZE": "20",
        "NSGA_UPDATE_INTERVAL_S": "300",
        "NSGA_LOOKBACK_MINUTES": "180",
        "NSGA_LOOKBACK_MAXROWS": "2000",
    },
    "resilience": {
        "CASCADE_WARNING_THRESHOLD": "0.3",
        "CASCADE_CRITICAL_THRESHOLD": "0.5",
        "RISK_FACTOR_SOTA_HIGH_UQ": "1.3",
        "RISK_FACTOR_LOCAL_HIGH_UQ": "0.6",
        "RISK_FACTOR_LOCAL_LOW_UQ": "1.1",
        "RISK_FACTOR_ADAPT_ENABLED": "0",
        "RISK_FACTOR_ADAPT_RATE": "0.02",
        "ADAPTIVE_TIMEOUT_ENABLED": "0",
        "ADAPTIVE_TIMEOUT_MULTIPLIER": "2.0",
        "ADAPTIVE_TIMEOUT_REASONING_MULTIPLIER": "3.0",
        "MIN_TIMEOUT": "30",
        "MAX_TIMEOUT": "1200",
        # Speculative (hedged) requests (perf #22): race the primary against one
        # backup to cut tail latency. Off by default (extra provider cost on the
        # hedged fraction). Delay is EMA*factor unless a fixed ms is set.
        "REQUEST_HEDGING_ENABLED": "0",
        "REQUEST_HEDGE_DELAY_MS": "0",
        "REQUEST_HEDGE_EMA_FACTOR": "1.3",
        "REQUEST_HEDGE_MAX_PARALLEL": "2",
        # Cadeia de fallback: quando o modelo escolhido falha, tenta os
        # alternativos do registry antes de devolver erro ao utilizador. Estava
        # a ser lida com default False e **não existia no catálogo**, por isso
        # um pedido falhado devolvia 502/503/504 com outros modelos saudáveis
        # configurados — e o CascadeDetector nunca disparava, porque conta
        # breakers que só `execute_with_fallback` alimenta.
        # Ao contrário do hedging, não custa chamadas extra: só corre depois de
        # uma falha, e `_fallback_budget` devolve 0 quando o deadline restante
        # não dá para outra tentativa.
        "REQUEST_FALLBACK_ENABLED": "1",
        "REQUEST_MAX_FALLBACKS": "2",
        # Orçamento de tempo por vazão medida (services/orcamento_tempo.py): o prazo de cada chamada é
        # latência inicial + tokens pedidos / tokens-por-segundo do modelo (EMA no Redis), e o da
        # requisição cresce com max_tokens até PRAZO_MAXIMO_S, reservando tempo para um fallback.
        "TPS_PADRAO_LOCAL": "10",
        "TPS_PADRAO_NUVEM": "40",
        "LATENCIA_INICIAL_S": "4",
        "RESERVA_FALLBACK_S": "45",
        # Teto do prazo síncrono: cobre modelos lentos com raciocínio longo (DeepSeek levou ~91 s no teste de 100
        # consultas de 2026-09-25; a 30 tok/s, 2048 de resposta + 4096 de raciocínio passam de 270 s).
        "PRAZO_MAXIMO_S": "420",
        # Cota de raciocínio (thinking) dos modelos de nuvem, somada ao teto da resposta visível:
        # sem ela o raciocínio consumia max_tokens e a resposta vinha vazia.
        "REASONING_BUDGET_TOKENS": "4096",
        # Período de estudo (ex.: 2026s2): estado do bandit/EMA próprio por período (quality_semantics).
        # Vazio = sem prefixo (o estado existente continua com as mesmas chaves).
        "BANDIT_POLICY_NAMESPACE": "",
        "CIRCUIT_BREAKER_FAIL_MAX": "5",
        "CIRCUIT_BREAKER_RESET_TIMEOUT": "60",
        "CIRCUIT_BREAKER_LOCAL_FAIL_MAX": "3",
        "CIRCUIT_BREAKER_LOCAL_RESET_TIMEOUT": "30",
    },
    # Execução em sombra (services/sombra): para uma amostra de requisições, as demais candidatas admissíveis
    # recebem o mesmo prompt e contexto e são julgadas pelo mesmo painel; só escores, custo, latência, tokens e
    # o hash do texto são gravados (shadow_evaluations). Nunca atrasa a resposta nem altera a política.
    # Padrões do protocolo emendado do Caso 1 (25/09/2026): Gemini (Vertex) e OpenRouter, sem restrição de região.
    # Regime de exploração do protocolo do Caso 1 (services/regime): sorteio explícito com probabilidade
    # registrada, teto de exploração por participante numa janela móvel de episódios e regeneração em
    # aproveitamento quando a resposta explorada falha na verificação de incerteza. Vazio = regime desligado.
    "regime": {
        "REGIME_EXPLORACAO_TENANTS": "ifrn-caso1",
        "REGIME_EPSILON": "0.15",
        "REGIME_TETO": "0.15",
        "REGIME_JANELA_EPISODIOS": "20",
        # Aquecimento antes do campo (até esta data, inclusive, no fuso de Fortaleza): exploração maior entre as
        # candidatas configuradas, sem teto por participante, para o bandit aprender a compará-las antes dos
        # estudantes. Vazio = sem aquecimento. O campo do Caso 1 começa em 16/11/2026.
        "REGIME_AQUECIMENTO_ATE": "2026-11-15",
        "REGIME_EPSILON_AQUECIMENTO": "0.5",
    },
    "shadow": {
        "SHADOW_EXECUTION_ENABLED": "1",
        "SHADOW_TENANT_ALLOWLIST": "ifrn-caso1",
        "SHADOW_SAMPLE_RATE": "0.15",
        "SHADOW_STRATA": "disciplina,faixa_incerteza,modalidade",
        "SHADOW_PROVIDER_ALLOWLIST": "gemini/,openrouter/",
        # Vazio desliga a verificação; preenchido, só passa candidata com região verificável e igual.
        "SHADOW_REQUIRED_CLOUD_REGION": "",
        "SHADOW_MAX_CONCURRENCY": "16",
        "SHADOW_LOCAL_MAX_CONCURRENCY": "1",
        "SHADOW_DAILY_BUDGET": "4.00",
        "SHADOW_RATE_LIMIT_PER_TENANT_HOUR": "30",
        "SHADOW_TIMEOUT_S": "300",
        "SHADOW_BUDGET_TZ": "America/Fortaleza",
        "SHADOW_JUDGE_MODELS": json.dumps(
            [
                "gemini/gemini-3.1-pro-preview",
                "openrouter/moonshotai/kimi-k3",
                "openrouter/x-ai/grok-4.7",
                "openrouter/openai/gpt-5.6-sol",
            ]
        ),
        # Custo: cada resposta tem 3 juízes (a ordem do painel é sorteada por requisição; 1 dos 4 fica de fora)
        # e cada requisição roda 25% das demais candidatas (sorteio uniforme, probabilidade gravada por linha).
        "SHADOW_JUDGES_PER_ANSWER": "3",
        "SHADOW_CANDIDATE_FRACTION": "0.25",
    },
    "adversarial_governance": {
        # Closed-loop adversarial governance (roadmap #17). Off by default so
        # production routing is unaffected until explicitly enabled.
        "ADVGOV_ENABLED": "0",
        # Judge score (0..10) below which an adversarial attack counts as a success.
        "ADVGOV_FAIL_SCORE_THRESHOLD": "7.0",
        # Minimum duels observed before a knowledge cluster can be flagged high-risk.
        "ADVGOV_CLUSTER_MIN_SAMPLES": "5",
        # Attack-success rate at/above which a cluster is considered high-risk.
        "ADVGOV_CLUSTER_FAILURE_RATE_THRESHOLD": "0.5",
        # TTL (seconds) for per-cluster risk state in Redis (default 30 days).
        "ADVGOV_CLUSTER_TTL_S": "2592000",
        # Whether high-risk clusters / high UQ escalate the chosen model.
        "ADVGOV_ESCALATION_ENABLED": "1",
        # Preferred escalation targets (JSON list); first match in candidates wins.
        "ADVGOV_ESCALATION_MODELS": json.dumps([]),
    },
    "feedback": {
        "DRIFT_THRESHOLD": "0.15",
        "DRIFT_WINDOW_SIZE": "100",
        "USER_FEEDBACK_WEIGHT": "0.7",
        "JUDGE_FEEDBACK_MIN_SAMPLES": "30",
        "JUDGE_FEEDBACK_ERROR_THRESHOLD": "5.0",
        "JUDGE_BACKGROUND_THROTTLE_ENABLED": "1",
        "BACKGROUND_THROTTLE_INFLIGHT_THRESHOLD": "4",
        "BACKGROUND_THROTTLE_QUEUE_WAIT_P95_MS": "750",
        "QUERY_JOB_BACKGROUND_THROTTLE_PENDING_THRESHOLD": "1",
    },
    "experimentation": {
        "AB_TESTING_ENABLED": "0",
        "META_OPT_ENABLED": "0",
        "META_OPT_SCHEDULE_HOUR": "3",
        "META_OPT_SCHEDULED_TRIALS": "20",
        "METAOPT_SCHEDULED_REPS": "2",
        "METAOPT_REPS": "5",
        "METAOPT_TRIALS": "100",
    },
    "local_cost": {
        # Imputação do custo de inferência local pelo tempo de ocupação do equipamento
        # (app.utils.pricing.impute_local_cost). Valores de hardware/energia PROVISÓRIOS:
        # confirmar preço, consumo sob carga e tarifa antes de rodar experimentos.
        "LOCAL_COST_IMPUTATION_ENABLED": "1",
        "LOCAL_COST_USD_PER_HOUR": "0",
        "LOCAL_COST_HW_PRICE_USD": "2300",
        "LOCAL_COST_HW_LIFETIME_YEARS": "3",
        "LOCAL_COST_HW_UTILIZATION": "0.5",
        "LOCAL_COST_POWER_W": "400",
        "LOCAL_COST_ENERGY_USD_PER_KWH": "0.16",
        "LOCAL_COST_PARALLEL_SLOTS": "1",
    },
    "prediction": {
        "PREDICTOR_VALIDATION_ENABLED": "1",
        "PREDICTOR_BRIER_SCORE_THRESHOLD": "0.25",
        "PREDICTOR_CALIBRATION_WINDOW": "1000",
        "UQ_CALIBRATION_ENABLED": "1",
        "UQ_QUALITY_GAP_RELAX": "0.5",
        "UQ_QUALITY_GAP_TIGHTEN": "2.0",
    },
}

SETTINGS_DEFAULTS: Dict[str, str] = {
    key: value for domain_defaults in SETTINGS_BY_DOMAIN.values() for key, value in domain_defaults.items()
}

REQUIRES_RESTART_KEYS = {
    "REDIS_HOST",
    "REDIS_PORT",
    "REDIS_DB",
    "REDIS_PASSWORD",
    "OLLAMA_HOST",
    "OLLAMA_BASE_URL",
    "EMBED_MODEL",
    "EMBED_PROVIDER",
    "EMBED_DEVICE",
    "TEXT_EMBEDDING_MODEL",
    "IMAGE_EMBEDDING_MODEL",
    "MULTIMODAL_EMBEDDING_MODEL",
    "RAG_DATA_DIR",
    "CHROMA_HOST",
    "CHROMA_PORT",
    "CHROMA_PATH",
}

SETTING_METADATA: Dict[str, Dict[str, str]] = {
    key: {
        "domain": domain,
        "mutability": "requires_restart" if key in REQUIRES_RESTART_KEYS else "runtime_safe",
    }
    for domain, domain_defaults in SETTINGS_BY_DOMAIN.items()
    for key in domain_defaults
}


def known_setting_keys() -> List[str]:
    """Return catalogued setting keys in stable order."""
    return list(SETTINGS_DEFAULTS.keys())


def metadata_for(key: str) -> Dict[str, str]:
    """Return metadata for one setting key."""
    return dict(SETTING_METADATA.get(key, {}))


def is_known_setting(key: str) -> bool:
    """Return whether one key belongs to the dynamic settings catalog."""
    return key in SETTING_METADATA


def is_runtime_mutable(key: str) -> bool:
    """Return whether one setting can be applied without restart."""
    return metadata_for(key).get("mutability") == "runtime_safe"


#: O domínio cuja alteração muda quem pode entrar, não como o router se comporta.
SECURITY_DOMAIN = "auth"


def is_security_setting(key: str) -> bool:
    """Whether changing this key changes *who may get in*.

    Duas categorias, e a segunda é a que uma lista escrita à mão esqueceria:

    - Tudo no domínio ``auth``: ``REQUIRE_API_AUTH``, ``TRUST_HEADER_ROLES``,
      ``JWT_SECRET``, ``ADMIN_UI_CORS_ORIGINS``, ``ENV``. Mudar qualquer um
      deles é mudar a política de acesso da instalação inteira.
    - Qualquer credencial, onde quer que viva no catálogo — ``REDIS_PASSWORD``
      está no domínio ``redis``, não em ``auth``.

    A segunda categoria reutiliza deliberadamente o mesmo predicado que a
    redacção da API de admin, a cifra em repouso e a auditoria já usam. Uma
    lista própria aqui divergiria dessas três no dia em que alguém
    acrescentasse uma chave nova.
    """
    from .settings_encryption import is_secret

    return key in SETTINGS_BY_DOMAIN.get(SECURITY_DOMAIN, {}) or is_secret(key)


def split_by_security(keys) -> tuple:
    """``(operacionais, de segurança)``, preservando a ordem dada."""
    security = [key for key in keys if is_security_setting(key)]
    operational = [key for key in keys if not is_security_setting(key)]
    return operational, security
