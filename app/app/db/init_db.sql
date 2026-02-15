-- ============================================================
-- 📘 init_db.sql (MySQL/MariaDB)
-- ------------------------------------------------------------
-- Cria tabelas essenciais e popula dados iniciais de teste.
-- Todas as tabelas com:
-- ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
-- ============================================================

-- ------------------------------------------------------------
-- Histórico de EMA (Exponential Moving Averages)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ema_history (
    model VARCHAR(255) PRIMARY KEY,
    ema_latency FLOAT NOT NULL,
    ema_quality FLOAT NOT NULL,
    ema_cost FLOAT NOT NULL,
    updates INT DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP 
        ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

INSERT IGNORE INTO ema_history (model, ema_latency, ema_quality, ema_cost, updates)
VALUES
('openai/gpt-4o-mini', 0.95, 9.2, 0.002, 12),
('ollama/llama3', 1.2, 8.4, 0.0001, 8),
('anthropic/claude-3-haiku', 1.5, 8.9, 0.001, 6);

-- ------------------------------------------------------------
-- Métricas detalhadas dos modelos
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_metrics (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    model VARCHAR(255) NOT NULL,
    latency_ms FLOAT DEFAULT 0,
    cost_usd FLOAT DEFAULT 0,
    quality_score FLOAT DEFAULT 0,
    fitness FLOAT DEFAULT 0,
    generation INT DEFAULT 0,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP 
        ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_model_timestamp (model, timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

INSERT IGNORE INTO model_metrics (model, latency_ms, cost_usd, quality_score, fitness, generation)
VALUES
('openai/gpt-4o-mini', 820, 0.002, 9.4, 0.88, 5),
('ollama/llama3', 1120, 0.0001, 8.5, 0.73, 4),
('anthropic/claude-3-haiku', 950, 0.001, 8.9, 0.81, 3);

-- ------------------------------------------------------------
-- Cache Semântico (metadados de consultas)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS semantic_cache (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    query_hash VARCHAR(64) UNIQUE,
    query_text TEXT,
    context_category VARCHAR(50),
    answer TEXT,
    model_used VARCHAR(255),
    embedding LONGBLOB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ------------------------------------------------------------
-- Log de Juízes (auditoria de fallback e notas)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS judge_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    query TEXT,
    answer TEXT,
    judge_model VARCHAR(255),
    score_before FLOAT,
    fallback_model VARCHAR(255),
    score_after FLOAT,
    event_type VARCHAR(50)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

INSERT IGNORE INTO judge_logs (query, answer, judge_model, score_before, fallback_model, score_after, event_type)
VALUES
('Explique o conceito de NSGA-II', 'NSGA-II é um algoritmo genético multiobjetivo...', 'openai/gpt-4o-mini', 0.0, 'ollama/llama3', 8.7, 'zero_score'),
('Defina aprendizado por reforço', 'É uma técnica baseada em recompensas...', 'anthropic/claude-3-haiku', 7.5, NULL, 7.5, 'normal');

-- ------------------------------------------------------------
-- Bandit Context Stats
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bandit_context_stats (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    context_type VARCHAR(50) NOT NULL,
    model VARCHAR(255) NOT NULL,
    avg_reward FLOAT DEFAULT 0,
    count INT DEFAULT 0,
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP 
        ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uq_ctx_model (context_type, model)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

INSERT IGNORE INTO bandit_context_stats (context_type, model, avg_reward, count)
VALUES
('default', 'openai/gpt-4o-mini', 0.87, 35),
('default', 'ollama/llama3', 0.75, 29);

-- ------------------------------------------------------------
-- Query Log
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS query_log (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    query_text TEXT NOT NULL,
    chosen_model VARCHAR(255) NOT NULL,
    answer TEXT,
    quality FLOAT,
    latency_s FLOAT,
    cost_per_1k FLOAT,
    reward FLOAT,
    context_label VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

INSERT IGNORE INTO query_log (query_text, chosen_model, answer, quality, latency_s, cost_per_1k, reward, context_label)
VALUES
('Explique o conceito de NSGA-II', 'openai/gpt-4o-mini', 'NSGA-II é um algoritmo evolutivo...', 9.3, 1.12, 0.002, 0.85, 'multiobjective'),
('Defina aprendizado supervisionado', 'ollama/llama3', 'É o tipo de aprendizado com rótulos...', 8.7, 1.45, 0.0001, 0.78, 'ml-basics');

-- ------------------------------------------------------------
-- EMA History Log
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ema_history_log (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    model VARCHAR(255) NOT NULL,
    ema_latency FLOAT NOT NULL,
    ema_quality FLOAT NOT NULL,
    ema_cost FLOAT NOT NULL,
    update_num INT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_model_created_at (model, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

INSERT IGNORE INTO ema_history_log (model, ema_latency, ema_quality, ema_cost, update_num)
VALUES
('openai/gpt-4o-mini', 0.95, 9.2, 0.002, 12),
('ollama/llama3', 1.2, 8.4, 0.0001, 8);

-- ------------------------------------------------------------
-- Finalização
-- ------------------------------------------------------------
SELECT '✅ init_db.sql executado com sucesso — banco inicial pronto.' AS status;


-- Criação do usuário de acesso ao banco de dados
CREATE USER IF NOT EXISTS 'router_user'@'%' IDENTIFIED BY 'router_pass';
GRANT ALL PRIVILEGES ON routerdb.* TO 'router_user'@'%';
FLUSH PRIVILEGES;
