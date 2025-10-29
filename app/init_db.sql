-- =============================================================
-- Estrutura inicial do banco routerdb (versão aprimorada)
-- =============================================================

CREATE DATABASE IF NOT EXISTS routerdb CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE routerdb;

-- Histórico de recompensas dos bandits
CREATE TABLE IF NOT EXISTS bandit_history (
  id           BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  ts_utc       DATETIME(6) NOT NULL,
  model        VARCHAR(128) NOT NULL,
  reward       DOUBLE NOT NULL,
  ema          DOUBLE NOT NULL,
  query_sample VARCHAR(256) NULL,
  PRIMARY KEY (id),
  KEY idx_ts (ts_utc),
  KEY idx_model_ts (model, ts_utc)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Histórico de pesos gerados pelo NSGA-II
CREATE TABLE IF NOT EXISTS nsga_weights (
  id            BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  created_at    DATETIME(6) NOT NULL,
  w_q           DOUBLE NOT NULL,
  w_c           DOUBLE NOT NULL,
  w_l           DOUBLE NOT NULL,
  fitness_mean  DOUBLE NOT NULL,
  generations   INT NOT NULL,
  model_name    VARCHAR(128) NOT NULL,
  model_family  VARCHAR(64)  NOT NULL,
  token_key     VARCHAR(32)  NOT NULL,
  PRIMARY KEY (id),
  KEY idx_created (created_at),
  KEY idx_family (model_family)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Último conjunto de pesos NSGA vigente
CREATE TABLE IF NOT EXISTS nsga_current_weights (
  id           TINYINT NOT NULL,
  updated_at   DATETIME(6) NOT NULL,
  w_q          DOUBLE NOT NULL,
  w_c          DOUBLE NOT NULL,
  w_l          DOUBLE NOT NULL,
  fitness_mean DOUBLE NOT NULL,
  generations  INT NOT NULL,
  model_name   VARCHAR(128) NOT NULL,
  model_family VARCHAR(64)  NOT NULL,
  token_key    VARCHAR(32)  NOT NULL,
  PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
