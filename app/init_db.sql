CREATE DATABASE IF NOT EXISTS routerdb CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE routerdb;

-- Histórico de execuções do bandit
CREATE TABLE IF NOT EXISTS bandit_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    model VARCHAR(128) NOT NULL,
    reward FLOAT NOT NULL,
    ema FLOAT NOT NULL,
    latency_s FLOAT DEFAULT 0,
    quality FLOAT DEFAULT 0,
    cost_usd FLOAT DEFAULT 0,
    query_sample TEXT
) ENGINE=InnoDB;

-- Pesos calculados pelo NSGA-II
CREATE TABLE IF NOT EXISTS nsga_weights (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    w_quality FLOAT NOT NULL,
    w_cost FLOAT NOT NULL,
    w_latency FLOAT NOT NULL,
    fitness FLOAT DEFAULT 0,
    generation INT DEFAULT 0
) ENGINE=InnoDB;
