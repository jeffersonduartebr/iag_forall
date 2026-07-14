# -*- coding: utf-8 -*-
# Objective: Application runtime code for roadmap features.
"""Roadmap hardening features.

Provides foundational capabilities for:
- multi-tenant budget/quota governance
- audit logging
- policy version registry
- lightweight academic eval harness
- RBAC role bindings
- statistical significance utilities
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from .db import get_engine
from .services.hot_path_cache import TTLCache, cache_hit

logger = logging.getLogger(__name__)

_HOTPATH_POLICY_CACHE = TTLCache(ttl_s=float(os.getenv("HOTPATH_POLICY_CACHE_TTL_S", "30")))
_HOTPATH_TENANT_BUDGET_CACHE = TTLCache(ttl_s=float(os.getenv("HOTPATH_TENANT_BUDGET_CACHE_TTL_S", "60")))
_HOTPATH_TENANT_USAGE_CACHE = TTLCache(ttl_s=float(os.getenv("HOTPATH_TENANT_USAGE_CACHE_TTL_S", "5")))


DDL_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS tenant_budgets (
        tenant_id VARCHAR(128) PRIMARY KEY,
        daily_usd_limit FLOAT NOT NULL DEFAULT 0,
        monthly_usd_limit FLOAT NOT NULL DEFAULT 0,
        enabled TINYINT NOT NULL DEFAULT 1,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """,
    """
    CREATE TABLE IF NOT EXISTS tenant_usage (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        tenant_id VARCHAR(128) NOT NULL,
        day_key DATE NOT NULL,
        month_key VARCHAR(7) NOT NULL,
        requests INT NOT NULL DEFAULT 0,
        tokens_in BIGINT NOT NULL DEFAULT 0,
        tokens_out BIGINT NOT NULL DEFAULT 0,
        cost_usd FLOAT NOT NULL DEFAULT 0,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY uq_tenant_day (tenant_id, day_key),
        INDEX idx_tenant_month (tenant_id, month_key)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """,
    """
    CREATE TABLE IF NOT EXISTS audit_log (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        actor VARCHAR(128) NOT NULL,
        action VARCHAR(128) NOT NULL,
        resource VARCHAR(128) NOT NULL,
        tenant_id VARCHAR(128) NULL,
        metadata LONGTEXT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_action_created (action, created_at),
        INDEX idx_tenant_created (tenant_id, created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """,
    """
    CREATE TABLE IF NOT EXISTS policy_versions (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        version VARCHAR(128) NOT NULL UNIQUE,
        description TEXT,
        config_json LONGTEXT,
        is_active TINYINT NOT NULL DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_active (is_active)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """,
    """
    CREATE TABLE IF NOT EXISTS eval_runs (
        id VARCHAR(64) PRIMARY KEY,
        status VARCHAR(32) NOT NULL,
        policy_version VARCHAR(128) NULL,
        tenant_id VARCHAR(128) NULL,
        notes TEXT,
        prompts_json LONGTEXT,
        summary_json LONGTEXT,
        metadata_json LONGTEXT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_status_created (status, created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """,
    """
    CREATE TABLE IF NOT EXISTS eval_run_results (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        run_id VARCHAR(64) NOT NULL,
        prompt_text TEXT,
        model VARCHAR(255),
        quality FLOAT,
        latency_s FLOAT,
        cost_usd FLOAT,
        metadata_json LONGTEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_run_created (run_id, created_at),
        INDEX idx_run_model (run_id, model)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """,
    """
    CREATE TABLE IF NOT EXISTS rbac_user_roles (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        user_id VARCHAR(128) NOT NULL,
        role_name VARCHAR(64) NOT NULL,
        tenant_id VARCHAR(128) NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY uq_user_role_tenant (user_id, role_name, tenant_id),
        INDEX idx_user (user_id),
        INDEX idx_role (role_name)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """,
    """
    CREATE TABLE IF NOT EXISTS response_reviews (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        correlation_id VARCHAR(128) NULL,
        tenant_id VARCHAR(128) NULL,
        query_text TEXT,
        answer LONGTEXT,
        chosen_model VARCHAR(255),
        confidence_score FLOAT NULL,
        confidence_band VARCHAR(16) NULL,
        grounded TINYINT DEFAULT 0,
        verification_status VARCHAR(32) NULL,
        review_status VARCHAR(32) NOT NULL DEFAULT 'needs_review',
        review_reason VARCHAR(64) NULL,
        reviewer_id VARCHAR(128) NULL,
        reviewer_notes LONGTEXT NULL,
        corrected_answer LONGTEXT NULL,
        metadata_json LONGTEXT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_review_status_created (review_status, created_at),
        INDEX idx_tenant_review_created (tenant_id, created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """,
    """
    CREATE TABLE IF NOT EXISTS expert_profiles (
        user_id VARCHAR(128) PRIMARY KEY,
        display_name VARCHAR(255) NULL,
        theme_ids LONGTEXT NULL,
        credentials_note TEXT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """,
    """
    CREATE TABLE IF NOT EXISTS expert_accounts (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        email VARCHAR(255) NOT NULL UNIQUE,
        password_hash VARCHAR(255) NOT NULL,
        display_name VARCHAR(255) NOT NULL,
        phone VARCHAR(32) NULL,
        enabled TINYINT NOT NULL DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_expert_enabled (enabled, created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """,
    """
    CREATE TABLE IF NOT EXISTS expert_assessments (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        expert_id VARCHAR(128) NOT NULL,
        benchmark_id VARCHAR(128) NOT NULL,
        theme VARCHAR(128) NOT NULL,
        query_text TEXT,
        answer LONGTEXT,
        reference LONGTEXT NULL,
        eval_run_id VARCHAR(64) NULL,
        judge_quality FLOAT NULL,
        quality_score FLOAT NOT NULL,
        rubric_json LONGTEXT NULL,
        notes LONGTEXT NULL,
        status VARCHAR(32) NOT NULL DEFAULT 'submitted',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY uq_expert_benchmark_run (expert_id, benchmark_id, eval_run_id),
        INDEX idx_theme_created (theme, created_at),
        INDEX idx_eval_run (eval_run_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """,
]


@dataclass
class BudgetCheck:
    """Represent `BudgetCheck` within this module.

The class groups the state and behavior required for BudgetCheck."""
    allowed: bool
    reason: str
    daily_spent: float
    monthly_spent: float
    daily_limit: float
    monthly_limit: float


@dataclass
class AccessDecision:
    """Represent `AccessDecision` within this module.

The class groups the state and behavior required for AccessDecision."""
    allowed: bool
    reason: str
    roles: List[str]


def ensure_roadmap_tables() -> None:
    """Ensure all governance/policy/eval/rbac tables exist."""
    try:
        with get_engine().begin() as conn:
            for ddl in DDL_STATEMENTS:
                conn.execute(text(ddl))
            conn.execute(text("ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS metadata_json LONGTEXT NULL"))
    except Exception as e:
        logger.warning("[roadmap_features] failed to ensure tables: %s", e)


def set_tenant_budget(tenant_id: str, daily_usd_limit: float, monthly_usd_limit: float, enabled: bool = True) -> None:
    """Create/update budget limits for a tenant."""
    _HOTPATH_TENANT_BUDGET_CACHE.invalidate(tenant_id)
    with get_engine().begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO tenant_budgets (tenant_id, daily_usd_limit, monthly_usd_limit, enabled)
                VALUES (:t, :d, :m, :e)
                ON DUPLICATE KEY UPDATE
                    daily_usd_limit=:d,
                    monthly_usd_limit=:m,
                    enabled=:e
                """
            ),
            {"t": tenant_id, "d": float(daily_usd_limit), "m": float(monthly_usd_limit), "e": 1 if enabled else 0},
        )


def get_tenant_budget(tenant_id: str) -> Dict[str, Any]:
    """Get budget configuration for tenant."""
    cached = _HOTPATH_TENANT_BUDGET_CACHE.get(tenant_id)
    if cache_hit(cached):
        return cached

    with get_engine().connect() as conn:
        row = conn.execute(
            text("SELECT tenant_id, daily_usd_limit, monthly_usd_limit, enabled FROM tenant_budgets WHERE tenant_id=:t"),
            {"t": tenant_id},
        ).mappings().first()
    if not row:
        out = {"tenant_id": tenant_id, "daily_usd_limit": 0.0, "monthly_usd_limit": 0.0, "enabled": False}
    else:
        out = dict(row)
    _HOTPATH_TENANT_BUDGET_CACHE.set(tenant_id, out)
    return out


def list_tenant_budgets() -> List[Dict[str, Any]]:
    """List all configured tenant budgets."""
    try:
        with get_engine().connect() as conn:
            rows = conn.execute(
                text("SELECT tenant_id, daily_usd_limit, monthly_usd_limit, enabled, updated_at FROM tenant_budgets ORDER BY tenant_id")
            ).mappings().all()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _usage_snapshot(tenant_id: str) -> Dict[str, float]:
    """Execute the usage snapshot routine.

This helper encapsulates one focused step used by the surrounding workflow."""
    cached = _HOTPATH_TENANT_USAGE_CACHE.get(tenant_id)
    if cache_hit(cached):
        return cached

    day_key = time.strftime("%Y-%m-%d")
    month_key = time.strftime("%Y-%m")
    with get_engine().connect() as conn:
        day_cost = conn.execute(
            text("SELECT COALESCE(cost_usd,0) FROM tenant_usage WHERE tenant_id=:t AND day_key=:d"),
            {"t": tenant_id, "d": day_key},
        ).scalar() or 0.0
        month_cost = conn.execute(
            text("SELECT COALESCE(SUM(cost_usd),0) FROM tenant_usage WHERE tenant_id=:t AND month_key=:m"),
            {"t": tenant_id, "m": month_key},
        ).scalar() or 0.0
    out = {"daily": float(day_cost), "monthly": float(month_cost)}
    _HOTPATH_TENANT_USAGE_CACHE.set(tenant_id, out)
    return out


def check_tenant_budget(tenant_id: Optional[str], projected_cost_usd: float = 0.0) -> BudgetCheck:
    """Check whether tenant can proceed considering budget limits."""
    if not tenant_id:
        return BudgetCheck(True, "no_tenant_budget", 0.0, 0.0, 0.0, 0.0)

    cfg = get_tenant_budget(tenant_id)
    if not cfg.get("enabled"):
        return BudgetCheck(True, "budget_disabled", 0.0, 0.0, float(cfg.get("daily_usd_limit", 0.0)), float(cfg.get("monthly_usd_limit", 0.0)))

    snapshot = _usage_snapshot(tenant_id)
    daily_limit = float(cfg.get("daily_usd_limit", 0.0))
    monthly_limit = float(cfg.get("monthly_usd_limit", 0.0))
    daily_after = snapshot["daily"] + max(0.0, projected_cost_usd)
    monthly_after = snapshot["monthly"] + max(0.0, projected_cost_usd)

    if daily_limit > 0 and daily_after > daily_limit:
        return BudgetCheck(False, "daily_limit_exceeded", snapshot["daily"], snapshot["monthly"], daily_limit, monthly_limit)
    if monthly_limit > 0 and monthly_after > monthly_limit:
        return BudgetCheck(False, "monthly_limit_exceeded", snapshot["daily"], snapshot["monthly"], daily_limit, monthly_limit)

    return BudgetCheck(True, "ok", snapshot["daily"], snapshot["monthly"], daily_limit, monthly_limit)


def record_tenant_usage(
    tenant_id: Optional[str],
    *,
    cost_usd: float,
    tokens_in: int = 0,
    tokens_out: int = 0,
    requests: int = 1,
) -> None:
    """Accumulate daily usage for tenant."""
    if not tenant_id:
        return
    _HOTPATH_TENANT_USAGE_CACHE.invalidate(tenant_id)
    day_key = time.strftime("%Y-%m-%d")
    month_key = time.strftime("%Y-%m")
    with get_engine().begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO tenant_usage (tenant_id, day_key, month_key, requests, tokens_in, tokens_out, cost_usd)
                VALUES (:t, :d, :m, :r, :ti, :to, :c)
                ON DUPLICATE KEY UPDATE
                    requests=requests + VALUES(requests),
                    tokens_in=tokens_in + VALUES(tokens_in),
                    tokens_out=tokens_out + VALUES(tokens_out),
                    cost_usd=cost_usd + VALUES(cost_usd),
                    month_key=VALUES(month_key)
                """
            ),
            {
                "t": tenant_id,
                "d": day_key,
                "m": month_key,
                "r": max(0, int(requests)),
                "ti": max(0, int(tokens_in)),
                "to": max(0, int(tokens_out)),
                "c": max(0.0, float(cost_usd)),
            },
        )


def get_usage_summary(tenant_id: Optional[str] = None) -> Dict[str, Any]:
    """Get usage summary for one tenant or all tenants."""
    month_key = time.strftime("%Y-%m")
    with get_engine().connect() as conn:
        if tenant_id:
            row = conn.execute(
                text(
                    """
                    SELECT tenant_id, COALESCE(SUM(requests),0) AS requests, COALESCE(SUM(tokens_in),0) AS tokens_in,
                           COALESCE(SUM(tokens_out),0) AS tokens_out, COALESCE(SUM(cost_usd),0) AS cost_usd
                    FROM tenant_usage WHERE tenant_id=:t AND month_key=:m GROUP BY tenant_id
                    """
                ),
                {"t": tenant_id, "m": month_key},
            ).mappings().first()
            return dict(row or {"tenant_id": tenant_id, "requests": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0})

        rows = conn.execute(
            text(
                """
                SELECT tenant_id, COALESCE(SUM(requests),0) AS requests, COALESCE(SUM(tokens_in),0) AS tokens_in,
                       COALESCE(SUM(tokens_out),0) AS tokens_out, COALESCE(SUM(cost_usd),0) AS cost_usd
                FROM tenant_usage WHERE month_key=:m GROUP BY tenant_id ORDER BY cost_usd DESC
                """
            ),
            {"m": month_key},
        ).mappings().all()
    return {"items": [dict(r) for r in rows], "month_key": month_key}


def log_audit_event(actor: str, action: str, resource: str, tenant_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Persist an audit event."""
    with get_engine().begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO audit_log (actor, action, resource, tenant_id, metadata)
                VALUES (:a, :ac, :r, :t, :m)
                """
            ),
            {
                "a": actor[:128],
                "ac": action[:128],
                "r": resource[:128],
                "t": tenant_id,
                "m": json.dumps(metadata or {}, ensure_ascii=False),
            },
        )


def list_audit_events(limit: int = 100) -> List[Dict[str, Any]]:
    """List latest audit events."""
    with get_engine().connect() as conn:
        rows = conn.execute(
            text("SELECT id, actor, action, resource, tenant_id, metadata, created_at FROM audit_log ORDER BY id DESC LIMIT :l"),
            {"l": max(1, min(int(limit), 1000))},
        ).mappings().all()
    out: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        try:
            item["metadata"] = json.loads(item.get("metadata") or "{}")
        except Exception:
            item["metadata"] = {}
        out.append(item)
    return out


def create_policy_version(version: str, config: Dict[str, Any], description: str = "") -> None:
    """Create/update policy version."""
    with get_engine().begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO policy_versions (version, description, config_json, is_active)
                VALUES (:v, :d, :c, 0)
                ON DUPLICATE KEY UPDATE description=:d, config_json=:c
                """
            ),
            {"v": version, "d": description, "c": json.dumps(config, ensure_ascii=False)},
        )


def activate_policy_version(version: str) -> bool:
    """Mark one policy as active."""
    _HOTPATH_POLICY_CACHE.clear()
    with get_engine().begin() as conn:
        exists = conn.execute(text("SELECT COUNT(*) FROM policy_versions WHERE version=:v"), {"v": version}).scalar() or 0
        if not exists:
            return False
        conn.execute(text("UPDATE policy_versions SET is_active=0 WHERE is_active=1"))
        conn.execute(text("UPDATE policy_versions SET is_active=1 WHERE version=:v"), {"v": version})
    return True


def get_active_policy() -> Optional[Dict[str, Any]]:
    """Fetch active policy version."""
    cached = _HOTPATH_POLICY_CACHE.get("active")
    if cache_hit(cached):
        return cached

    with get_engine().connect() as conn:
        row = conn.execute(
            text("SELECT version, description, config_json, is_active, created_at, updated_at FROM policy_versions WHERE is_active=1 LIMIT 1")
        ).mappings().first()
    if not row:
        _HOTPATH_POLICY_CACHE.set("active", None)
        return None
    out = dict(row)
    try:
        out["config"] = json.loads(out.pop("config_json") or "{}")
    except Exception:
        out["config"] = {}
    _HOTPATH_POLICY_CACHE.set("active", out)
    return out


def list_policy_versions(limit: int = 100) -> List[Dict[str, Any]]:
    """List policy versions."""
    with get_engine().connect() as conn:
        rows = conn.execute(
            text("SELECT version, description, config_json, is_active, created_at, updated_at FROM policy_versions ORDER BY id DESC LIMIT :l"),
            {"l": max(1, min(int(limit), 500))},
        ).mappings().all()
    out: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        try:
            item["config"] = json.loads(item.pop("config_json") or "{}")
        except Exception:
            item["config"] = {}
        out.append(item)
    return out


def create_eval_run(
    run_id: str,
    prompts: List[str],
    policy_version: Optional[str] = None,
    tenant_id: Optional[str] = None,
    notes: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Create eval run header."""
    with get_engine().begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO eval_runs (id, status, policy_version, tenant_id, notes, prompts_json, summary_json, metadata_json)
                VALUES (:i, 'queued', :p, :t, :n, :pr, '{}', :md)
                """
            ),
            {
                "i": run_id,
                "p": policy_version,
                "t": tenant_id,
                "n": notes,
                "pr": json.dumps(prompts, ensure_ascii=False),
                "md": json.dumps(metadata or {}, ensure_ascii=False),
            },
        )


def update_eval_run_status(run_id: str, status: str, summary: Optional[Dict[str, Any]] = None) -> None:
    """Update eval run status and summary."""
    with get_engine().begin() as conn:
        conn.execute(
            text("UPDATE eval_runs SET status=:s, summary_json=:j WHERE id=:i"),
            {"s": status, "j": json.dumps(summary or {}, ensure_ascii=False), "i": run_id},
        )


def add_eval_result(run_id: str, prompt_text: str, model: str, quality: float, latency_s: float, cost_usd: float, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Persist one eval sample result."""
    with get_engine().begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO eval_run_results (run_id, prompt_text, model, quality, latency_s, cost_usd, metadata_json)
                VALUES (:r, :p, :m, :q, :l, :c, :md)
                """
            ),
            {
                "r": run_id,
                "p": prompt_text,
                "m": model,
                "q": float(quality),
                "l": float(latency_s),
                "c": float(cost_usd),
                "md": json.dumps(metadata or {}, ensure_ascii=False),
            },
        )


def list_eval_run_results(run_id: str, limit: int = 5000) -> List[Dict[str, Any]]:
    """List individual result rows for an eval run."""
    with get_engine().connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT run_id, prompt_text, model, quality, latency_s, cost_usd, metadata_json, created_at
                FROM eval_run_results
                WHERE run_id=:r
                ORDER BY id ASC
                LIMIT :l
                """
            ),
            {"r": run_id, "l": max(1, min(int(limit), 20000))},
        ).mappings().all()
    out: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        try:
            item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
        except Exception:
            item["metadata"] = {}
        out.append(item)
    return out


def get_eval_run(run_id: str) -> Optional[Dict[str, Any]]:
    """Fetch eval run with aggregated stats."""
    with get_engine().connect() as conn:
        head = conn.execute(
            text("SELECT id, status, policy_version, tenant_id, notes, prompts_json, summary_json, metadata_json, created_at, updated_at FROM eval_runs WHERE id=:i"),
            {"i": run_id},
        ).mappings().first()
        if not head:
            return None
        agg = conn.execute(
            text(
                """
                SELECT COUNT(*) AS n,
                       COALESCE(AVG(quality),0) AS quality_mean,
                       COALESCE(AVG(latency_s),0) AS latency_mean,
                       COALESCE(AVG(cost_usd),0) AS cost_mean
                FROM eval_run_results WHERE run_id=:i
                """
            ),
            {"i": run_id},
        ).mappings().first()

    out = dict(head)
    try:
        out["prompts"] = json.loads(out.pop("prompts_json") or "[]")
    except Exception:
        out["prompts"] = []
    try:
        out["summary"] = json.loads(out.pop("summary_json") or "{}")
    except Exception:
        out["summary"] = {}
    try:
        out["metadata"] = json.loads(out.pop("metadata_json") or "{}")
    except Exception:
        out["metadata"] = {}
    out["aggregate"] = dict(agg or {})
    return out


def list_eval_runs(limit: int = 50) -> List[Dict[str, Any]]:
    """List latest eval runs."""
    with get_engine().connect() as conn:
        rows = conn.execute(
            text("SELECT id, status, policy_version, tenant_id, notes, metadata_json, created_at, updated_at FROM eval_runs ORDER BY created_at DESC LIMIT :l"),
            {"l": max(1, min(int(limit), 500))},
        ).mappings().all()
    out: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        try:
            item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
        except Exception:
            item["metadata"] = {}
        out.append(item)
    return out


def create_response_review(
    *,
    correlation_id: Optional[str],
    tenant_id: Optional[str],
    query_text: str,
    answer: str,
    chosen_model: str,
    confidence_score: Optional[float],
    confidence_band: Optional[str],
    grounded: bool,
    verification_status: Optional[str],
    review_reason: Optional[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> int:
    """Insert one answer into the human-review queue and return its identifier."""
    with get_engine().begin() as conn:
        result = conn.execute(
            text(
                """
                INSERT INTO response_reviews (
                    correlation_id, tenant_id, query_text, answer, chosen_model,
                    confidence_score, confidence_band, grounded, verification_status,
                    review_status, review_reason, metadata_json
                )
                VALUES (
                    :cid, :tenant_id, :query_text, :answer, :chosen_model,
                    :confidence_score, :confidence_band, :grounded, :verification_status,
                    'needs_review', :review_reason, :metadata_json
                )
                """
            ),
            {
                "cid": correlation_id,
                "tenant_id": tenant_id,
                "query_text": query_text,
                "answer": answer,
                "chosen_model": chosen_model,
                "confidence_score": confidence_score,
                "confidence_band": confidence_band,
                "grounded": 1 if grounded else 0,
                "verification_status": verification_status,
                "review_reason": review_reason,
                "metadata_json": json.dumps(metadata or {}, ensure_ascii=False),
            },
        )
        return int(result.lastrowid or 0)


def list_response_reviews(status: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """List queued or completed response reviews for human follow-up."""
    sql = """
        SELECT id, correlation_id, tenant_id, query_text, answer, chosen_model,
               confidence_score, confidence_band, grounded, verification_status,
               review_status, review_reason, reviewer_id, reviewer_notes, corrected_answer,
               metadata_json, created_at, updated_at
        FROM response_reviews
    """
    params: Dict[str, Any] = {"l": max(1, min(int(limit), 1000))}
    if status:
        sql += " WHERE review_status=:s"
        params["s"] = status
    sql += " ORDER BY id DESC LIMIT :l"
    with get_engine().connect() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
    out: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        try:
            item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
        except Exception:
            item["metadata"] = {}
        out.append(item)
    return out


def update_response_review(
    review_id: int,
    *,
    review_status: str,
    reviewer_id: Optional[str] = None,
    reviewer_notes: Optional[str] = None,
    corrected_answer: Optional[str] = None,
) -> bool:
    """Apply a reviewer decision to one queued response review item."""
    with get_engine().begin() as conn:
        result = conn.execute(
            text(
                """
                UPDATE response_reviews
                SET review_status=:status,
                    reviewer_id=:reviewer_id,
                    reviewer_notes=:reviewer_notes,
                    corrected_answer=:corrected_answer
                WHERE id=:review_id
                """
            ),
            {
                "status": review_status,
                "reviewer_id": reviewer_id,
                "reviewer_notes": reviewer_notes,
                "corrected_answer": corrected_answer,
                "review_id": int(review_id),
            },
        )
    return bool(result.rowcount)


def grant_role(user_id: str, role_name: str, tenant_id: Optional[str] = None) -> None:
    """Grant role to a user."""
    with get_engine().begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO rbac_user_roles (user_id, role_name, tenant_id)
                VALUES (:u, :r, :t)
                ON DUPLICATE KEY UPDATE role_name=:r
                """
            ),
            {"u": user_id[:128], "r": role_name[:64], "t": tenant_id},
        )


def revoke_role(user_id: str, role_name: str, tenant_id: Optional[str] = None) -> int:
    """Revoke role from user."""
    with get_engine().begin() as conn:
        result = conn.execute(
            text(
                """
                DELETE FROM rbac_user_roles
                WHERE user_id=:u AND role_name=:r
                AND ((tenant_id IS NULL AND :t IS NULL) OR tenant_id=:t)
                """
            ),
            {"u": user_id, "r": role_name, "t": tenant_id},
        )
    return int(result.rowcount or 0)


def list_roles(user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """List all role bindings or those from one user."""
    with get_engine().connect() as conn:
        if user_id:
            rows = conn.execute(
                text("SELECT user_id, role_name, tenant_id, created_at FROM rbac_user_roles WHERE user_id=:u ORDER BY id DESC"),
                {"u": user_id},
            ).mappings().all()
        else:
            rows = conn.execute(
                text("SELECT user_id, role_name, tenant_id, created_at FROM rbac_user_roles ORDER BY id DESC LIMIT 1000")
            ).mappings().all()
    return [dict(r) for r in rows]


def get_roles_for_user(user_id: str, tenant_id: Optional[str] = None) -> List[str]:
    """Resolve effective roles for user."""
    with get_engine().connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT role_name FROM rbac_user_roles
                WHERE user_id=:u AND (tenant_id IS NULL OR tenant_id=:t)
                """
            ),
            {"u": user_id, "t": tenant_id},
        ).fetchall()
    return [str(r[0]) for r in rows]


def check_access(
    *,
    user_id: Optional[str],
    required_roles: List[str],
    tenant_id: Optional[str] = None,
    header_roles: Optional[List[str]] = None,
    jwt_roles: Optional[List[str]] = None,
) -> AccessDecision:
    """Check access by required roles."""
    if not required_roles:
        return AccessDecision(True, "no_required_role", [])

    roles: List[str] = []
    if jwt_roles:
        roles.extend([str(r).strip() for r in jwt_roles if str(r).strip()])
    if user_id:
        roles.extend(get_roles_for_user(user_id=user_id, tenant_id=tenant_id))
    trust_headers = False
    try:
        from .settings_dynamic import settings as _settings

        trust_headers = _settings._get_bool("TRUST_HEADER_ROLES", False)
    except Exception:
        trust_headers = False
    if header_roles and trust_headers:
        roles.extend([str(r).strip() for r in header_roles if str(r).strip()])
    roles = sorted(set(roles))

    if any(role in roles for role in required_roles):
        source = "jwt" if jwt_roles else "rbac"
        return AccessDecision(True, source, roles)
    return AccessDecision(False, "missing_required_role", roles)


def eval_significance_report(run_id: str) -> Dict[str, Any]:
    """Build per-model leaderboard and significance against the top model by quality."""
    from .services.academic_stats import build_model_comparison_report

    rows = list_eval_run_results(run_id)
    if not rows:
        return {"run_id": run_id, "error": "no_results"}

    by_model: Dict[str, Dict[str, List[float]]] = {}
    for row in rows:
        model = str(row.get("model") or "unknown")
        slot = by_model.setdefault(model, {"quality": [], "latency": [], "cost": []})
        slot["quality"].append(float(row.get("quality") or 0.0))
        slot["latency"].append(float(row.get("latency_s") or 0.0))
        slot["cost"].append(float(row.get("cost_usd") or 0.0))

    report = build_model_comparison_report(by_model)
    report["run_id"] = run_id
    return report


# Diretorio de especialistas revisores extraido para app.roadmap_experts (item #8).
from .roadmap_experts import (  # noqa: E402,F401
    create_expert_account,
    create_expert_assessment,
    get_expert_account_by_email,
    get_expert_account_by_id,
    get_expert_assessment_stats,
    get_expert_profile,
    list_assessed_benchmark_ids,
    list_expert_accounts,
    list_expert_assessments,
    update_expert_account,
    upsert_expert_profile,
)
