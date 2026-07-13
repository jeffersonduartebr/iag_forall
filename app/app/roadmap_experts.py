# -*- coding: utf-8 -*-
# Objective: Expert-reviewer directory & assessment persistence (split from roadmap_features).
"""Cadastro de especialistas revisores e suas avaliacoes (perfis, contas, assessments).

Extraido de ``roadmap_features`` (item #8, refatoracao de SLOC): este dominio de
"expert review" e autocontido — usa apenas o engine do banco e ``text`` — e nao
depende do restante das features de governanca. ``roadmap_features`` reexporta
estes nomes para preservar os imports existentes.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from .db import get_engine


def upsert_expert_profile(
    user_id: str,
    *,
    display_name: Optional[str] = None,
    theme_ids: Optional[List[str]] = None,
    credentials_note: Optional[str] = None,
) -> None:
    """Create or update one expert reviewer profile."""
    with get_engine().begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO expert_profiles (user_id, display_name, theme_ids, credentials_note)
                VALUES (:u, :d, :t, :c)
                ON DUPLICATE KEY UPDATE
                    display_name=COALESCE(:d, display_name),
                    theme_ids=COALESCE(:t, theme_ids),
                    credentials_note=COALESCE(:c, credentials_note)
                """
            ),
            {
                "u": user_id[:128],
                "d": display_name,
                "t": json.dumps(theme_ids or [], ensure_ascii=False) if theme_ids is not None else None,
                "c": credentials_note,
            },
        )


def get_expert_profile(user_id: str) -> Optional[Dict[str, Any]]:
    """Fetch expert profile for one user."""
    with get_engine().connect() as conn:
        row = conn.execute(
            text("SELECT user_id, display_name, theme_ids, credentials_note, created_at, updated_at FROM expert_profiles WHERE user_id=:u"),
            {"u": user_id},
        ).mappings().first()
    if not row:
        return None
    out = dict(row)
    try:
        out["theme_ids"] = json.loads(out.pop("theme_ids") or "[]")
    except Exception:
        out["theme_ids"] = []
    return out


def create_expert_account(
    *,
    email: str,
    password_hash: str,
    display_name: str,
    phone: Optional[str] = None,
) -> int:
    """Insert one expert account and return its id."""
    with get_engine().begin() as conn:
        result = conn.execute(
            text(
                """
                INSERT INTO expert_accounts (email, password_hash, display_name, phone, enabled)
                VALUES (:email, :password_hash, :display_name, :phone, 1)
                """
            ),
            {
                "email": email[:255],
                "password_hash": password_hash,
                "display_name": display_name[:255],
                "phone": phone[:32] if phone else None,
            },
        )
        return int(result.lastrowid or 0)


def get_expert_account_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Fetch expert account by email including password hash."""
    with get_engine().connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT id, email, password_hash, display_name, phone, enabled, created_at, updated_at
                FROM expert_accounts WHERE email=:email
                """
            ),
            {"email": email},
        ).mappings().first()
    return dict(row) if row else None


def get_expert_account_by_id(account_id: int) -> Optional[Dict[str, Any]]:
    """Fetch expert account by id including password hash."""
    with get_engine().connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT id, email, password_hash, display_name, phone, enabled, created_at, updated_at
                FROM expert_accounts WHERE id=:id
                """
            ),
            {"id": int(account_id)},
        ).mappings().first()
    return dict(row) if row else None


def list_expert_accounts(limit: int = 500) -> List[Dict[str, Any]]:
    """List expert accounts ordered by newest first."""
    with get_engine().connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT id, email, password_hash, display_name, phone, enabled, created_at, updated_at
                FROM expert_accounts
                ORDER BY id DESC
                LIMIT :l
                """
            ),
            {"l": max(1, min(int(limit), 1000))},
        ).mappings().all()
    return [dict(row) for row in rows]


def update_expert_account(
    account_id: int,
    *,
    display_name: Optional[str] = None,
    phone: Optional[str] = None,
    password_hash: Optional[str] = None,
    enabled: Optional[bool] = None,
) -> bool:
    """Update mutable expert account fields."""
    sets: List[str] = []
    params: Dict[str, Any] = {"id": int(account_id)}
    if display_name is not None:
        sets.append("display_name=:display_name")
        params["display_name"] = display_name[:255]
    if phone is not None:
        sets.append("phone=:phone")
        params["phone"] = phone[:32] if phone else None
    if password_hash is not None:
        sets.append("password_hash=:password_hash")
        params["password_hash"] = password_hash
    if enabled is not None:
        sets.append("enabled=:enabled")
        params["enabled"] = 1 if enabled else 0
    if not sets:
        return False
    sql = f"UPDATE expert_accounts SET {', '.join(sets)} WHERE id=:id"
    with get_engine().begin() as conn:
        result = conn.execute(text(sql), params)
    return bool(result.rowcount)


def create_expert_assessment(
    *,
    expert_id: str,
    benchmark_id: str,
    theme: str,
    query_text: str,
    answer: str,
    reference: Optional[str],
    eval_run_id: Optional[str],
    judge_quality: Optional[float],
    quality_score: float,
    rubric: Dict[str, Any],
    notes: Optional[str] = None,
) -> int:
    """Persist one human expert assessment."""
    with get_engine().begin() as conn:
        result = conn.execute(
            text(
                """
                INSERT INTO expert_assessments (
                    expert_id, benchmark_id, theme, query_text, answer, reference,
                    eval_run_id, judge_quality, quality_score, rubric_json, notes
                )
                VALUES (
                    :expert_id, :benchmark_id, :theme, :query_text, :answer, :reference,
                    :eval_run_id, :judge_quality, :quality_score, :rubric_json, :notes
                )
                ON DUPLICATE KEY UPDATE
                    quality_score=:quality_score,
                    rubric_json=:rubric_json,
                    notes=:notes,
                    judge_quality=COALESCE(:judge_quality, judge_quality),
                    answer=:answer,
                    reference=:reference
                """
            ),
            {
                "expert_id": expert_id[:128],
                "benchmark_id": benchmark_id[:128],
                "theme": theme[:128],
                "query_text": query_text,
                "answer": answer,
                "reference": reference,
                "eval_run_id": eval_run_id,
                "judge_quality": judge_quality,
                "quality_score": float(quality_score),
                "rubric_json": json.dumps(rubric, ensure_ascii=False),
                "notes": notes,
            },
        )
        return int(result.lastrowid or 0)


def list_expert_assessments(
    expert_id: Optional[str] = None,
    theme: Optional[str] = None,
    eval_run_id: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """List expert assessments with optional filters."""
    sql = """
        SELECT id, expert_id, benchmark_id, theme, query_text, answer, reference,
               eval_run_id, judge_quality, quality_score, rubric_json, notes, status,
               created_at, updated_at
        FROM expert_assessments
        WHERE 1=1
    """
    params: Dict[str, Any] = {"l": max(1, min(int(limit), 2000))}
    if expert_id:
        sql += " AND expert_id=:expert_id"
        params["expert_id"] = expert_id
    if theme:
        sql += " AND theme=:theme"
        params["theme"] = theme
    if eval_run_id:
        sql += " AND eval_run_id=:eval_run_id"
        params["eval_run_id"] = eval_run_id
    sql += " ORDER BY id DESC LIMIT :l"
    with get_engine().connect() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
    out: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        try:
            item["rubric"] = json.loads(item.pop("rubric_json") or "{}")
        except Exception:
            item["rubric"] = {}
        out.append(item)
    return out


def list_assessed_benchmark_ids(expert_id: str, eval_run_id: Optional[str] = None) -> List[str]:
    """Return benchmark ids already assessed by one expert."""
    sql = "SELECT benchmark_id FROM expert_assessments WHERE expert_id=:expert_id"
    params: Dict[str, Any] = {"expert_id": expert_id}
    if eval_run_id:
        sql += " AND eval_run_id=:eval_run_id"
        params["eval_run_id"] = eval_run_id
    else:
        sql += " AND eval_run_id IS NULL"
    with get_engine().connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    return [str(r[0]) for r in rows]


def get_expert_assessment_stats() -> Dict[str, Any]:
    """Aggregate counts for expert review dashboard."""
    try:
        with get_engine().connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT COUNT(*) AS total,
                           COUNT(DISTINCT expert_id) AS experts,
                           COUNT(DISTINCT theme) AS themes,
                           COALESCE(AVG(ABS(quality_score - judge_quality)), 0) AS mae
                    FROM expert_assessments
                    WHERE judge_quality IS NOT NULL
                    """
                )
            ).mappings().first()
        return dict(row or {"total": 0, "experts": 0, "themes": 0, "mae": 0.0})
    except Exception:
        return {"total": 0, "experts": 0, "themes": 0, "mae": 0.0}
