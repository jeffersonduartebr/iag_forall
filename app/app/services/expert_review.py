# Objective: Human expert review queue and judge agreement metrics.
"""Expert portal backend — theme-based review items and kappa metrics."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from app.benchmark_catalog import list_themes_summary, load_catalog_entries
from app.benchmark_splits import filter_entries_by_split, resolve_catalog_split
from app.roadmap_features import (
    create_expert_assessment,
    get_expert_account_by_email,
    get_expert_profile,
    list_assessed_benchmark_ids,
    list_eval_run_results,
    list_expert_assessments,
    upsert_expert_profile,
)
from app.services.academic_stats import cohens_kappa


def ensure_expert_profile(user_id: str) -> Dict[str, Any]:
    """Return profile, creating one from account data when missing."""
    profile = get_expert_profile(user_id)
    account = get_expert_account_by_email(user_id)
    if profile:
        return _merge_account_into_profile(profile, account)
    display_name = str(account.get("display_name") if account else user_id)
    upsert_expert_profile(user_id, display_name=display_name, theme_ids=[])
    profile = get_expert_profile(user_id) or {"user_id": user_id, "theme_ids": [], "display_name": display_name}
    return _merge_account_into_profile(profile, account)


def _merge_account_into_profile(profile: Dict[str, Any], account: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(profile)
    if account:
        out["email"] = account.get("email")
        out["phone"] = account.get("phone")
        out["account_enabled"] = bool(account.get("enabled"))
        if not out.get("display_name"):
            out["display_name"] = account.get("display_name")
    return out


def update_expert_profile(
    user_id: str,
    *,
    display_name: Optional[str] = None,
    theme_ids: Optional[List[str]] = None,
    credentials_note: Optional[str] = None,
) -> Dict[str, Any]:
    """Update expert areas of expertise."""
    upsert_expert_profile(
        user_id,
        display_name=display_name,
        theme_ids=theme_ids,
        credentials_note=credentials_note,
    )
    return ensure_expert_profile(user_id)


def _catalog_review_pool(
    theme_ids: List[str],
    *,
    split: Optional[str] = None,
    seed: Optional[int] = None,
    require_reference: bool = True,
) -> List[Dict[str, Any]]:
    pool: List[Dict[str, Any]] = []
    for theme in theme_ids:
        rows = load_catalog_entries(theme=theme)
        rows = filter_entries_by_split(rows, split, seed=seed)
        for row in rows:
            if require_reference and not row.get("reference"):
                continue
            entry_id = str(row.get("id") or "").strip()
            pool.append(
                {
                    "source": "catalog",
                    "benchmark_id": entry_id,
                    "theme": str(row.get("theme") or theme),
                    "query_text": str(row.get("query") or ""),
                    "reference": row.get("reference"),
                    "split": resolve_catalog_split(entry_id, seed=seed),
                    "difficulty": row.get("difficulty"),
                    "tags": row.get("tags") or [],
                }
            )
    return pool


def _eval_review_pool(
    eval_run_id: str,
    theme_ids: List[str],
) -> List[Dict[str, Any]]:
    rows = list_eval_run_results(eval_run_id, limit=5000)
    pool: List[Dict[str, Any]] = []
    for row in rows:
        meta = row.get("metadata") or {}
        theme = str(meta.get("benchmark_theme") or "")
        if theme_ids and theme not in theme_ids:
            continue
        benchmark_id = str(meta.get("benchmark_id") or "").strip()
        if not benchmark_id:
            continue
        pool.append(
            {
                "source": "eval",
                "benchmark_id": benchmark_id,
                "theme": theme,
                "query_text": str(row.get("prompt_text") or ""),
                "answer": str(meta.get("answer") or ""),
                "reference": meta.get("reference"),
                "judge_quality": float(row.get("quality", 0.0) or 0.0),
                "eval_run_id": eval_run_id,
                "model": row.get("model"),
                "metadata": meta,
            }
        )
    return pool


def get_next_review_item(
    expert_id: str,
    *,
    eval_run_id: Optional[str] = None,
    split: Optional[str] = "held_out",
    seed: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Return the next unreviewed item for an expert within their theme areas."""
    profile = ensure_expert_profile(expert_id)
    theme_ids = [str(t) for t in (profile.get("theme_ids") or []) if str(t).strip()]
    if not theme_ids:
        return None

    assessed = set(list_assessed_benchmark_ids(expert_id, eval_run_id=eval_run_id))

    if eval_run_id:
        pool = _eval_review_pool(eval_run_id, theme_ids)
        for item in pool:
            if item["benchmark_id"] not in assessed:
                return item
        return None

    pool = _catalog_review_pool(theme_ids, split=split, seed=seed)
    rng = random.Random(seed)
    rng.shuffle(pool)
    for item in pool:
        if item["benchmark_id"] not in assessed:
            return item
    return None


def submit_expert_assessment(
    expert_id: str,
    *,
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
) -> Dict[str, Any]:
    """Persist expert analysis and return saved record summary."""
    create_expert_assessment(
        expert_id=expert_id,
        benchmark_id=benchmark_id,
        theme=theme,
        query_text=query_text,
        answer=answer,
        reference=reference,
        eval_run_id=eval_run_id,
        judge_quality=judge_quality,
        quality_score=quality_score,
        rubric=rubric,
        notes=notes,
    )
    items = list_expert_assessments(expert_id=expert_id, limit=1)
    return items[0] if items else {"status": "saved"}


def expert_judge_agreement_report(
    *,
    eval_run_id: Optional[str] = None,
    theme: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute Cohen's kappa between human experts and automated judge scores."""
    assessments = list_expert_assessments(eval_run_id=eval_run_id, theme=theme, limit=2000)
    pairs = [
        a
        for a in assessments
        if a.get("judge_quality") is not None and a.get("quality_score") is not None
    ]
    if len(pairs) < 2:
        return {"kappa": None, "n": len(pairs), "method": "insufficient_pairs"}

    def _bucket(score: float) -> int:
        if score < 4.0:
            return 0
        if score < 6.0:
            return 1
        if score < 8.0:
            return 2
        return 3

    human = [_bucket(float(a["quality_score"])) for a in pairs]
    judge = [_bucket(float(a["judge_quality"])) for a in pairs]
    kappa = cohens_kappa(human, judge)
    abs_errors = [abs(float(a["quality_score"]) - float(a["judge_quality"])) for a in pairs]
    return {
        **kappa,
        "mean_absolute_error": sum(abs_errors) / len(abs_errors),
        "pairs": len(pairs),
        "by_theme": _kappa_by_theme(pairs),
    }


def _kappa_by_theme(pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_theme: Dict[str, List[Dict[str, Any]]] = {}
    for item in pairs:
        by_theme.setdefault(str(item.get("theme") or "unknown"), []).append(item)

    out: Dict[str, Any] = {}
    for theme, rows in by_theme.items():

        def _bucket(score: float) -> int:
            if score < 4.0:
                return 0
            if score < 6.0:
                return 1
            if score < 8.0:
                return 2
            return 3

        human = [_bucket(float(r["quality_score"])) for r in rows]
        judge = [_bucket(float(r["judge_quality"])) for r in rows]
        out[theme] = cohens_kappa(human, judge)
    return out


def list_available_themes() -> List[Dict[str, Any]]:
    """Return benchmark themes for expert area selection."""
    return list_themes_summary()


def build_expert_kappa_dashboard(*, eval_run_id: Optional[str] = None) -> Dict[str, Any]:
    """Build dashboard payload for judge vs human agreement by theme."""
    from app.roadmap_features import get_expert_assessment_stats

    report = expert_judge_agreement_report(eval_run_id=eval_run_id)
    stats = get_expert_assessment_stats()
    theme_titles = {str(t["id"]): str(t.get("title") or t["id"]) for t in list_themes_summary()}

    by_theme_raw = report.get("by_theme") or {}
    themes_table: List[Dict[str, Any]] = []
    for theme_id, metrics in sorted(by_theme_raw.items(), key=lambda x: x[0]):
        if not isinstance(metrics, dict):
            continue
        themes_table.append(
            {
                "theme_id": theme_id,
                "theme_title": theme_titles.get(theme_id, theme_id),
                "kappa": metrics.get("kappa"),
                "n": metrics.get("n", 0),
                "observed_agreement": metrics.get("observed_agreement"),
            }
        )

    return {
        "eval_run_id": eval_run_id,
        "global_kappa": report.get("kappa"),
        "global_n": report.get("pairs", report.get("n", 0)),
        "mean_absolute_error": report.get("mean_absolute_error"),
        "total_assessments": int(stats.get("total") or 0),
        "active_experts": int(stats.get("experts") or 0),
        "themes_reviewed": int(stats.get("themes") or 0),
        "by_theme": themes_table,
        "insufficient_data": report.get("kappa") is None,
    }
