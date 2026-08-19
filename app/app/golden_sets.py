# Objective: Application runtime code for curated golden-set evaluation prompts.
"""Built-in golden sets and gate heuristics for repeatable evaluation runs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

GOLDEN_SETS: Dict[str, Dict[str, Any]] = {
    "education_core_v1": {
        "id": "education_core_v1",
        "title": "Education Core v1",
        "description": "Cross-domain school-oriented prompts that stress factuality and concise explanations.",
        "gates": {
            "quality_mean_min": 5.5,
            "unsupported_rate_max": 0.20,
            "abstain_rate_max": 0.35,
            "grounded_rate_min": 0.20,
        },
        "items": [
            {
                "id": "bio_photosynthesis",
                "prompt": "Explique a fotossintese de forma simples para alunos do ensino fundamental.",
                "reference": "A fotossintese converte luz, agua e gas carbonico em glicose e oxigenio.",
                "require_evidence": False,
                "criticality": "medium",
            },
            {
                "id": "math_fraction",
                "prompt": "Como explicar a diferenca entre 1/2 e 1/3 para uma turma do 6o ano?",
                "reference": "Metades representam partes maiores do que terços quando o todo e o mesmo.",
                "require_evidence": False,
                "criticality": "medium",
            },
            {
                "id": "history_french_revolution",
                "prompt": "Resuma as causas principais da Revolucao Francesa.",
                "reference": "Crise fiscal, desigualdade social e influencia do Iluminismo.",
                "require_evidence": False,
                "criticality": "high",
            },
            {
                "id": "geography_water_cycle",
                "prompt": "Descreva o ciclo da agua em etapas curtas.",
                "reference": "Evaporacao, condensacao, precipitacao e infiltracao/escoamento.",
                "require_evidence": False,
                "criticality": "medium",
            },
        ],
    }
}


def list_golden_sets() -> List[Dict[str, Any]]:
    """Return summary information for all built-in golden sets."""
    out: List[Dict[str, Any]] = []
    for golden_set in GOLDEN_SETS.values():
        out.append(
            {
                "id": golden_set["id"],
                "title": golden_set["title"],
                "description": golden_set["description"],
                "item_count": len(golden_set.get("items") or []),
                "gates": dict(golden_set.get("gates") or {}),
            }
        )
    return out


def get_golden_set(golden_set_id: str) -> Optional[Dict[str, Any]]:
    """Fetch one golden-set definition by identifier."""
    golden_set = GOLDEN_SETS.get(golden_set_id)
    if not golden_set:
        return None
    return {
        "id": golden_set["id"],
        "title": golden_set["title"],
        "description": golden_set["description"],
        "gates": dict(golden_set.get("gates") or {}),
        "items": [dict(item) for item in golden_set.get("items") or []],
    }


def get_golden_set_prompts(golden_set_id: str) -> List[str]:
    """Return only prompt texts for one golden set."""
    golden_set = get_golden_set(golden_set_id)
    if not golden_set:
        return []
    return [str(item.get("prompt") or "") for item in golden_set.get("items") or [] if str(item.get("prompt") or "").strip()]


def evaluate_golden_set_gate(results: List[Dict[str, Any]], gates: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Evaluate whether one eval result set satisfies the configured quality gates."""
    gates = dict(gates or {})
    total = max(1, len(results))
    unsupported = 0
    abstained = 0
    grounded = 0
    quality_values: List[float] = []

    for result in results:
        metadata = result.get("metadata") or {}
        quality_values.append(float(result.get("quality") or 0.0))
        if str(metadata.get("verification_status") or "") == "unsupported":
            unsupported += 1
        if bool(metadata.get("abstained")):
            abstained += 1
        if bool(metadata.get("grounded")):
            grounded += 1

    quality_mean = (sum(quality_values) / len(quality_values)) if quality_values else 0.0
    unsupported_rate = unsupported / total
    abstain_rate = abstained / total
    grounded_rate = grounded / total

    checks = {
        "quality_mean": quality_mean >= float(gates.get("quality_mean_min", 0.0)),
        "unsupported_rate": unsupported_rate <= float(gates.get("unsupported_rate_max", 1.0)),
        "abstain_rate": abstain_rate <= float(gates.get("abstain_rate_max", 1.0)),
        "grounded_rate": grounded_rate >= float(gates.get("grounded_rate_min", 0.0)),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "quality_mean": quality_mean,
            "unsupported_rate": unsupported_rate,
            "abstain_rate": abstain_rate,
            "grounded_rate": grounded_rate,
            "n": len(results),
        },
        "gates": gates,
    }
