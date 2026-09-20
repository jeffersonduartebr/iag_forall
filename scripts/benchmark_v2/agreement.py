# Objective: Agreement statistics and cross-checks between LLM pre-classifications and generator ground truth.
"""What the pre-classification is actually measuring.

Two classifiers labelling the same items give three useful signals:

* **agreement between them** (Cohen's kappa) tells us how well-posed the label
  is at all, and picks the items worth a human's attention first;
* **agreement with ``steps_required``**, which the generator knows exactly,
  measures whether a classifier recovers the logical depth of a question;
* the same comparison **restricted to the verbosity traps** measures whether the
  padding works. A classifier that scores the traps as harder than their
  one-step bases has been fooled by length — which is the effect the partition
  was built to produce.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

APP_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "app")
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

LABELS = ("BAIXA", "MEDIA", "ALTA")
LABEL_INDEX = {label: i for i, label in enumerate(LABELS)}


def _kappa(a: Sequence[int], b: Sequence[int]) -> Dict[str, Any]:
    """Cohen's kappa, reusing the implementation the eval reports already use."""
    from app.services.academic_stats import cohens_kappa

    return cohens_kappa(list(a), list(b))


def paired_labels(
    records: Iterable[Dict[str, Any]], model_a: str, model_b: str
) -> Tuple[List[str], List[str], List[str]]:
    """Line up the two classifiers' labels on the items both of them answered."""
    by_item: Dict[str, Dict[str, str]] = {}
    for record in records:
        label = record.get("rotulo_complexidade_final")
        if label in LABEL_INDEX:
            by_item.setdefault(record["item_id"], {})[record["model"]] = label

    item_ids, labels_a, labels_b = [], [], []
    for item_id in sorted(by_item):
        pair = by_item[item_id]
        if model_a in pair and model_b in pair:
            item_ids.append(item_id)
            labels_a.append(pair[model_a])
            labels_b.append(pair[model_b])
    return item_ids, labels_a, labels_b


def classifier_agreement(records: List[Dict[str, Any]], model_a: str, model_b: str) -> Dict[str, Any]:
    """Kappa between the two classifiers, plus the items where they disagree."""
    item_ids, labels_a, labels_b = paired_labels(records, model_a, model_b)
    if not item_ids:
        return {"kappa": None, "n": 0, "disagreements": [], "method": "no_paired_items"}
    stats = _kappa([LABEL_INDEX[x] for x in labels_a], [LABEL_INDEX[x] for x in labels_b])
    disagreements = [
        {"item_id": item_id, model_a: left, model_b: right}
        for item_id, left, right in zip(item_ids, labels_a, labels_b)
        if left != right
    ]
    return {**stats, "disagreements": disagreements}


def agreement_with_generator(
    records: List[Dict[str, Any]], items_by_id: Dict[str, Dict[str, Any]], model: Optional[str] = None
) -> Dict[str, Any]:
    """Compare a classifier's labels against the label implied by ``steps_required``."""
    pairs: List[Tuple[int, int]] = []
    for record in records:
        if model and record["model"] != model:
            continue
        label = record.get("rotulo_complexidade_final")
        item = items_by_id.get(record["item_id"])
        if label in LABEL_INDEX and item:
            pairs.append((LABEL_INDEX[item["complexity_gold"]], LABEL_INDEX[label]))
    if not pairs:
        return {"kappa": None, "n": 0, "method": "no_items"}
    gold, predicted = zip(*pairs)
    exact = sum(1 for g, p in pairs if g == p) / len(pairs)
    overestimated = sum(1 for g, p in pairs if p > g) / len(pairs)
    return {
        **_kappa(gold, predicted),
        "exact_match": round(exact, 4),
        "overestimated_share": round(overestimated, 4),
    }


def trap_effect(records: List[Dict[str, Any]], items_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Measure whether the verbosity padding inflated the perceived complexity.

    Each trap is compared with its own base item, labelled by the same
    classifier, so the only difference between the two is the padding.
    """
    labels: Dict[Tuple[str, str], str] = {}
    for record in records:
        label = record.get("rotulo_complexidade_final")
        if label in LABEL_INDEX:
            labels[(record["model"], record["item_id"])] = label

    inflated = same = deflated = 0
    examples: List[Dict[str, str]] = []
    for (model, item_id), label in sorted(labels.items()):
        item = items_by_id.get(item_id)
        if not item or item.get("partition") != "verbosity_trap":
            continue
        base_label = labels.get((model, item.get("base_id")))
        if base_label is None:
            continue
        delta = LABEL_INDEX[label] - LABEL_INDEX[base_label]
        if delta > 0:
            inflated += 1
            if len(examples) < 10:
                examples.append({"model": model, "trap": item_id, "base": item["base_id"], "label": label})
        elif delta == 0:
            same += 1
        else:
            deflated += 1

    total = inflated + same + deflated
    return {
        "n_pairs": total,
        "inflated": inflated,
        "unchanged": same,
        "deflated": deflated,
        "inflation_rate": round(inflated / total, 4) if total else None,
        "examples": examples,
    }


def anchoring_confusion(
    records: List[Dict[str, Any]], items_by_id: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Confusion matrix of ``dependencia_ancoragem`` against the corpus's ``requires_rag``."""
    counts: Counter = Counter()
    mistakes: List[Dict[str, Any]] = []
    for record in records:
        verdict = (record.get("dependencia_ancoragem") or "").strip().upper()
        item = items_by_id.get(record["item_id"])
        if verdict not in {"SIM", "NAO"} or not item:
            continue
        truth = "SIM" if item.get("requires_rag") else "NAO"
        counts[(truth, verdict)] += 1
        if truth != verdict and len(mistakes) < 20:
            mistakes.append({"item_id": record["item_id"], "model": record["model"], "esperado": truth, "obtido": verdict})

    total = sum(counts.values())
    correct = counts[("SIM", "SIM")] + counts[("NAO", "NAO")]
    return {
        "n": total,
        "accuracy": round(correct / total, 4) if total else None,
        "true_positive": counts[("SIM", "SIM")],
        "false_negative": counts[("SIM", "NAO")],
        "false_positive": counts[("NAO", "SIM")],
        "true_negative": counts[("NAO", "NAO")],
        "mistakes": mistakes,
    }


def review_priority(
    records: List[Dict[str, Any]], items_by_id: Dict[str, Dict[str, Any]], models: Sequence[str]
) -> List[Tuple[str, int, str]]:
    """Order items by how much a human decision on them is worth.

    Disagreement between classifiers comes first (the label is genuinely
    contested), then disagreement with the generator (one side is wrong), then
    the perturbed items, then everything else.
    """
    by_item: Dict[str, Dict[str, str]] = {}
    for record in records:
        label = record.get("rotulo_complexidade_final")
        if label in LABEL_INDEX:
            by_item.setdefault(record["item_id"], {})[record["model"]] = label

    ranked: List[Tuple[str, int, str]] = []
    for item_id, labels in by_item.items():
        item = items_by_id.get(item_id, {})
        distinct = set(labels.values())
        if len(distinct) > 1:
            ranked.append((item_id, 0, "classificadores discordam"))
        elif item.get("complexity_gold") and distinct and item["complexity_gold"] not in distinct:
            ranked.append((item_id, 1, "diverge do gerador"))
        elif item.get("partition") == "semantic_outlier":
            ranked.append((item_id, 2, "outlier semantico"))
        elif item.get("partition") == "verbosity_trap":
            ranked.append((item_id, 3, "armadilha de verbosidade"))
        else:
            ranked.append((item_id, 4, "rotina"))
    ranked.sort(key=lambda row: (row[1], row[0]))
    return ranked


def summarize(
    records: List[Dict[str, Any]], items_by_id: Dict[str, Dict[str, Any]], models: Sequence[str]
) -> Dict[str, Any]:
    """Full report over a finished pre-classification run."""
    report: Dict[str, Any] = {
        "n_records": len(records),
        "by_model": dict(Counter(r["model"] for r in records)),
        "failures": sum(1 for r in records if r.get("error")),
        "per_model_vs_generator": {
            model: agreement_with_generator(records, items_by_id, model) for model in models
        },
        "trap_effect": trap_effect(records, items_by_id),
        "anchoring": anchoring_confusion(records, items_by_id),
        "cost_usd": round(sum(float(r.get("cost_usd") or 0.0) for r in records), 4),
    }
    if len(models) >= 2:
        report["classifier_agreement"] = classifier_agreement(records, models[0], models[1])
    return report
