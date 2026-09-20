#!/usr/bin/env python3
# Objective: Interactive CLI for human review of LLM pre-classifications, producing the golden ground truth.
"""Human anchoring of the pre-classified corpus.

Usage::

    python3 -m scripts.benchmark_v2.review_cli
    python3 -m scripts.benchmark_v2.review_cli --only-conflicts   # just the contested items

Items are presented in order of how much your decision is worth: first where the
two classifiers disagree, then where they disagree with the generator's own
``steps_required``, then the perturbed items, then the rest. Reviewing in that
order means the session can be stopped at any point and the labels that were
actually in doubt are already settled.

Keys: ``Enter`` accepts the proposed label, ``b``/``m``/``a`` overrides it,
``n`` adds a note, ``f`` flags a broken item, ``s`` skips, ``q`` saves and exits.
State lives in ``review_state.json`` and the session is resumable.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "scripts.benchmark_v2"

from .agreement import LABEL_INDEX, review_priority  # noqa: E402
from .schema import read_jsonl  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "benchmark_v2"
DEFAULT_CORPUS = DATA / "corpus.jsonl"
DEFAULT_PRECLASS = DATA / "preclassification.jsonl"
DEFAULT_STATE = DATA / "review_state.json"
DEFAULT_OUT = DATA / "ground_truth.jsonl"

KEY_TO_LABEL = {"b": "BAIXA", "m": "MEDIA", "a": "ALTA"}

WIDTH = 96
RULE = "=" * WIDTH


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------


def wrap(text: str, indent: str = "  ") -> str:
    """Wrap a block of text to the review width."""
    return "\n".join(
        textwrap.fill(line, WIDTH, initial_indent=indent, subsequent_indent=indent) or indent
        for line in str(text).splitlines()
    )


def render(item: Dict[str, Any], verdicts: List[Dict[str, Any]], position: str, expand: bool) -> str:
    """Build the screen shown for one item."""
    statement = item["query"] if expand else shorten(item["query"])
    lines = [
        RULE,
        f"{position}  {item['id']}  [{item['discipline']} / {item['topic']}]",
        f"particao: {item['partition']}   passos do gerador: {item['steps_required']} "
        f"({item['complexity_gold']})   ancorado: {'SIM' if item.get('requires_rag') else 'NAO'}",
        "-" * WIDTH,
        wrap(statement),
        "-" * WIDTH,
    ]
    for verdict in verdicts:
        if verdict.get("error"):
            lines.append(f"  [{verdict['model']}] FALHOU: {verdict['error']}")
            continue
        lines.append(
            f"  [{verdict['model']}] -> {verdict['rotulo_complexidade_final']}"
            f"   ancoragem: {verdict['dependencia_ancoragem']}"
        )
        lines.append(wrap(f"inferencia: {verdict['analise_inferencia']}", "     "))
        lines.append(wrap(f"distratores: {verdict['analise_distratores']}", "     "))
    return "\n".join(lines)


def shorten(text: str, limit: int = 700) -> str:
    """Trim a long statement, keeping both ends so the question stays visible."""
    if len(text) <= limit:
        return text
    head, tail = text[: limit // 2], text[-limit // 2 :]
    return f"{head}\n  [... {len(text) - limit} caracteres omitidos, tecla 'x' mostra tudo ...]\n{tail}"


def proposed_label(verdicts: Sequence[Dict[str, Any]], item: Dict[str, Any]) -> str:
    """The label offered on Enter: the classifiers' when they agree, else the generator's."""
    labels = [v["rotulo_complexidade_final"] for v in verdicts if v.get("rotulo_complexidade_final")]
    if labels and len(set(labels)) == 1:
        return labels[0]
    if labels:
        # Contested: offer the median of the proposals, which is the least
        # committal choice and still one that a classifier actually made.
        ordered = sorted(labels, key=lambda label: LABEL_INDEX[label])
        return ordered[len(ordered) // 2]
    return item["complexity_gold"]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


def load_state(path: Path) -> Dict[str, Any]:
    """Read the resumable review state."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"decisions": {}}


def save_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def write_ground_truth(
    path: Path, items_by_id: Dict[str, Dict[str, Any]], decisions: Dict[str, Dict[str, Any]]
) -> int:
    """Write the reviewed corpus: every item, carrying the human label where there is one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for item_id in sorted(items_by_id):
            decision = decisions.get(item_id)
            if not decision:
                continue
            record = dict(items_by_id[item_id])
            record["complexity_human"] = decision["label"]
            record["review_source"] = decision["source"]
            record["review_note"] = decision.get("note") or ""
            record["flagged"] = bool(decision.get("flagged"))
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

HELP = (
    "Enter=aceitar  b/m/a=BAIXA/MEDIA/ALTA  n=nota  f=marcar defeituoso  "
    "x=ver enunciado completo  s=pular  q=salvar e sair"
)


def _decide(decisions: Dict[str, Dict[str, Any]], item_id: str, label: str, source: str, **extra: Any) -> None:
    """Record a decision, keeping anything already attached to the item.

    Notes are written before the label is chosen, so replacing the entry here
    would throw away what the reviewer just typed.
    """
    entry = decisions.setdefault(item_id, {})
    entry.update({"label": label, "source": source, **extra})


def review_session(
    queue: List[str],
    items_by_id: Dict[str, Dict[str, Any]],
    verdicts_by_item: Dict[str, List[Dict[str, Any]]],
    state: Dict[str, Any],
    state_path: Path,
    reader=input,
) -> Dict[str, Any]:
    """Run the interactive loop; returns the updated state."""
    decisions: Dict[str, Dict[str, Any]] = state.setdefault("decisions", {})
    total = len(queue)

    for index, item_id in enumerate(queue, start=1):
        item = items_by_id[item_id]
        verdicts = verdicts_by_item.get(item_id, [])
        proposal = proposed_label(verdicts, item)
        expand = False

        while True:
            print(render(item, verdicts, f"[{index}/{total}]", expand))
            print(f"  proposta: {proposal}")
            print(f"  {HELP}")
            try:
                answer = reader("  > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\ninterrompido; estado salvo.")
                save_state(state_path, state)
                return state

            if answer == "x":
                expand = True
                continue
            if answer == "q":
                save_state(state_path, state)
                return state
            if answer == "s":
                break
            if answer == "n":
                note = reader("  nota: ").strip()
                _decide(decisions, item_id, proposal, "aceito", note=note)
                save_state(state_path, state)
                continue
            if answer == "f":
                _decide(decisions, item_id, proposal, "defeituoso", flagged=True)
                save_state(state_path, state)
                break
            if answer in KEY_TO_LABEL:
                _decide(decisions, item_id, KEY_TO_LABEL[answer], "humano")
                save_state(state_path, state)
                break
            if answer == "":
                _decide(decisions, item_id, proposal, "aceito")
                save_state(state_path, state)
                break
            print("  tecla desconhecida.")

    save_state(state_path, state)
    return state


def final_report(
    items_by_id: Dict[str, Dict[str, Any]],
    verdicts_by_item: Dict[str, List[Dict[str, Any]]],
    decisions: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Human-versus-LLM agreement, broken down by partition.

    High agreement on canonical items with low agreement on the verbosity traps
    is the result the trap partition exists to produce: it means the padding,
    not the logic, drove the classifier's label.
    """
    from .agreement import _kappa

    per_partition: Dict[str, List[tuple]] = {}
    for item_id, decision in decisions.items():
        item = items_by_id.get(item_id)
        verdicts = [v for v in verdicts_by_item.get(item_id, []) if v.get("rotulo_complexidade_final")]
        if not item or not verdicts:
            continue
        llm_label = proposed_label(verdicts, item)
        per_partition.setdefault(item["partition"], []).append(
            (LABEL_INDEX[llm_label], LABEL_INDEX[decision["label"]])
        )

    report: Dict[str, Any] = {
        "reviewed": len(decisions),
        "overridden": sum(1 for d in decisions.values() if d["source"] == "humano"),
        "flagged": sum(1 for d in decisions.values() if d.get("flagged")),
        "labels": dict(Counter(d["label"] for d in decisions.values())),
        "kappa_human_vs_llm": {},
    }
    for partition, pairs in sorted(per_partition.items()):
        llm, human = zip(*pairs)
        report["kappa_human_vs_llm"][partition] = _kappa(llm, human)
    return report


def build_queue(
    records: List[Dict[str, Any]],
    items_by_id: Dict[str, Dict[str, Any]],
    models: Sequence[str],
    decisions: Dict[str, Any],
    only_conflicts: bool,
) -> List[str]:
    """Order the pending items by review value, skipping what is already decided."""
    ranked = review_priority(records, items_by_id, models)
    queue = [item_id for item_id, rank, _ in ranked if item_id not in decisions and (rank <= 1 or not only_conflicts)]
    return queue


def main() -> None:
    parser = argparse.ArgumentParser(description="Revisao humana das pre-classificacoes do benchmark v2.")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--preclassification", type=Path, default=DEFAULT_PRECLASS)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--only-conflicts", action="store_true", help="revisa apenas itens contestados")
    parser.add_argument("--report-only", action="store_true", help="nao revisa, so imprime o relatorio")
    args = parser.parse_args()

    items_by_id = {item["id"]: item for item in read_jsonl(args.corpus)}
    records = list(read_jsonl(args.preclassification)) if args.preclassification.exists() else []
    if not records:
        parser.error(f"nenhuma pre-classificacao em {args.preclassification}; rode preclassify.py antes")

    verdicts_by_item: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        verdicts_by_item.setdefault(record["item_id"], []).append(record)
    models = sorted({record["model"] for record in records})

    state = load_state(args.state)
    decisions = state.setdefault("decisions", {})

    if not args.report_only:
        queue = build_queue(records, items_by_id, models, decisions, args.only_conflicts)
        if queue:
            print(f"{len(queue)} item(ns) na fila, por ordem de valor de revisao.\n")
            state = review_session(queue, items_by_id, verdicts_by_item, state, args.state)
            decisions = state["decisions"]
        else:
            print("nada pendente na fila.")

    written = write_ground_truth(args.out, items_by_id, decisions)
    report = final_report(items_by_id, verdicts_by_item, decisions)
    report["ground_truth_items"] = written
    print("\n" + RULE)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(f"ground truth em {args.out}")


if __name__ == "__main__":
    main()
