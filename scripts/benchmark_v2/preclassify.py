#!/usr/bin/env python3
# Objective: Async pipeline that pre-classifies benchmark v2 items with frontier LLMs via OpenRouter.
"""Pre-classify every corpus item with one or more frontier models.

Usage::

    # pilot: 20 items, both classifiers, cost reported at the end
    python3 -m scripts.benchmark_v2.preclassify --limit 20

    # full run
    python3 -m scripts.benchmark_v2.preclassify

    # exercise the whole pipeline with no API calls and no cost
    python3 -m scripts.benchmark_v2.preclassify --dry-run

Calls go through ``app.providers_async.call_model``, so they inherit the retry
policy the router itself uses (``COMMON_RETRY_STRATEGY`` in
``app/app/providers/_infra.py``: five attempts, ``wait_random_exponential``
capped at 60s) plus the provider circuit breakers. This script only adds what
that layer cannot know about: a repair attempt for malformed JSON, and a
resumable output so a run can be interrupted at any point.

Results are appended to JSONL as they arrive, keyed by ``(item_id, model)``.
Re-running skips pairs already recorded without an error.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "scripts.benchmark_v2"

from .agreement import summarize  # noqa: E402
from .schema import read_jsonl  # noqa: E402

LOG = logging.getLogger("benchmark_v2.preclassify")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = ROOT / "data" / "benchmark_v2" / "corpus.jsonl"
DEFAULT_OUT = ROOT / "data" / "benchmark_v2" / "preclassification.jsonl"

#: Both reached through OpenRouter. The ``openrouter/`` prefix is stripped by
#: ``providers_async.call_model`` and the remainder is sent as the model id.
DEFAULT_MODELS = (
    "openrouter/anthropic/claude-sonnet-4.5",
    "openrouter/openai/gpt-5.6",
)

REQUIRED_FIELDS = (
    "analise_inferencia",
    "analise_distratores",
    "dependencia_ancoragem",
    "rotulo_complexidade_final",
)

VALID_LABELS = {"BAIXA", "MEDIA", "ALTA"}
VALID_ANCHORING = {"SIM", "NAO"}


SYSTEM_PROMPT = """És um avaliador ortogonal de complexidade cognitiva. A tua função é classificar perguntas educacionais isolando a verdadeira exigência lógica da mera verbosidade do texto.
Deves preencher o seguinte esquema JSON para cada item:
{
"analise_inferencia": "Descrição estrita dos passos lógicos necessários para a resolução, ignorando o volume de caracteres do enunciado.",
"analise_distratores": "Identificação e descarte explícito de sentenças no enunciado que não alteram a resposta analítica final.",
"dependencia_ancoragem": "Avaliação binária (SIM/NAO): a resposta exige documento externo (RAG) para ser validada segundo um critério específico, ou resolve-se por conhecimento paramétrico universal?",
"rotulo_complexidade_final": "BAIXA | MEDIA | ALTA"
}
Critérios do Rótulo Final:

    •    BAIXA: Evocação direta, tradução simples ou aplicação de fórmula única. Um só passo lógico.
    •    MEDIA: Aplicação sequencial de 2 a 3 conceitos estruturados (ex: conversão de base + cálculo de memória).
    •    ALTA: Raciocínio não-linear, decomposição de sub-hipóteses e modelação abstrata (ex: desenho conceptual de base de dados)."""


class SchemaError(ValueError):
    """The model replied with something that is not the requested schema."""


# ---------------------------------------------------------------------------
# Prompting and parsing
# ---------------------------------------------------------------------------


def build_user_prompt(item: Dict[str, Any]) -> str:
    """Present one item for classification, with no hint about its true depth."""
    return (
        "Classifica o item abaixo e responde exclusivamente com o objeto JSON do esquema.\n\n"
        f"[DISCIPLINA] {item['discipline']}\n"
        f"[ENUNCIADO]\n{item['query']}"
    )


def extract_json(raw: str) -> Dict[str, Any]:
    """Pull the JSON object out of a reply that may be fenced or padded with prose."""
    text = raw.strip()
    fenced = re.findall(r"```(?:json)?\s*(.+?)```", text, flags=re.S | re.I)
    if fenced:
        text = fenced[-1].strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise SchemaError("nenhum objeto JSON na resposta")
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise SchemaError(f"JSON invalido: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SchemaError("o JSON de topo nao e um objeto")
    return parsed


def validate_classification(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Check the four fields and normalize the two controlled vocabularies."""
    missing = [field for field in REQUIRED_FIELDS if field not in parsed]
    if missing:
        raise SchemaError(f"campos ausentes: {', '.join(missing)}")

    anchoring = str(parsed["dependencia_ancoragem"]).strip().upper()
    anchoring = "SIM" if anchoring.startswith("SIM") else "NAO" if anchoring.startswith(("NAO", "NÃO", "NO")) else anchoring
    if anchoring not in VALID_ANCHORING:
        raise SchemaError(f"dependencia_ancoragem invalida: {parsed['dependencia_ancoragem']!r}")

    label = str(parsed["rotulo_complexidade_final"]).strip().upper().replace("É", "E").replace("Ó", "O")
    if label not in VALID_LABELS:
        raise SchemaError(f"rotulo_complexidade_final invalido: {parsed['rotulo_complexidade_final']!r}")

    return {
        "analise_inferencia": str(parsed["analise_inferencia"]).strip(),
        "analise_distratores": str(parsed["analise_distratores"]).strip(),
        "dependencia_ancoragem": anchoring,
        "rotulo_complexidade_final": label,
    }


# ---------------------------------------------------------------------------
# Model calls
# ---------------------------------------------------------------------------

CallFn = Callable[[str, str, Optional[str]], Awaitable[tuple]]


async def real_call(model: str, prompt: str, repair_hint: Optional[str] = None) -> tuple:
    """Call the model through the router's own provider layer."""
    from app.providers_async import call_model

    user_prompt = prompt
    if repair_hint:
        user_prompt = (
            f"{prompt}\n\n[CORRECAO] A resposta anterior foi rejeitada: {repair_hint}. "
            "Responde apenas com o objeto JSON valido do esquema, sem texto em volta."
        )
    return await call_model(
        model=model,
        prompt=user_prompt,
        system_prompt=SYSTEM_PROMPT,
        temperature=0.0,
        max_tokens=900,
        response_format={"type": "json_object"},
    )


def make_dry_run_call(items_by_id: Dict[str, Any]) -> CallFn:
    """A deterministic stub that answers from the corpus itself.

    It exists to exercise the pipeline — parsing, resume, reporting — without
    spending money. Its labels are the generator's own, so the reports it
    produces show perfect agreement and must never be mistaken for a real run.
    """

    async def _call(model: str, prompt: str, repair_hint: Optional[str] = None) -> tuple:
        match = re.search(r"\[ENUNCIADO\]\n(.+)", prompt, flags=re.S)
        statement = (match.group(1) if match else prompt).strip()
        item = next(
            (i for i in items_by_id.values() if i["query"].strip() == statement),
            None,
        )
        payload = {
            "analise_inferencia": "stub: rotulo derivado do proprio corpus",
            "analise_distratores": "stub: nenhum distrator analisado",
            "dependencia_ancoragem": "SIM" if item and item.get("requires_rag") else "NAO",
            "rotulo_complexidade_final": item["complexity_gold"] if item else "MEDIA",
        }
        return json.dumps(payload, ensure_ascii=False), {"cost_per_1k": 0.0, "prompt_tokens": 0, "completion_tokens": 0}

    return _call


async def classify_one(item: Dict[str, Any], model: str, call: CallFn) -> Dict[str, Any]:
    """Classify one item, with a single repair attempt on a malformed reply."""
    prompt = build_user_prompt(item)
    started = time.monotonic()
    attempts = 0
    hint: Optional[str] = None
    last_raw = ""

    for attempt in range(2):
        attempts = attempt + 1
        try:
            raw, meta = await call(model, prompt, hint)
            last_raw = raw or ""
            parsed = validate_classification(extract_json(last_raw))
            LOG.debug("%s/%s ok em %d tentativa(s)", item["id"], model, attempts)
            return {
                "item_id": item["id"],
                "model": model,
                **parsed,
                "attempts": attempts,
                "latency_s": round(time.monotonic() - started, 3),
                "cost_usd": float(meta.get("cost_per_1k") or 0.0),
                "prompt_tokens": meta.get("prompt_tokens"),
                "completion_tokens": meta.get("completion_tokens"),
                "error": None,
            }
        except SchemaError as exc:
            hint = str(exc)
            LOG.warning("%s/%s resposta fora do esquema (%s)", item["id"], model, exc)
        except Exception as exc:  # provider failure after its own retries
            LOG.error("%s/%s falhou: %s: %s", item["id"], model, type(exc).__name__, exc)
            return _failure(item, model, started, attempts, f"{type(exc).__name__}: {exc}", last_raw)

    return _failure(item, model, started, attempts, f"schema: {hint}", last_raw)


def _failure(item, model, started, attempts, error, raw) -> Dict[str, Any]:
    """Record a classification that could not be produced; it goes to the human queue."""
    return {
        "item_id": item["id"],
        "model": model,
        "analise_inferencia": None,
        "analise_distratores": None,
        "dependencia_ancoragem": None,
        "rotulo_complexidade_final": None,
        "attempts": attempts,
        "latency_s": round(time.monotonic() - started, 3),
        "cost_usd": 0.0,
        "error": error,
        "needs_human": True,
        "raw": (raw or "")[:2000],
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def load_done(path: Path) -> set:
    """``(item_id, model)`` pairs already classified without error."""
    if not path.exists():
        return set()
    done = set()
    for record in read_jsonl(path):
        if not record.get("error"):
            done.add((record["item_id"], record["model"]))
    return done


async def run(
    items: Sequence[Dict[str, Any]],
    models: Sequence[str],
    out_path: Path,
    call: CallFn,
    concurrency: int,
) -> List[Dict[str, Any]]:
    """Classify every (item, model) pair not already done, appending as we go."""
    done = load_done(out_path)
    pending = [(item, model) for model in models for item in items if (item["id"], model) not in done]
    LOG.info("%d par(es) pendente(s); %d ja concluido(s)", len(pending), len(done))

    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []
    completed = 0

    async def worker(item: Dict[str, Any], model: str) -> None:
        nonlocal completed
        async with semaphore:
            record = await classify_one(item, model, call)
        async with write_lock:
            with out_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            results.append(record)
            completed += 1
            if completed % 25 == 0 or completed == len(pending):
                LOG.info("progresso: %d/%d", completed, len(pending))

    await asyncio.gather(*(worker(item, model) for item, model in pending))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-classifica o corpus de benchmark v2 com LLMs de fronteira.")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--limit", type=int, default=0, help="classifica apenas os N primeiros itens (piloto de custo)")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="executa o pipeline sem chamar nenhuma API")
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None, help="grava o relatorio de concordancia em JSON")
    args = parser.parse_args()

    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.DEBUG if args.dry_run else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=handlers,
    )

    items = list(read_jsonl(args.corpus))
    if args.limit:
        items = items[: args.limit]
    items_by_id = {item["id"]: item for item in items}

    if args.dry_run:
        models = ["dry-run-a", "dry-run-b"]
        call: CallFn = make_dry_run_call(items_by_id)
        LOG.warning("modo dry-run: rotulos copiados do corpus, nao use para analise real")
    else:
        models = list(args.models)
        if not os.getenv("OPENROUTER_API_KEY"):
            parser.error("OPENROUTER_API_KEY nao esta definida; use --dry-run para testar o pipeline")
        call = real_call

    started = time.monotonic()
    asyncio.run(run(items, models, args.out, call, args.concurrency))

    all_records = list(read_jsonl(args.out))
    report = summarize(all_records, items_by_id, models)
    report["elapsed_s"] = round(time.monotonic() - started, 1)

    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
