# -*- coding: utf-8 -*-
# Objective: Test coverage for smoke behavior and regressions.
"""
test_smoke.py — Smoke Tests for LLM Router
-------------------------------------------
Comprehensive smoke tests covering multiple knowledge domains.
These tests validate that the system can handle diverse queries correctly.

Run manually:
    pytest tests/test_smoke.py -v

Or programmatically:
    from tests.test_smoke import run_smoke_tests
    results = await run_smoke_tests()
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional

import pytest

logger = logging.getLogger(__name__)
requires_openai = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY is required for smoke/integration tests",
)

# ==============================================================================
# Smoke Test Queries - Covering 10+ Knowledge Domains
# ==============================================================================

SMOKE_TEST_QUERIES = [
    # 1. Technology/Programming
    {
        "query": "Explique a diferença entre processamento síncrono e assíncrono em Python.",
        "domain": "technology",
        "min_length": 100,
    },
    # 2. History/Geopolitics
    {
        "query": "Quais foram as principais consequências geopolíticas da Queda do Muro de Berlim?",
        "domain": "history",
        "min_length": 100,
    },
    # 3. Physics/Science
    {
        "query": "Como funciona o princípio da incerteza de Heisenberg em termos simples?",
        "domain": "physics",
        "min_length": 80,
    },
    # 4. Literature/Arts
    {
        "query": "Escreva um breve poema no estilo barroco sobre a passagem do tempo.",
        "domain": "literature",
        "min_length": 50,
    },
    # 5. Mathematics
    {
        "query": "Se um trem viaja a 120km/h, quanto tempo leva para percorrer 450km?",
        "domain": "math",
        "min_length": 20,
    },
    # 6. Biology
    {
        "query": "Descreva o processo de fotossíntese e sua importância para a vida na Terra.",
        "domain": "biology",
        "min_length": 100,
    },
    # 7. Economics
    {
        "query": "Qual a relação entre taxa de juros e inflação em uma economia de mercado?",
        "domain": "economics",
        "min_length": 80,
    },
    # 8. Philosophy
    {
        "query": "Resuma o conceito de 'Imperativo Categórico' de Immanuel Kant.",
        "domain": "philosophy",
        "min_length": 80,
    },
    # 9. Law
    {
        "query": "O que são direitos fundamentais segundo a Constituição Brasileira de 1988?",
        "domain": "law",
        "min_length": 80,
    },
    # 10. Culture/Geography
    {
        "query": "Quais são as principais características culturais e culinárias do Nordeste brasileiro?",
        "domain": "culture",
        "min_length": 80,
    },
    # 11. Geology
    {
        "query": "Explique como ocorreu a separação dos continentes ao longo dos milênios.",
        "domain": "geology",
        "min_length": 80,
    },
    # 12. Geometry
    {
        "query": "Se eu tenho um triângulo retângulo cujos valores dos catetos são 8 e 10, qual é o valor da hipotenusa?",
        "domain": "geometry",
        "min_length": 20,
    },
    # 13. History/Economics
    {
        "query": "Cite três características do mercantilismo.",
        "domain": "history_economics",
        "min_length": 50,
    },
]


# ==============================================================================
# Smoke Test Runner
# ==============================================================================

async def run_single_smoke_test(
    query: str,
    domain: str,
    min_length: int,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """
    Run a single smoke test query.

    Args:
        query: The query to test
        domain: Knowledge domain category
        min_length: Minimum expected response length
        timeout: Maximum time to wait for response

    Returns:
        Dict with test results
    """
    # Import here to avoid circular dependencies
    try:
        from app.router_core import route_and_answer
    except ImportError:
        return {
            "query": query,
            "domain": domain,
            "success": False,
            "error": "Could not import route_and_answer",
            "latency_s": 0,
        }

    start = time.time()
    result = {
        "query": query,
        "domain": domain,
        "success": False,
        "error": None,
        "latency_s": 0,
        "answer_length": 0,
        "model": None,
    }

    try:
        response = await asyncio.wait_for(
            route_and_answer(query=query, modality="text", use_cache=False),
            timeout=timeout,
        )

        latency = time.time() - start
        result["latency_s"] = round(latency, 3)

        if response:
            answer = response.get("answer", "")
            result["answer_length"] = len(answer)
            result["model"] = response.get("model")

            # Validate response
            if len(answer) >= min_length:
                result["success"] = True
            else:
                result["error"] = f"Response too short: {len(answer)} < {min_length}"
        else:
            result["error"] = "Empty response"

    except asyncio.TimeoutError:
        result["error"] = f"Timeout after {timeout}s"
        result["latency_s"] = timeout
    except Exception as e:
        result["error"] = str(e)
        result["latency_s"] = time.time() - start

    return result


async def run_smoke_tests(
    queries: Optional[List[Dict]] = None,
    parallel: bool = False,
    timeout_per_query: float = 120.0,
) -> Dict[str, Any]:
    """
    Run all smoke tests.

    Args:
        queries: List of query dicts (uses defaults if None)
        parallel: Whether to run tests in parallel
        timeout_per_query: Timeout for each query

    Returns:
        Dict with overall results and individual test results
    """
    if queries is None:
        queries = SMOKE_TEST_QUERIES

    results = {
        "total": len(queries),
        "passed": 0,
        "failed": 0,
        "tests": [],
        "total_latency_s": 0,
        "avg_latency_s": 0,
    }

    start_time = time.time()

    if parallel:
        # Run all tests in parallel
        tasks = [
            run_single_smoke_test(
                q["query"], q["domain"], q["min_length"], timeout_per_query
            )
            for q in queries
        ]
        test_results = await asyncio.gather(*tasks, return_exceptions=True)

        for tr in test_results:
            if isinstance(tr, Exception):
                results["failed"] += 1
                results["tests"].append({"error": str(tr), "success": False})
            else:
                results["tests"].append(tr)
                if tr.get("success"):
                    results["passed"] += 1
                else:
                    results["failed"] += 1
    else:
        # Run tests sequentially
        for q in queries:
            tr = await run_single_smoke_test(
                q["query"], q["domain"], q["min_length"], timeout_per_query
            )
            results["tests"].append(tr)
            if tr.get("success"):
                results["passed"] += 1
            else:
                results["failed"] += 1

    # Calculate totals
    total_latency = sum(t.get("latency_s", 0) for t in results["tests"])
    results["total_latency_s"] = round(total_latency, 3)
    results["avg_latency_s"] = round(total_latency / len(queries), 3) if queries else 0
    results["wall_time_s"] = round(time.time() - start_time, 3)

    return results


# ==============================================================================
# Pytest Tests
# ==============================================================================

@pytest.mark.asyncio
@pytest.mark.integration
@requires_openai
async def test_smoke_technology():
    """Test technology/programming query."""
    result = await run_single_smoke_test(
        query="Explique a diferença entre processamento síncrono e assíncrono em Python.",
        domain="technology",
        min_length=50,
        timeout=120.0,
    )
    assert result["success"], f"Failed: {result.get('error')}"


@pytest.mark.asyncio
@pytest.mark.integration
@requires_openai
async def test_smoke_math():
    """Test math query."""
    result = await run_single_smoke_test(
        query="Se um trem viaja a 120km/h, quanto tempo leva para percorrer 450km?",
        domain="math",
        min_length=10,
        timeout=120.0,
    )
    assert result["success"], f"Failed: {result.get('error')}"


@pytest.mark.asyncio
@pytest.mark.integration
@requires_openai
async def test_smoke_science():
    """Test science query."""
    result = await run_single_smoke_test(
        query="Descreva o processo de fotossíntese.",
        domain="biology",
        min_length=50,
        timeout=120.0,
    )
    assert result["success"], f"Failed: {result.get('error')}"


@pytest.mark.asyncio
@pytest.mark.slow
@pytest.mark.integration
@requires_openai
async def test_smoke_full_suite():
    """Run full smoke test suite (slow)."""
    results = await run_smoke_tests(parallel=False)
    assert results["passed"] >= results["total"] * 0.8, (
        f"Smoke tests failed: {results['passed']}/{results['total']} passed"
    )


# ==============================================================================
# CLI Entry Point
# ==============================================================================

if __name__ == "__main__":
    import sys

    async def main():
        print("Running smoke tests...")
        results = await run_smoke_tests()

        print(f"\nResults: {results['passed']}/{results['total']} passed")
        print(f"Total time: {results['wall_time_s']}s")
        print(f"Avg latency: {results['avg_latency_s']}s")

        for i, test in enumerate(results["tests"], 1):
            status = "PASS" if test.get("success") else "FAIL"
            domain = test.get("domain", "unknown")
            latency = test.get("latency_s", 0)
            error = test.get("error", "")

            if test.get("success"):
                print(f"  [{status}] {i}. {domain} ({latency:.2f}s)")
            else:
                print(f"  [{status}] {i}. {domain}: {error}")

        sys.exit(0 if results["failed"] == 0 else 1)

    asyncio.run(main())
