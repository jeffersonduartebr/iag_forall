# Objective: Locust workloads for load, stress, and routing-behavior validation.
"""Locust workloads for load, stress, and routing-behavior validation.

Queries are loaded from data/benchmark_queries/ via benchmark_catalog.py.
"""

from __future__ import annotations

import os
import random
import time

from benchmark_catalog import load_all_queries, load_programming_challenges
from gevent.lock import Semaphore
from locust import HttpUser, between, events, task

QUERIES = load_all_queries()
CHALLENGE_QUERIES = load_programming_challenges()

# ==========================================================
# Classe Locust
# ==========================================================
class RouterUser(HttpUser):
    """Standard user sending random queries."""
    wait_time = between(1, 3)
    host = "http://llm_router_api:8000"

    @task(10)
    def send_query(self):
        """Standard query task."""
        q = random.choice(QUERIES)
        payload = {
            "query": q["query"],
            "enable_rag_for_answer": False,
            "max_tokens": 2048,
            "temperature": 0.3
        }
        self.client.post("/query", json=payload)

    @task(2)
    def health_check(self):
        """Health check task - lower weight."""
        self.client.get("/health")

    @task(1)
    def metrics(self):
        """Metrics endpoint task - lowest weight."""
        self.client.get("/metrics")


class ProgrammingChallengeUser(HttpUser):
    """Expert programming/system-design challenges with low traffic share."""
    wait_time = between(8, 15)
    host = "http://llm_router_api:8000"
    weight = 1

    @task
    def send_challenge(self):
        """High-difficulty expert challenge requiring long, structured answers."""
        q = random.choice(CHALLENGE_QUERIES)
        payload = {
            "query": q["query"],
            "workload_hints": {
                "theme": "programacao_desafios",
                "expected_tokens": 8192,
            },
            "enable_rag_for_answer": False,
            "max_tokens": 8192,
            "temperature": 0.2,
        }
        self.client.post("/query", json=payload)


class RAGUser(HttpUser):
    """User that primarily uses RAG-enabled queries."""
    wait_time = between(2, 5)
    host = "http://llm_router_api:8000"
    weight = 3  # Lower weight compared to standard users

    @task
    def send_rag_query(self):
        """RAG-enabled query task."""
        q = random.choice(QUERIES)
        payload = {
            "query": q["query"],
            "enable_rag_for_answer": True,
            "rag_modality": "text",
            "max_tokens": 2048,
            "temperature": 0.5
        }
        self.client.post("/query", json=payload)


class HighTemperatureUser(HttpUser):
    """Creative user with high temperature settings."""
    wait_time = between(2, 4)
    host = "http://llm_router_api:8000"
    weight = 2

    @task
    def send_creative_query(self):
        """High temperature query for creative tasks."""
        creative_queries = [
            "Escreva um breve poema no estilo barroco sobre a passagem do tempo.",
            "Crie uma história curta sobre um viajante do tempo que visita o Brasil colonial.",
            "Invente uma teoria científica absurda mas que soe plausível.",
            "Escreva um diálogo entre Einstein e Newton sobre física moderna.",
        ]
        payload = {
            "query": random.choice(creative_queries),
            "enable_rag_for_answer": False,
            "max_tokens": 4096,
            "temperature": 1.5  # High temperature for creativity
        }
        self.client.post("/query", json=payload)


class BurstUser(HttpUser):
    """Simulates burst traffic patterns."""
    wait_time = between(0.1, 0.5)  # Very fast requests
    host = "http://llm_router_api:8000"
    weight = 1  # Lowest weight, sporadic bursts

    @task
    def burst_queries(self):
        """Rapid burst of queries."""
        q = random.choice(QUERIES)
        payload = {
            "query": q["query"],
            "enable_rag_for_answer": False,
            "max_tokens": 512,  # Shorter responses for speed
            "temperature": 0.3
        }
        self.client.post("/query", json=payload)


class LongRunningUser(HttpUser):
    """User that sends complex, long-running queries."""
    wait_time = between(5, 10)
    host = "http://llm_router_api:8000"
    weight = 1

    @task
    def send_complex_query(self):
        """Complex query requiring more processing."""
        complex_queries = [
            "Analise detalhadamente o impacto de longo prazo do Tratado de Tordesilhas nas Américas, considerando aspectos políticos, econômicos e culturais.",
            "Desenvolva uma solução completa em Python usando programação orientada a objetos para um sistema de biblioteca, incluindo classes para livros, usuários, empréstimos e reservas.",
            "Compare e contraste as teorias da relatividade especial e geral de Einstein, explicando as implicações de cada uma para nossa compreensão do universo.",
            "Explique o ciclo completo de Krebs em detalhes, incluindo todas as reações, enzimas envolvidas e a produção de ATP.",
        ]
        payload = {
            "query": random.choice(complex_queries),
            "enable_rag_for_answer": True,
            "max_tokens": 8192,
            "temperature": 0.7,
            "timeout_seconds": 180
        }
        self.client.post("/query", json=payload)


class MixedWorkloadUser(HttpUser):
    """Simulates realistic mixed workload with varying request types."""
    wait_time = between(1, 5)
    host = "http://llm_router_api:8000"
    weight = 5  # Most common user type

    @task(7)
    def simple_query(self):
        """Simple, short queries - most common."""
        simple = [
            "O que é Python?",
            "Quanto é 2+2?",
            "Qual a capital do Brasil?",
            "O que é uma API REST?",
            "Explique o conceito de classe.",
        ]
        payload = {
            "query": random.choice(simple),
            "max_tokens": 512,
            "temperature": 0.3
        }
        self.client.post("/query", json=payload)

    @task(2)
    def medium_query(self):
        """Medium complexity queries."""
        q = random.choice(QUERIES)
        payload = {
            "query": q["query"],
            "max_tokens": 2048,
            "temperature": 0.5
        }
        self.client.post("/query", json=payload)

    @task(1)
    def complex_query(self):
        """Complex queries - least common."""
        payload = {
            "query": "Explique detalhadamente como funciona o algoritmo NSGA-II para otimização multiobjetivo e como ele pode ser aplicado em problemas de roteamento de LLMs.",
            "enable_rag_for_answer": True,
            "max_tokens": 4096,
            "temperature": 0.7
        }
        self.client.post("/query", json=payload)


class APIVersionUser(HttpUser):
    """Tests versioned API endpoints."""
    wait_time = between(2, 4)
    host = "http://llm_router_api:8000"
    weight = 2

    @task
    def v1_query(self):
        """Test v1 API endpoint."""
        q = random.choice(QUERIES)
        payload = {
            "query": q["query"],
            "max_tokens": 1024,
            "temperature": 0.3
        }
        self.client.post("/v1/query", json=payload)

    @task
    def v1_health(self):
        """Test v1 health endpoint."""
        self.client.get("/v1/health")


EXACT_QUERY_TARGET = int(os.getenv("EXACT_QUERY_TARGET", "100"))
_exact_query_lock = Semaphore()
_exact_queries_issued = 0
_exact_queries_completed = 0
_exact_queries_inflight = 0
_exact_response_samples = []
_exact_report_printed = False
_exact_sync_successes = 0
_exact_async_accepted = 0
_exact_async_completed = 0
_exact_async_failed = 0
_exact_async_expired = 0
_exact_async_timeouts = 0
_exact_prompt_bank = [
    "Explique em uma frase o que e fotossintese.",
    "Explique em uma frase o que e a Revolucao Francesa.",
    "Explique em uma frase o que e a agua potavel.",
    "Explique em uma frase o que e a biodiversidade.",
    "Explique em uma frase o que e a energia solar.",
]


class ExactHundredQueriesUser(HttpUser):
    """Runs exactly 100 /query requests with a maximum of two concurrent users."""
    def wait_time(self):
        return 0
    host = "http://llm_router_api:8000"
    weight = 1

    _async_poll_interval_seconds = 0.5
    _async_timeout_seconds = float(os.getenv("EXACT_QUERY_ASYNC_TIMEOUT_SECONDS", "60"))

    @staticmethod
    def _safe_json(response):
        """Decode one HTTP response body as JSON without crashing the Locust task."""
        try:
            return response.json()
        except Exception:
            return None

    def _record_sample(self, query_text, status, body, mode):
        """Store a small sample of synchronous and asynchronous responses."""
        with _exact_query_lock:
            if len(_exact_response_samples) < 5:
                _exact_response_samples.append(
                    {
                        "status": status,
                        "query": query_text[:80],
                        "body": body,
                        "mode": mode,
                    }
                )

    def _poll_async_query_job(self, query_text, queued_payload):
        """Poll one queued query job until it completes, fails, expires, or times out."""
        global _exact_async_completed, _exact_async_failed, _exact_async_expired, _exact_async_timeouts

        job_id = queued_payload.get("job_id")
        status_url = queued_payload.get("poll_url") or f"/query/jobs/{job_id}"
        result_url = queued_payload.get("result_url") or f"/query/jobs/{job_id}/result"
        deadline = time.time() + self._async_timeout_seconds

        while time.time() < deadline:
            with self.client.get(status_url, name="/query/jobs/status", catch_response=True) as status_response:
                body = status_response.text[:220] if status_response.text else ""
                if status_response.status_code >= 400:
                    status_response.failure(f"HTTP {status_response.status_code}: {body}")
                    with _exact_query_lock:
                        _exact_async_failed += 1
                    self._record_sample(query_text, status_response.status_code, body, "async-status-failed")
                    return False

                status_payload = self._safe_json(status_response)
                if not isinstance(status_payload, dict):
                    status_response.failure(f"Invalid JSON payload: {body}")
                    with _exact_query_lock:
                        _exact_async_failed += 1
                    self._record_sample(query_text, status_response.status_code, body, "async-status-invalid-json")
                    return False
                job_status = (status_payload.get("status") or "").strip().lower()
                status_response.success()

            if job_status == "completed":
                with self.client.get(result_url, name="/query/jobs/result", catch_response=True) as result_response:
                    body = result_response.text[:220] if result_response.text else ""
                    if result_response.status_code >= 400:
                        result_response.failure(f"HTTP {result_response.status_code}: {body}")
                        with _exact_query_lock:
                            _exact_async_failed += 1
                        self._record_sample(query_text, result_response.status_code, body, "async-result-failed")
                        return False

                    result_payload = self._safe_json(result_response)
                    if not isinstance(result_payload, dict):
                        result_response.failure(f"Invalid JSON payload: {body}")
                        with _exact_query_lock:
                            _exact_async_failed += 1
                        self._record_sample(query_text, result_response.status_code, body, "async-result-invalid-json")
                        return False
                    answer = (result_payload.get("answer") or "").strip()
                    abstained = bool(result_payload.get("abstained"))
                    if answer or abstained:
                        result_response.success()
                        with _exact_query_lock:
                            _exact_async_completed += 1
                        self._record_sample(query_text, result_response.status_code, body, "async-completed")
                        return True

                    result_response.failure(f"Invalid final payload: {body}")
                    with _exact_query_lock:
                        _exact_async_failed += 1
                    self._record_sample(query_text, result_response.status_code, body, "async-invalid-result")
                    return False

            if job_status == "failed":
                with _exact_query_lock:
                    _exact_async_failed += 1
                self._record_sample(query_text, status_response.status_code, body, "async-failed")
                return False

            if job_status == "expired":
                with _exact_query_lock:
                    _exact_async_expired += 1
                self._record_sample(query_text, status_response.status_code, body, "async-expired")
                return False

            time.sleep(self._async_poll_interval_seconds)

        with _exact_query_lock:
            _exact_async_timeouts += 1
        self._record_sample(query_text, 408, "Async job polling timeout", "async-timeout")
        return False

    @task
    def run_exact_queries(self):
        """Reserve and execute one query until the global target is reached."""
        global _exact_queries_issued, _exact_queries_completed, _exact_queries_inflight
        global _exact_sync_successes, _exact_async_accepted

        with _exact_query_lock:
            if _exact_queries_issued >= EXACT_QUERY_TARGET:
                if (
                    _exact_queries_issued >= EXACT_QUERY_TARGET
                    and _exact_queries_inflight <= 0
                    and self.environment.runner
                ):
                    self.environment.runner.quit()
                return
            index = _exact_queries_issued
            _exact_queries_issued += 1
            _exact_queries_inflight += 1

        query_text = _exact_prompt_bank[index % len(_exact_prompt_bank)]
        payload = {
            "query": query_text,
            "enable_rag_for_answer": False,
            "max_tokens": 64,
            "temperature": 0.1,
            "timeout_seconds": 30,
        }

        with self.client.post("/query", json=payload, name="/query", catch_response=True) as response:
            body = response.text[:220] if response.text else ""
            if response.status_code >= 400:
                response.failure(f"HTTP {response.status_code}: {body}")
                self._record_sample(query_text, response.status_code, body, "sync-failed")
            elif response.status_code == 202:
                response.success()
                queued_payload = self._safe_json(response)
                if not isinstance(queued_payload, dict):
                    response.failure(f"Invalid queue JSON payload: {body}")
                    self._record_sample(query_text, response.status_code, body, "async-accepted-invalid-json")
                    queued_payload = None
                with _exact_query_lock:
                    _exact_async_accepted += 1
                self._record_sample(query_text, response.status_code, body, "async-accepted")
                if queued_payload is not None:
                    self._poll_async_query_job(query_text, queued_payload)
            else:
                payload_json = self._safe_json(response)
                if not isinstance(payload_json, dict):
                    response.failure(f"Invalid JSON payload: {body}")
                    self._record_sample(query_text, response.status_code, body, "sync-invalid-json")
                    payload_json = None
                if payload_json is None:
                    pass
                else:
                    answer = (payload_json.get("answer") or "").strip()
                    abstained = bool(payload_json.get("abstained"))
                    if answer or abstained:
                        response.success()
                        with _exact_query_lock:
                            _exact_sync_successes += 1
                        self._record_sample(query_text, response.status_code, body, "sync-completed")
                    else:
                        response.failure(f"Invalid final payload: {body}")
                        self._record_sample(query_text, response.status_code, body, "sync-invalid-result")

            with _exact_query_lock:
                _exact_queries_completed += 1
                _exact_queries_inflight = max(0, _exact_queries_inflight - 1)
                should_stop = (
                    _exact_queries_issued >= EXACT_QUERY_TARGET
                    and _exact_queries_inflight <= 0
                )

        if should_stop and self.environment.runner:
            self.environment.runner.quit()


@events.quitting.add_listener
def _report_exact_query_samples(environment, **_kwargs):
    """Print a small sample of responses from the exact-100 run."""
    global _exact_report_printed
    if _exact_report_printed or not _exact_response_samples:
        return
    _exact_report_printed = True
    print(
        "[ExactHundredQueriesUser] summary "
        f"sync_successes={_exact_sync_successes} "
        f"async_accepted={_exact_async_accepted} "
        f"async_completed={_exact_async_completed} "
        f"async_failed={_exact_async_failed} "
        f"async_expired={_exact_async_expired} "
        f"async_timeouts={_exact_async_timeouts}"
    )
    print("[ExactHundredQueriesUser] response samples:")
    for idx, sample in enumerate(_exact_response_samples, start=1):
        print(
            f"[ExactHundredQueriesUser] sample#{idx} status={sample['status']} "
            f"mode={sample['mode']!r} query={sample['query']!r} body={sample['body']!r}"
        )
