# Objective: Test coverage for query workload profiling and reliability annotations.
"""Workload classification, runtime profile (RAG mode, timeouts) and reliability hints.

Split from test_query_runtime.py, which keeps the request orchestration tests.
"""

from types import SimpleNamespace


def _base_result():
    return {
        "answer": "raw-answer",
        "model": "ollama/test",
        "modality": "text",
        "latency_s": 0.2,
        "cost_per_1k": 0.01,
        "metadata": {"raw_payload": "{}", "prompt_tokens": 11, "completion_tokens": 7, "uncertainty_score": 0.2},
        "route": {
            "chosen_model": "ollama/test",
            "objectives": {},
            "pareto_front": [],
            "explanation": "",
            "fallback": {"used": False, "models_tried": [], "errors": []},
        },
        "candidates": [],
    }


def _request(**overrides):
    values = {
        "query": "Como funciona?",
        "tenant_id": "school-1",
        "modality": "text",
        "image_b64": None,
        "images": [],
        "policy_version": None,
        "experiment_id": None,
        "user_key": None,
        "system_prompt": "",
        "enable_rag_for_answer": False,
        "enable_rag_for_image": False,
        "max_tokens": None,
        "temperature": None,
        "rag_modality": "text",
        "use_cache": True,
        "timeout_seconds": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _settings_map(monkeypatch, qr, mapping):
    """Patch dynamic settings lookups through the stable get() surface."""
    monkeypatch.setattr(qr.settings, "get", lambda key, fallback=None: mapping.get(key, fallback))


def test_enrich_result_reliability_marks_low_confidence_answer_for_review():
    """Low-confidence, unsupported answers should be abstained and routed to review."""
    from app.services import query_runtime as qr

    result = _base_result()
    result["answer"] = "talvez"
    result["metadata"].update(
        {
            "uncertainty_score": 0.92,
            "grounded": False,
            "retrieval_mode": "light_retrieval",
            "workload_class": "knowledge_lookup",
            "guardrail_output_tags": [],
        }
    )
    enriched = qr.enrich_result_reliability(result)

    assert enriched["abstained"] is True
    assert enriched["verification_status"] == "unsupported"
    assert enriched["review_status"] == "needs_review"
    assert "evidencia suficiente" in enriched["answer"].lower()


def test_classify_query_workload_recognizes_fast_and_heavy_paths():
    """Workload classification should distinguish simple, retrieval-heavy, reasoning, and vision traffic."""
    from app.services import query_runtime as qr

    assert (
        qr.classify_query_workload(_request(query="Quanto é 2+2?"), modality="text", image_input=None) == "simple_text"
    )
    assert (
        qr.classify_query_workload(
            _request(query="Cite a policy e o documento de referência sobre avaliação escolar."),
            modality="text",
            image_input=None,
        )
        == "knowledge_lookup"
    )
    assert (
        qr.classify_query_workload(
            _request(query="Explique passo a passo como resolver uma equação do segundo grau."),
            modality="text",
            image_input=None,
        )
        == "reasoning"
    )
    assert (
        qr.classify_query_workload(_request(query="O que há na imagem?"), modality="vision", image_input="img")
        == "vision"
    )


def test_apply_query_runtime_profile_bypasses_rag_for_simple_text(monkeypatch):
    """Simple text traffic should use the fast path when performance mode and bypass are enabled."""
    from app.services import query_runtime as qr

    _settings_map(
        monkeypatch,
        qr,
        {
            "ROUTER_PERF_MODE": "1",
            "ROUTER_SIMPLE_QUERY_MAX_TOKENS": "128",
            "RAG_SIMPLE_QUERY_BYPASS_ENABLED": "1",
        },
    )
    profile = qr.apply_query_runtime_profile(
        _request(query="Quanto é 2+2?", enable_rag_for_answer=True, max_tokens=600),
        modality="text",
        image_input=None,
    )

    assert profile["workload_class"] == "simple_text"
    assert profile["perf_mode_enabled"] is True
    assert profile["use_rag"] is False
    # O piso de tokens por complexidade (apply_complexity_runtime_adjustments) eleva
    # o cap de simple_text (128) para o mínimo da complexidade detectada (256).
    assert profile["max_tokens"] == 256
    assert profile["runtime_hints"]["retrieval_mode"] == "no_retrieval"
    assert profile["runtime_hints"]["max_fallbacks"] == 1
    assert profile["runtime_hints"]["needs_retrieval"] is False
    assert profile["runtime_hints"]["interactive_priority"] == "high"
    assert profile["runtime_hints"]["provider_timeout_seconds"] == 20
    # Prazo por vazão: 4 s + 1.3 × (256 + 4096 de raciocínio) / 40 tok/s + 45 s de reserva para o fallback.
    assert profile["runtime_hints"]["sync_deadline_seconds"] == 190


def test_apply_query_runtime_profile_keeps_vision_rag(monkeypatch):
    """Vision traffic should not silently lose retrieval just because simple-query bypass is enabled."""
    from app.services import query_runtime as qr

    _settings_map(
        monkeypatch,
        qr,
        {
            "ROUTER_PERF_MODE": "1",
            "ROUTER_SIMPLE_QUERY_MAX_TOKENS": "64",
            "RAG_SIMPLE_QUERY_BYPASS_ENABLED": "1",
        },
    )
    profile = qr.apply_query_runtime_profile(
        _request(query="O que aparece aqui?", enable_rag_for_image=True, max_tokens=256),
        modality="vision",
        image_input="img",
    )

    assert profile["workload_class"] == "vision"
    assert profile["use_rag"] is True
    # Piso de tokens por complexidade eleva 256 -> 512 para a query de visão detectada.
    assert profile["max_tokens"] == 512
    assert profile["runtime_hints"]["retrieval_mode"] == "full_retrieval"


def test_apply_query_runtime_profile_uses_light_retrieval_for_knowledge_lookup(monkeypatch):
    """Knowledge-lookup traffic should request a lighter retrieval profile by default."""
    from app.services import query_runtime as qr

    _settings_map(
        monkeypatch,
        qr,
        {
            "ROUTER_PERF_MODE": "1",
            "RAG_LIGHT_TOP_K": "2",
            "RAG_LIGHT_CONTEXT_TOKEN_BUDGET": "320",
            "RERANK_ENABLED_FOR_LIGHT_RETRIEVAL": "0",
        },
    )
    profile = qr.apply_query_runtime_profile(
        _request(
            query="Cite a policy oficial sobre recuperação paralela.", enable_rag_for_answer=False, max_tokens=700
        ),
        modality="text",
        image_input=None,
    )

    assert profile["workload_class"] == "knowledge_lookup"
    assert profile["use_rag"] is True
    assert profile["runtime_hints"]["retrieval_mode"] == "light_retrieval"
    assert profile["runtime_hints"]["rag_top_k"] == 2
    assert profile["runtime_hints"]["rag_context_token_budget"] == 320
    assert profile["runtime_hints"]["rag_rerank_enabled"] is False
    assert profile["runtime_hints"]["needs_retrieval"] is True
    assert profile["runtime_hints"]["interactive_priority"] == "high"
    # O multiplicador/bônus de deadline por complexidade ajusta 35 -> 36 e 40 -> 47.
    assert profile["runtime_hints"]["provider_timeout_seconds"] == 36
    assert profile["runtime_hints"]["sync_deadline_seconds"] == 204  # idem, com o max_tokens maior deste perfil


def test_apply_query_runtime_profile_promotes_source_seeking_queries_to_full_retrieval(monkeypatch):
    """Source-heavy knowledge lookups should keep full retrieval despite the cheap path defaults."""
    from app.services import query_runtime as qr

    _settings_map(
        monkeypatch,
        qr,
        {
            "ROUTER_PERF_MODE": "1",
            "RAG_LIGHT_TOP_K": "2",
            "RAG_LIGHT_CONTEXT_TOKEN_BUDGET": "320",
            "RAG_FULL_CONTEXT_TOKEN_BUDGET": "900",
        },
    )
    profile = qr.apply_query_runtime_profile(
        _request(query="Cite as fontes e o artigo oficial sobre avaliação formativa.", enable_rag_for_answer=False),
        modality="text",
        image_input=None,
    )

    assert profile["workload_class"] == "knowledge_lookup"
    assert profile["runtime_hints"]["retrieval_mode"] == "full_retrieval"
    assert profile["runtime_hints"]["rag_context_token_budget"] == 900
    assert profile["runtime_hints"]["needs_rerank"] is True


def test_effective_sync_timeout_clamps_request_override_to_workload_deadline():
    """Client timeout overrides should not stretch a fast workload beyond its sync budget."""
    from app.services import query_runtime as qr

    runtime_profile = {
        "runtime_hints": {
            "sync_deadline_seconds": 25,
        }
    }

    assert qr._effective_sync_timeout_seconds(30, runtime_profile) == 25
    assert qr._effective_sync_timeout_seconds(10, runtime_profile) == 10
    assert qr._effective_sync_timeout_seconds(None, runtime_profile) == 25


def test_classify_query_workload_respects_disabled_classifier(monkeypatch):
    """Disabling the classifier should fall back to the safest reasoning profile."""
    from app.services import query_runtime as qr

    _settings_map(monkeypatch, qr, {"ROUTER_QUERY_CLASSIFIER_ENABLED": "0"})
    assert qr.classify_query_workload(_request(query="Quanto é 2+2?"), modality="text", image_input=None) == "reasoning"


def test_enrich_result_reliability_marks_supported_grounded_answers():
    """Grounded confident answers should remain supported and auto-approved."""
    from app.services import query_runtime as qr

    result = _base_result()
    result["answer"] = "Resposta com apoio documental."
    result["metadata"].update({"uncertainty_score": 0.1, "grounded": True, "retrieval_mode": "full_retrieval"})
    enriched = qr.enrich_result_reliability(result)
    assert enriched["verification_status"] == "supported"
    assert enriched["review_status"] == "auto_approved"
    assert enriched["abstained"] is False
