# Objective: Service-layer helpers for router execution.
"""Fast-path routing implementation extracted from router_core."""

from __future__ import annotations

import time
from typing import Any, Dict

from app.model_registry import filter_configured_model_names, filter_tool_capable_model_names
from app.native_tools import filter_native_tool_capable_model_names, split_tools
from app.services.adversarial_governance import advgov_escalate


def _deadline_remaining_seconds(runtime_hints: Dict[str, Any] | None) -> float:
    """Return the remaining synchronous request budget."""
    deadline_ts = float((runtime_hints or {}).get("request_deadline_ts", 0.0) or 0.0)
    if deadline_ts <= 0:
        return float("inf")
    return max(0.0, deadline_ts - time.monotonic())


def _effective_provider_timeout_seconds(runtime_hints: Dict[str, Any] | None) -> float | None:
    """Clamp one provider timeout to the remaining synchronous deadline."""
    configured_timeout = float((runtime_hints or {}).get("provider_timeout_seconds", 0.0) or 0.0)
    remaining = _deadline_remaining_seconds(runtime_hints)
    if remaining == float("inf"):
        return configured_timeout or None
    budget = max(1.0, remaining - 0.5)
    if configured_timeout <= 0:
        return budget
    return max(1.0, min(configured_timeout, budget))


def _setting_value(settings: Any, key: str, default: Any) -> Any:
    """Return one setting from either the dynamic settings facade or a simple stub object."""
    getter = getattr(settings, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(settings, key, default)


def _should_bypass_rag(
    *,
    deps: Dict[str, Any],
    query: str,
    modality: str,
    image_b64: str | None,
    use_rag: bool,
    uncertainty_score: float,
    runtime_hints: Dict[str, Any] | None,
) -> bool:
    """Return whether retrieval should be skipped for the current request."""
    if not use_rag:
        return True
    if image_b64 or modality in {"vision", "multimodal"}:
        return False
    retrieval_mode = (runtime_hints or {}).get("retrieval_mode")
    if retrieval_mode == "no_retrieval":
        return True
    if (runtime_hints or {}).get("needs_retrieval") is False:
        return True
    if retrieval_mode in {"light_retrieval", "full_retrieval"}:
        return False
    if str(_setting_value(deps["settings"], "RAG_SIMPLE_QUERY_BYPASS_ENABLED", "1")).strip() != "1":
        return False

    query_text = str(query or "").strip()
    token_count = len(query_text.split())
    if uncertainty_score > 0.45:
        return False
    if token_count <= 18 and len(query_text) <= 140:
        return True
    return False


async def route_and_answer_internal_impl(
    *,
    deps: Dict[str, Any],
    query: str,
    system_prompt: str,
    use_rag: bool,
    max_tokens: int | None,
    temperature: float | None,
    modality: str,
    image_b64: str | None,
    rag_modality: str,
    use_cache: bool,
    runtime_hints: Dict[str, Any] | None = None,
    tenant_id: str | None = None,
    tools: list | None = None,
    tool_choice: Any = None,
    messages: list | None = None,
    response_format: dict | None = None,
) -> Dict[str, Any]:
    """Execute the synchronous routing path using injected dependencies."""
    start_time = time.time()
    tools_requested = bool(tools)
    # Turnos de tool (com tools declaradas ou histórico multi-turn) não são
    # cacheáveis: a resposta depende de resultados de tool específicos do turno.
    if tools_requested or messages:
        use_cache = False

    stage_timings_ms: Dict[str, float] = {}

    def _observe_stage(stage: str, started_at: float) -> None:
        elapsed = time.time() - started_at
        # Per-request breakdown surfaced in diagnostics for single-request triage.
        stage_timings_ms[stage] = round(stage_timings_ms.get(stage, 0.0) + elapsed * 1000.0, 3)
        try:
            deps["ROUTER_STAGE_LATENCY"].labels(stage=stage).observe(elapsed)
        except Exception:
            pass

    modality = deps["normalize_modality"](modality, image_b64)
    max_tokens = max_tokens or deps["settings"].MAX_TOKENS_DEFAULT
    temperature = temperature or deps["settings"].TEMPERATURE_DEFAULT

    if use_cache:
        cached = None
        cache_started_at = time.time()
        try:
            if deps["_dep_cache_breaker"].current_state != "open":
                try:
                    cached = await deps["_dep_cache_breaker"].call_async(
                        deps["check_cache"],
                        query,
                        modality=modality,
                        image_b64=image_b64,
                        tenant_id=tenant_id,
                    )
                except Exception:
                    cached = await deps["check_cache"](
                        query, modality=modality, image_b64=image_b64, tenant_id=tenant_id
                    )
            deps["_record_dependency_breaker_metrics"]()
        except Exception:
            try:
                deps["DEPENDENCY_FAILURES"].labels(dependency="cache").inc()
            except Exception:
                pass
            deps["_record_dependency_breaker_metrics"]()
        finally:
            _observe_stage("cache_lookup", cache_started_at)

        if cached:
            deps["logger"].info(f"[router] Cache HIT ({cached.get('similarity', 0):.2f})")
            try:
                deps["ROUTER_ROUTE_COST"].labels(route_type="cache").inc(0.0)
            except Exception:
                pass
            try:
                deps["ROUTER_ATTEMPTS_PER_QUERY"].observe(1)
            except Exception:
                pass
            return {
                "model": "semantic_cache",
                "modality": modality,
                "answer": cached.get("text", ""),
                "image_output_b64": cached.get("image_output_b64"),
                "latency_s": round(time.time() - start_time, 3),
                "estimated_cost_usd": 0.0,
                "cost_per_1k": 0.0,
                "metadata": {"cached": True, "stage_timings_ms": dict(stage_timings_ms)},
                "route": {
                    "chosen_model": "semantic_cache",
                    "objectives": {"latency": 0, "cost": 0, "uncertainty": 0},
                    "pareto_front": [],
                    "explanation": "Cache",
                    "fallback": {"used": False, "models_tried": ["semantic_cache"], "errors": []},
                },
                "candidates": [],
            }

    uncertainty_score = 0.5
    precheck_started_at = time.time()
    try:
        if deps["_dep_uq_breaker"].current_state != "open":
            uncertainty_score = deps["_dep_uq_breaker"].call(deps["get_uncertainty_score"], query, modality)
        deps["_record_dependency_breaker_metrics"]()
    except Exception as exc:
        try:
            deps["DEPENDENCY_FAILURES"].labels(dependency="uq").inc()
        except Exception:
            pass
        deps["_record_dependency_breaker_metrics"]()
        deps["logger"].warning(f"[router] UQ fail: {exc}")
    finally:
        _observe_stage("precheck", precheck_started_at)

    selection_started_at = time.time()
    all_candidates = (
        deps["settings"].CANDIDATE_MODELS_LIST
        + deps["settings"].CANDIDATE_VISION_MODELS_LIST
        + deps["settings"].CANDIDATE_MULTIMODAL_MODELS_LIST
    )
    valid_models = list(
        set(
            [
                model
                for model in all_candidates
                if isinstance(model, str) and not any(model.startswith(prefix) for prefix in deps["BLOCKED_PREFIXES"])
            ]
        )
    )
    valid_models = filter_configured_model_names(valid_models)
    if "apply_ollama_performance_preferences" in deps:
        try:
            valid_models = deps["apply_ollama_performance_preferences"](valid_models, runtime_hints=runtime_hints)
        except Exception as exc:
            deps["logger"].warning(f"[router] Failed to apply Ollama performance preferences: {exc}")

    if (runtime_hints or {}).get("prefer_cloud_models"):
        cloud_models = [model for model in valid_models if not str(model).startswith("ollama/")]
        local_models = [model for model in valid_models if str(model).startswith("ollama/")]
        if cloud_models:
            valid_models = cloud_models + local_models

    if not valid_models:
        cloud_fallback = [
            m
            for m in (deps["settings"].CANDIDATE_MODELS_LIST or [])
            if isinstance(m, str) and m.startswith(("openrouter/", "openai/", "anthropic/", "gemini/"))
        ]
        valid_models = filter_configured_model_names(cloud_fallback)
    if not valid_models:
        valid_models = ["ollama/phi4:latest"]
    else:
        budget_check_async = deps.get("_is_error_budget_exceeded_async")
        if budget_check_async is not None:
            exceeded = await budget_check_async()
        else:
            exceeded = deps["_is_error_budget_exceeded"]()
        if exceeded:
            local_models = [model for model in valid_models if model.startswith("ollama/")]
            if local_models:
                valid_models = local_models
                deps["logger"].warning("[router] Error budget exceeded; forcing local-only candidate set")

    # Roteamento estrito por capacidade: se o cliente pediu tools, restringe os
    # candidatos aos modelos com suporte a function calling. Se nenhum candidato
    # configurado suportar, falha com erro claro (mapeado para HTTP 422).
    if tools_requested:
        configured_tool_models = [m for m in (getattr(deps["settings"], "CANDIDATE_TOOL_MODELS_LIST", None) or []) if isinstance(m, str) and not any(m.startswith(prefix) for prefix in deps["BLOCKED_PREFIXES"])]
        if configured_tool_models:
            tool_candidates = filter_configured_model_names(configured_tool_models)
        else:
            tool_candidates = filter_tool_capable_model_names(valid_models)
        native_tools = split_tools(tools)[1]
        if native_tools and tool_candidates:
            tool_candidates = filter_native_tool_capable_model_names(tool_candidates, native_tools)
        if not tool_candidates:
            raise deps["ProviderCallError"](
                model="none",
                message=(
                    "Nenhum modelo com suporte a tools está configurado. "
                    "Defina CANDIDATE_TOOL_MODELS_LIST ou inclua um modelo com function calling."
                ),
                category="no_tool_model",
                retryable=False,
            )
        valid_models = tool_candidates

    exploration_mode = False
    exploration_info: Dict[str, Any] = {}
    # Exploração escolhe modelos fora da lista de candidatos, que podem não
    # suportar tools; desabilitada quando há tools na requisição.
    pick_exploration = deps.get("maybe_pick_openrouter_exploration") if not tools_requested else None
    if pick_exploration is not None:
        try:
            exploration_result = await pick_exploration(
                known_models=set(valid_models),
                modality=modality,
                settings=deps["settings"],
                uncertainty_score=uncertainty_score,
            )
            if exploration_result:
                chosen, exploration_info = exploration_result
                exploration_mode = True
                try:
                    current_weights = await deps["get_dynamic_strategy_weights_async"](modality)
                    top2_incumbent = await deps["asyncio"].to_thread(
                        deps["choose_top2_models"],
                        valid_models,
                        current_weights,
                        query,
                        modality,
                        uncertainty_score,
                    )
                    select_async = deps.get("select_model_async")
                    if select_async is not None:
                        incumbent = await select_async(top2_incumbent, query, modality)
                    else:
                        incumbent = await deps["asyncio"].to_thread(
                            deps["select_model"], top2_incumbent, query, modality
                        )
                    exploration_info["incumbent_model"] = incumbent
                except Exception as exc:
                    deps["logger"].debug("[router] incumbent pick for shadow failed: %s", exc)
                deps["logger"].info("[router] OpenRouter exploration: %s", chosen)
        except Exception as exc:
            deps["logger"].warning("[router] OpenRouter exploration pick failed: %s", exc)

    if not exploration_mode:
        current_weights = await deps["get_dynamic_strategy_weights_async"](modality)
        top2 = await deps["asyncio"].to_thread(
            deps["choose_top2_models"],
            valid_models,
            current_weights,
            query,
            modality,
            uncertainty_score,
        )
        select_async = deps.get("select_model_async")
        if select_async is not None:
            chosen = await select_async(top2, query, modality)
        else:
            chosen = await deps["asyncio"].to_thread(deps["select_model"], top2, query, modality)
        # Adversarial governance (roadmap #17): a high-risk knowledge cluster or
        # high epistemic uncertainty may escalate to a stronger candidate. No-op
        # unless ADVGOV_ENABLED; self-gated inside the helper.
        chosen, top2 = advgov_escalate(deps, chosen, top2, valid_models, uncertainty_score, runtime_hints)
    else:
        top2 = [chosen]

    _observe_stage("selection", selection_started_at)
    deps["logger"].info(f"[router] Model: {chosen} | UQ: {uncertainty_score:.2f} | exploration={exploration_mode}")

    if chosen.startswith("ollama/") and not deps["is_ollama_model_verified"](chosen.replace("ollama/", "")):
        deps["asyncio"].create_task(
            deps["asyncio"].to_thread(deps["_ensure_ollama_model"], chosen.replace("ollama/", ""))
        )

    # Follow-up multi-turn (messages) é a fonte da verdade: não há "query" única
    # para aumentar, então o RAG é ignorado (grounding só se aplica ao 1º turno).
    should_skip_rag = bool(messages) or _should_bypass_rag(
        deps=deps,
        query=query,
        modality=modality,
        image_b64=image_b64,
        use_rag=use_rag,
        uncertainty_score=float(uncertainty_score or 0.0),
        runtime_hints=runtime_hints,
    )

    retrieval_bundle: Dict[str, Any] = {
        "citations": [],
        "evidence_snippets": [],
        "grounded": False,
        "knowledge_version": None,
        "retrieval_mode": (runtime_hints or {}).get("retrieval_mode", "no_retrieval"),
        "retrieval_skipped_reason": "bypassed",
    }

    if not should_skip_rag:
        retrieval_started_at = time.time()
        try:
            rag_mode = rag_modality if not (image_b64 and modality != "text") else modality
            if "build_retrieval_bundle" in deps:
                retrieval_bundle = await deps["build_retrieval_bundle"](
                    query,
                    modality=rag_mode,
                    image_b64=image_b64,
                    k=int((runtime_hints or {}).get("rag_top_k", 3)),
                    retrieval_mode=(runtime_hints or {}).get("retrieval_mode"),
                    context_token_budget=(runtime_hints or {}).get("rag_context_token_budget"),
                    rerank_enabled=(runtime_hints or {}).get("rag_rerank_enabled"),
                )
                aug = retrieval_bundle.get("augmented_prompt") or query
            else:
                aug = await deps["build_augmented_prompt"](
                    query,
                    modality=rag_mode,
                    image_b64=image_b64,
                    k=int((runtime_hints or {}).get("rag_top_k", 3)),
                    retrieval_mode=(runtime_hints or {}).get("retrieval_mode"),
                    context_token_budget=(runtime_hints or {}).get("rag_context_token_budget"),
                    rerank_enabled=(runtime_hints or {}).get("rag_rerank_enabled"),
                )
            if not retrieval_bundle.get("context"):
                retrieval_bundle["grounded"] = False
            final_prompt = deps["build_final_prompt"](
                query=query,
                system_prompt=system_prompt,
                use_rag=bool(retrieval_bundle.get("context")),
                rag_text=aug if retrieval_bundle.get("context") else None,
            )
        except Exception as exc:
            deps["logger"].warning(f"[router] RAG fail: {exc}")
            final_prompt = deps["build_final_prompt"](
                query=query,
                system_prompt=system_prompt,
                use_rag=True,
                rag_text=None,
            )
            retrieval_bundle = {
                "citations": [],
                "evidence_snippets": [],
                "grounded": False,
                "knowledge_version": None,
                "retrieval_mode": (runtime_hints or {}).get("retrieval_mode", "full_retrieval"),
                "retrieval_skipped_reason": "rag_failure",
            }
        finally:
            _observe_stage("retrieval", retrieval_started_at)
    else:
        retrieval_bundle["retrieval_skipped_reason"] = "runtime_bypass"
        final_prompt = deps["build_final_prompt"](
            query=query,
            system_prompt=system_prompt,
            use_rag=False,
            rag_text=None,
        )

    use_fallback_chain = deps["_safe_setting_bool"]("REQUEST_FALLBACK_ENABLED", False)
    fallback_used = False
    fallback_models_tried = [chosen]
    fallback_errors = []

    if use_fallback_chain:
        max_fallbacks = int((runtime_hints or {}).get("max_fallbacks", deps["_safe_setting_int"]("REQUEST_MAX_FALLBACKS", 2)))
        provider_timeout_seconds = _effective_provider_timeout_seconds(runtime_hints)
        remaining_budget = _deadline_remaining_seconds(runtime_hints)
        if remaining_budget <= 1.0:
            raise deps["ProviderCallError"](
                model=chosen,
                message="Synchronous deadline exhausted before provider execution",
                category="provider_timeout",
                retryable=True,
            )
        if remaining_budget != float("inf") and provider_timeout_seconds is not None and remaining_budget <= provider_timeout_seconds + 2.0:
            max_fallbacks = 0
            try:
                deps["ROUTER_FALLBACK_SKIPPED"].labels(reason="insufficient_deadline_budget").inc()
            except Exception:
                pass

        async def _execute_provider(model_name: str):
            return await deps["call_model"](
                model=model_name,
                prompt=final_prompt,
                modality=modality,
                image_b64=image_b64,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_seconds=_effective_provider_timeout_seconds(runtime_hints),
                workload_class=(runtime_hints or {}).get("workload_class"),
                tools=tools,
                tool_choice=tool_choice,
                messages=messages,
                response_format=response_format,
                system_prompt=system_prompt,
            )

        provider_started_at = time.time()
        fallback_result = await deps["execute_with_fallback"](
            primary_model=chosen,
            execute_fn=_execute_provider,
            max_fallbacks=max_fallbacks,
        )
        _observe_stage("provider_call", provider_started_at)
        if not fallback_result.success:
            try:
                for err in fallback_result.errors:
                    deps["ROUTER_RETRY_TOTAL"].labels(
                        model=fallback_result.model_used or chosen,
                        reason=err.get("category", "provider_error"),
                    ).inc()
            except Exception:
                pass
            if fallback_result.errors:
                last = fallback_result.errors[-1]
                raise deps["ProviderCallError"](
                    model=fallback_result.model_used,
                    message=last.get("error", "All fallback models failed"),
                    category=last.get("category", "provider_unavailable"),
                    retryable=True,
                )
            raise deps["ProviderCallError"](
                model=chosen,
                message="All fallback models failed",
                category="provider_unavailable",
                retryable=True,
            )

        out, meta = fallback_result.result
        chosen = fallback_result.model_used
        fallback_used = len(fallback_result.models_tried) > 1
        fallback_models_tried = fallback_result.models_tried
        fallback_errors = fallback_result.errors
        try:
            retry_count = max(0, len(fallback_models_tried) - 1)
            for err in fallback_errors:
                deps["ROUTER_RETRY_TOTAL"].labels(
                    model=chosen,
                    reason=err.get("category", "provider_error"),
                ).inc()
            if fallback_used and len(fallback_models_tried) >= 2:
                deps["FALLBACK_USED"].labels(
                    first_model=fallback_models_tried[0],
                    second_model=fallback_models_tried[1],
                ).inc()
        except Exception:
            retry_count = max(0, len(fallback_models_tried) - 1)
    else:
        provider_started_at = time.time()
        if _deadline_remaining_seconds(runtime_hints) <= 1.0:
            raise deps["ProviderCallError"](
                model=chosen,
                message="Synchronous deadline exhausted before provider execution",
                category="provider_timeout",
                retryable=True,
            )
        out, meta = await deps["call_model"](
            model=chosen,
            prompt=final_prompt,
            modality=modality,
            image_b64=image_b64,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=_effective_provider_timeout_seconds(runtime_hints),
            workload_class=(runtime_hints or {}).get("workload_class"),
            tools=tools,
            tool_choice=tool_choice,
            messages=messages,
            response_format=response_format,
            system_prompt=system_prompt,
        )
        _observe_stage("provider_call", provider_started_at)
        retry_count = 0

    latency_s = round(time.time() - start_time, 3)

    postprocess_started_at = time.time()
    try:
        p_tok, c_tok, total_cost, load_time_s, meta_safe = deps["parse_meta_cost"](
            meta=meta,
            chosen_model=chosen,
            cost_lookup=deps["get_model_cost"],
        )
    except Exception as exc:
        deps["logger"].warning(f"[router] Metadata error: {exc}")
        p_tok, c_tok, total_cost, load_time_s, meta_safe = 0, 0, 0.0, 0.0, {}
    finally:
        _observe_stage("postprocess", postprocess_started_at)

    route_type = "fallback" if fallback_used else "direct"
    try:
        deps["ROUTER_ROUTE_COST"].labels(route_type=route_type).inc(float(total_cost))
    except Exception:
        pass
    try:
        deps["ROUTER_ATTEMPTS_PER_QUERY"].observe(max(1, retry_count + 1))
    except Exception:
        pass

    answer_text = out if isinstance(out, str) else str(out)
    tool_calls = meta.get("tool_calls") if isinstance(meta, dict) else None
    finish_reason = meta.get("finish_reason") if isinstance(meta, dict) else None
    # Um turno de tool call legitimamente tem answer vazio — não conta como resposta vazia.
    if not answer_text.strip() and not tool_calls:
        empty_reason = "thinking_only" if (meta_safe.get("reasoning") or meta.get("reasoning")) else "empty_content"
        try:
            deps["ROUTER_RESPONSE_EMPTY"].labels(model=chosen, reason=empty_reason).inc()
        except Exception:
            pass

    return {
        "answer": answer_text,
        "tool_calls": tool_calls,
        "finish_reason": finish_reason,
        "model": chosen,
        "modality": modality,
        "image_output_b64": meta_safe.get("image_output_b64"),
        "latency_s": latency_s,
        "load_time_s": load_time_s,
        "estimated_cost_usd": total_cost,
        "cost_per_1k": total_cost,
        "metadata": {
            "raw_payload": meta_safe.get("raw_payload"),
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "load_time": load_time_s,
            "uncertainty_score": uncertainty_score,
            "citations": retrieval_bundle.get("citations", []),
            "evidence_snippets": retrieval_bundle.get("evidence_snippets", []),
            "grounded": bool(retrieval_bundle.get("grounded")),
            "knowledge_version": retrieval_bundle.get("knowledge_version"),
            "retrieval_mode": retrieval_bundle.get("retrieval_mode"),
            "retrieval_skipped_reason": retrieval_bundle.get("retrieval_skipped_reason"),
            "openrouter_exploration": exploration_mode,
            "exploration_info": exploration_info,
            "stage_timings_ms": dict(stage_timings_ms),
        },
        "route": {
            "chosen_model": chosen,
            "modality_selected": modality,
            "is_multimodal_route": bool(image_b64),
            "objectives": {
                "latency": latency_s,
                "cost": total_cost,
                "uncertainty": uncertainty_score,
            },
            "pareto_front": [],
            "explanation": (
                f"OpenRouter exploration: {chosen}"
                if exploration_mode
                else f"Selected {chosen} (UQ={uncertainty_score:.2f})"
            ),
            "fallback": {
                "used": fallback_used,
                "models_tried": fallback_models_tried,
                "errors": fallback_errors,
            },
        },
        "candidates": [],
    }
