# Objective: Optional OpenTelemetry bootstrap for distributed tracing in production.
"""Initialize OTLP tracing when OTEL_EXPORTER_OTLP_ENDPOINT is configured."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_initialized = False


def setup_opentelemetry(app=None) -> None:
    """Configure OpenTelemetry tracing for FastAPI when exporter endpoint is set."""
    global _initialized
    if _initialized:
        return

    endpoint = (os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or "").strip()
    if not endpoint:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.info("[otel] OpenTelemetry packages not installed; tracing disabled")
        return

    service_name = os.getenv("OTEL_SERVICE_NAME", "iag-router-api")
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    if app is not None:
        try:
            FastAPIInstrumentor.instrument_app(app)
        except Exception as exc:
            logger.warning("[otel] FastAPI instrumentation skipped: %s", exc)

    _initialized = True
    logger.info("[otel] OpenTelemetry tracing enabled -> %s", endpoint)
