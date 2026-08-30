"""
OpenTelemetry tracing + metrics, ported from ecommerce-flipkart-agent's
observability.py and adapted for curalink (HuggingFace LLM instead of google-genai).

  - LLM generation spans   -> Langfuse AND Grafana (one unified provider)
  - HTTP endpoint spans     -> Grafana only (separate provider, so Langfuse stays LLM-only)
  - chat_messages_total     -> Grafana metric (plain PromQL alerting)

Flipkart got its LLM spans free from openinference's GoogleGenAIInstrumentor. We
call the HF Inference API directly, so there is no auto-instrumentor — the LLM
span is created by hand (see llm_generation) with Langfuse v4-native attributes
(langfuse.observation.*) that Langfuse maps to a GENERATION. The Langfuse OTLP
exporter is tagged with the x-langfuse-ingestion-version: 4 header.

Every backend stays OFF unless its env vars are set (LANGFUSE_* / GRAFANA_OTLP_*),
and nothing here raises — tracing must never break a request.
"""

import base64
import logging
import os
from contextlib import contextmanager

logger = logging.getLogger(__name__)

_llm_provider = None   # unified TracerProvider for LLM spans, or None when disabled
_llm_tracer = None     # tracer from that provider (for the generation span)
_message_counter = None


def _have_langfuse() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def _have_grafana() -> bool:
    return bool(os.getenv("GRAFANA_OTLP_ENDPOINT") and os.getenv("GRAFANA_OTLP_AUTH"))


def _resource():
    from opentelemetry.sdk.resources import Resource

    return Resource.create({
        "service.name": os.getenv("OTEL_SERVICE_NAME", "curalink-research-assistant"),
        "service.namespace": "curalink",
        "deployment.environment": os.getenv("DEPLOYMENT_ENV", "development"),
    })


def _langfuse_host() -> str:
    return (os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL")
            or "https://cloud.langfuse.com").rstrip("/")


def init_observability():
    """Set up the unified LLM-span provider; export to whichever backends are configured."""
    global _llm_provider, _llm_tracer
    if not (_have_langfuse() or _have_grafana()):
        logger.info("LLM tracing disabled (no Langfuse/Grafana env).")
        return
    try:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider = TracerProvider(resource=_resource())
        enabled = []

        if _have_langfuse():
            creds = f'{os.environ["LANGFUSE_PUBLIC_KEY"]}:{os.environ["LANGFUSE_SECRET_KEY"]}'
            auth = base64.b64encode(creds.encode()).decode()
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(
                endpoint=f"{_langfuse_host()}/api/public/otel/v1/traces",
                headers={
                    "Authorization": f"Basic {auth}",
                    "x-langfuse-ingestion-version": "4",
                },
            )))
            enabled.append(f"Langfuse ({_langfuse_host()})")

        if _have_grafana():
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(
                endpoint=f"{os.environ['GRAFANA_OTLP_ENDPOINT'].rstrip('/')}/v1/traces",
                headers={"Authorization": os.environ["GRAFANA_OTLP_AUTH"]},
            )))
            enabled.append("Grafana Cloud")

        _llm_provider = provider
        _llm_tracer = provider.get_tracer("curalink.llm")
        logger.info("LLM tracing enabled via OTLP: %s", ", ".join(enabled))
    except Exception:
        logger.exception("LLM tracing init failed — continuing without it.")


@contextmanager
def llm_generation(model: str, input_text: str):
    """Wrap one HF LLM call as a Langfuse GENERATION. Yields the span, or None if off.

    Uses Langfuse v4-native OTEL attributes. This span is the root observation
    for the Langfuse trace, so the request input/output live on it directly.
    """
    if _llm_tracer is None:
        yield None
        return
    try:
        with _llm_tracer.start_as_current_span("llm-generation") as span:
            span.set_attribute("langfuse.observation.type", "generation")
            span.set_attribute("langfuse.trace.name", "llm-generation")
            span.set_attribute("langfuse.environment",
                               os.getenv("DEPLOYMENT_ENV", "development"))
            span.set_attribute("langfuse.observation.model.name", model or "")
            span.set_attribute("gen_ai.request.model", model or "")
            from pii_redactor import redact
            span.set_attribute("langfuse.observation.input", redact((input_text or "")[:8000]))
            yield span
    except Exception as e:
        logger.warning("llm_generation failed — continuing untraced: %s", e)
        yield None


def set_generation_output(span, text: str):
    """Attach the model's output to the generation span."""
    if span is None:
        return
    try:
        from pii_redactor import redact
        span.set_attribute("langfuse.observation.output", redact((text or "")[:12000]))
    except Exception as e:
        logger.debug("set_generation_output failed: %s", e)


def flush():
    """Force-send buffered spans. Render can freeze the instance and drop the last trace."""
    if _llm_provider is None:
        return
    try:
        _llm_provider.force_flush()
    except Exception as e:
        logger.debug("flush failed: %s", e)


def init_http_tracing(app):
    """Trace every HTTP endpoint to Grafana, on its own provider so Langfuse stays LLM-only."""
    if not _have_grafana():
        logger.info("Grafana HTTP tracing disabled (GRAFANA_OTLP_* not set).")
        return
    try:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        provider = TracerProvider(resource=_resource())
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(
            endpoint=f"{os.environ['GRAFANA_OTLP_ENDPOINT'].rstrip('/')}/v1/traces",
            headers={"Authorization": os.environ["GRAFANA_OTLP_AUTH"]},
        )))
        FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
        logger.info("Grafana HTTP tracing enabled via OTLP.")
    except Exception:
        logger.exception("Grafana HTTP tracing init failed — continuing without it.")


def init_metrics():
    """Export a chat_messages_total counter. A metric, not traces, so alerts are plain PromQL."""
    global _message_counter
    if not _have_grafana():
        return
    try:
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

        reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(
                endpoint=f"{os.environ['GRAFANA_OTLP_ENDPOINT'].rstrip('/')}/v1/metrics",
                headers={"Authorization": os.environ["GRAFANA_OTLP_AUTH"]},
            ),
            export_interval_millis=15000,
        )
        provider = MeterProvider(resource=_resource(), metric_readers=[reader])
        _message_counter = provider.get_meter("curalink.chat").create_counter(
            "chat_messages_total",
            description="Chat pipeline runs handled, by status (ok/error/cache)",
        )
        logger.info("Grafana metrics enabled via OTLP.")
    except Exception:
        logger.exception("Grafana metrics init failed — continuing without it.")


def record_message(status: str):
    """Increment the message counter (status = ok | error | cache)."""
    if _message_counter is None:
        return
    try:
        _message_counter.add(1, {"status": status})
    except Exception as e:
        logger.debug("record_message failed: %s", e)
