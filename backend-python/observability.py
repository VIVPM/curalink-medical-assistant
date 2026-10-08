"""Content-free LLM generation tracing exported to Langfuse over OTLP."""

import base64
import logging
import os
from contextlib import contextmanager

logger = logging.getLogger(__name__)

_llm_provider = None
_llm_tracer = None


def _have_langfuse() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


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
    """Set up the LLM-span provider when Langfuse is configured."""
    global _llm_provider, _llm_tracer
    if not _have_langfuse():
        logger.info("LLM tracing disabled (no Langfuse env).")
        return
    try:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider = TracerProvider(resource=_resource())
        creds = f'{os.environ["LANGFUSE_PUBLIC_KEY"]}:{os.environ["LANGFUSE_SECRET_KEY"]}'
        auth = base64.b64encode(creds.encode()).decode()
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(
            endpoint=f"{_langfuse_host()}/api/public/otel/v1/traces",
            headers={
                "Authorization": f"Basic {auth}",
                "x-langfuse-ingestion-version": "4",
            },
        )))
        _llm_provider = provider
        _llm_tracer = provider.get_tracer("curalink.llm")
        logger.info("LLM tracing enabled via OTLP: Langfuse (%s)", _langfuse_host())
    except Exception:
        logger.exception("LLM tracing init failed — continuing without it.")


@contextmanager
def llm_generation(model: str, _input_text: str):
    """Wrap an LLM call as a generation span without recording prompt content."""
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
            yield span
    except Exception as e:
        logger.warning("llm_generation failed — continuing untraced: %s", e)
        yield None


def set_generation_output(_span, _text: str):
    """Intentionally omit generated content from third-party telemetry."""
    return


def flush():
    """Force-send buffered spans. Render can freeze the instance and drop the last trace."""
    if _llm_provider is None:
        return
    try:
        _llm_provider.force_flush()
    except Exception as e:
        logger.debug("flush failed: %s", e)
