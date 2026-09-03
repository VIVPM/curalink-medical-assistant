"""
Egress allowlist — restricts outbound HTTP to known API hosts only.

Prevents SSRF if a future feature accepts user-supplied URLs. Applied as a
validator before any httpx request in the retrieval pipeline.

To add a new host: append to ALLOWED_HOSTS below.
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

ALLOWED_HOSTS: frozenset[str] = frozenset({
    # Medical data sources
    "eutils.ncbi.nlm.nih.gov",
    "efetch.ncbi.nlm.nih.gov",
    "api.openalex.org",
    "clinicaltrials.gov",
    "www.clinicaltrials.gov",
    # LLM providers
    "api-inference.huggingface.co",
    "api.cloudflare.com",
    # Geocoding
    "nominatim.openstreetmap.org",
    # Observability (outbound export)
    "us.cloud.langfuse.com",
    "cloud.langfuse.com",
    "otlp-gateway-prod-ap-south-1.grafana.net",
})


class EgressBlocked(Exception):
    """Raised when a request targets a host not in the allowlist."""


def check_url(url: str) -> None:
    """Raise EgressBlocked if the URL's host is not in ALLOWED_HOSTS."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host not in ALLOWED_HOSTS:
        logger.warning("[egress] blocked request to %s", host)
        raise EgressBlocked(f"Outbound HTTP to {host} is not allowed")


def is_allowed(url: str) -> bool:
    """Return True if the URL's host is in the allowlist."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    return host in ALLOWED_HOSTS


async def httpx_event_hook(request):
    """httpx event hook — validates every outbound request against the allowlist.

    Usage: httpx.AsyncClient(event_hooks={"request": [httpx_event_hook]})
    """
    check_url(str(request.url))
