"""
PII redaction for observability spans and logs.

Strips emails, phone numbers, SSNs, and common PII patterns before
export to Langfuse/Grafana. Applied as a filter on span attributes.
"""

from __future__ import annotations

import re

_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_PHONE_RE = re.compile(
    r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b"
)
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def redact(text: str | None) -> str:
    """Replace PII patterns with placeholders. Returns text unchanged if None."""
    if not text:
        return text or ""
    text = _SSN_RE.sub("[SSN]", text)
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _PHONE_RE.sub("[PHONE]", text)
    return text
