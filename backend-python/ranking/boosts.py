"""
Recency and source-credibility multiplicative boosts on RRF scores.

Formula: final = rrf_score * (1 + 0.15 * recency_boost) * (1 + 0.10 * credibility_boost)

recency_boost: 1.0 for current year, linearly decays to 0.0 at 10+ years old.
credibility_boost: 1.0 if multi-source, 0.7 trial, 0.5 PubMed-only, 0.3 OpenAlex-only.
"""

from __future__ import annotations

from datetime import datetime

from schemas.document import Document

CURRENT_YEAR = datetime.now().year


def _recency_boost(year: int | None) -> float:
    """1.0 for current year, linear decay to 0.0 at 10+ years old."""
    if year is None:
        return 0.0
    age = CURRENT_YEAR - year
    if age <= 0:
        return 1.0
    if age >= 10:
        return 0.0
    return 1.0 - (age / 10.0)


def _credibility_boost(doc: Document) -> float:
    """
    1.0 multi-source, 0.7 trial, 0.5 PubMed-only, 0.3 OpenAlex-only.
    """
    if len(doc.sources) > 1:
        return 1.0
    if doc.doc_type == "trial":
        return 0.7
    source = doc.sources[0] if doc.sources else ""
    if source == "pubmed":
        return 0.5
    if source == "openalex":
        return 0.3
    return 0.3


def apply_boosts(
    rrf_scores: list[float],
    docs: list[Document],
    recency_weight: float = 0.15,
    credibility_weight: float = 0.10,
) -> list[float]:
    """
    Apply recency and credibility multiplicative boosts to RRF scores.

    Args:
        rrf_scores: fused RRF scores from rrf_fuse()
        docs: corresponding Documents (same order)
        recency_weight: weight for recency boost (default 0.15)
        credibility_weight: weight for credibility boost (default 0.10)

    Returns:
        list of boosted scores, same length and order.
    """
    boosted = []
    for score, doc in zip(rrf_scores, docs):
        r = _recency_boost(doc.year)
        c = _credibility_boost(doc)
        final = score * (1.0 + recency_weight * r) * (1.0 + credibility_weight * c)
        boosted.append(final)
    return boosted
