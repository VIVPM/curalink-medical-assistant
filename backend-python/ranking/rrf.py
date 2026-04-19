"""
Reciprocal Rank Fusion (RRF).

Combines multiple ranked lists into a single fused ranking using rank
positions only. Avoids the score-normalization problem (BM25 and cosine
live on different scales).

Formula per doc: sum(1 / (k + rank_i))  where k=60 (standard default).
"""

from __future__ import annotations


def rrf_fuse(
    score_lists: list[list[float]],
    k: int = 60,
) -> list[float]:
    """
    Fuse multiple score lists via Reciprocal Rank Fusion.

    Args:
        score_lists: list of score lists, each same length as the doc list.
                     Each list contains raw scores (higher = better).
                     Example: [bm25_scores, cosine_scores]
        k: RRF constant (default 60). Higher k = less weight to top ranks.

    Returns:
        list of fused RRF scores, same length as input. Higher = better.
    """
    if not score_lists or not score_lists[0]:
        return []

    n = len(score_lists[0])
    fused = [0.0] * n

    for scores in score_lists:
        # Convert scores to ranks (0-indexed, higher score = lower rank number)
        indexed = sorted(range(n), key=lambda i: scores[i], reverse=True)
        for rank, doc_idx in enumerate(indexed):
            fused[doc_idx] += 1.0 / (k + rank)

    return fused
