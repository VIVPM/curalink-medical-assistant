"""
Maximal Marginal Relevance (MMR) diversity selection.

After cross-encoder scores the top-20, MMR picks the final top 6-8
by balancing relevance and diversity. Prevents the final set from being
8 near-duplicate review papers on a popular topic.

Formula: mmr = lambda * relevance - (1 - lambda) * max_cosine_to_picked
lambda=0.7 (70% relevance, 30% diversity)
"""

from __future__ import annotations

import numpy as np

from schemas.document import Document
from embeddings.embedder import Embedder


def _doc_text(doc: Document) -> str:
    parts: list[str] = []
    if doc.title:
        parts.append(doc.title)
    if doc.abstract:
        parts.append(doc.abstract)
    if doc.full_text and doc.full_text != doc.abstract:
        parts.append(doc.full_text)
    return " ".join(parts) if parts else ""


def mmr_select(
    docs: list[Document],
    scores: list[float],
    embedder: Embedder,
    top_k: int = 8,
    lambda_: float = 0.7,
    precomputed_vecs: np.ndarray | None = None,
) -> list[int]:
    """
    Select top_k docs via MMR from a scored candidate list.

    Args:
        docs: candidate Documents (e.g. top-20 from cross-encoder)
        scores: relevance scores for each doc (same order)
        embedder: loaded Embedder for computing pairwise similarity
        top_k: how many to select (default 8)
        lambda_: relevance vs diversity trade-off (default 0.7)
        precomputed_vecs: if provided, skip embedding (reuse from cosine step)

    Returns:
        list of indices into docs, in MMR selection order.
    """
    if not docs:
        return []
    if len(docs) <= top_k:
        return list(range(len(docs)))

    # Normalize scores to [0, 1] for fair combination with cosine
    max_s = max(scores)
    min_s = min(scores)
    if max_s == min_s:
        norm_scores = [1.0] * len(scores)
    else:
        norm_scores = [(s - min_s) / (max_s - min_s) for s in scores]

    # Use precomputed embeddings if available, otherwise embed
    if precomputed_vecs is not None:
        vecs = precomputed_vecs
    else:
        texts = [_doc_text(d) for d in docs]
        vecs = np.array(embedder.embed_batch(texts))

    # Pairwise cosine similarity matrix (embeddings already normalized)
    sim_matrix = vecs @ vecs.T

    selected: list[int] = []
    remaining = set(range(len(docs)))

    for _ in range(top_k):
        best_idx = -1
        best_mmr = -float("inf")

        for idx in remaining:
            relevance = norm_scores[idx]

            if selected:
                max_sim = max(sim_matrix[idx][j] for j in selected)
            else:
                max_sim = 0.0

            mmr_score = lambda_ * relevance - (1.0 - lambda_) * max_sim
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx

        selected.append(best_idx)
        remaining.discard(best_idx)

    return selected
