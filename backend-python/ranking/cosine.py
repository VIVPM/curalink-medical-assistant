"""
Dense cosine similarity scoring using PubMedBERT embeddings.

Takes a query and list of Documents, embeds everything with the bi-encoder,
and returns cosine similarity scores. Catches semantic matches that BM25
misses (e.g. "supplementing with Vit D" matches "cholecalciferol administration").
"""

from __future__ import annotations

import numpy as np

from schemas.document import Document
from embeddings.embedder import Embedder


def _doc_text(doc: Document) -> str:
    """Extract the text to embed from a Document."""
    parts: list[str] = []
    if doc.title:
        parts.append(doc.title)
    if doc.abstract:
        parts.append(doc.abstract)
    if doc.full_text and doc.full_text != doc.abstract:
        parts.append(doc.full_text)
    return " ".join(parts) if parts else ""


def rank_cosine(
    query: str, docs: list[Document], embedder: Embedder,
    return_vecs: bool = False,
) -> list[float] | tuple[list[float], np.ndarray]:
    """
    Score each document against the query using cosine similarity
    via PubMedBERT embeddings.

    Args:
        query: free-text search query
        docs: list of Documents to score
        embedder: loaded Embedder instance (PubMedBERT-MS-MARCO)
        return_vecs: if True, also return the doc embedding matrix

    Returns:
        list of float scores in [0, 1] (embeddings are normalized),
        same length and order as docs. Higher = more similar.
        If return_vecs=True, returns (scores, doc_vecs_ndarray).
    """
    if not docs:
        return ([], np.array([])) if return_vecs else []

    texts = [_doc_text(d) for d in docs]

    # Embed query + all docs in one batch for efficiency
    all_texts = [query] + texts
    all_vecs = embedder.embed_batch(all_texts)

    query_vec = np.array(all_vecs[0])
    doc_vecs = np.array(all_vecs[1:])

    # Cosine similarity (vectors are already L2-normalized by Embedder)
    scores = doc_vecs @ query_vec

    if return_vecs:
        return scores.tolist(), doc_vecs
    return scores.tolist()
