"""
BM25 keyword scoring over Document abstracts/text.

Takes a query string and a list of Documents, returns a list of BM25 scores
in the same order as the input docs. Higher score = more keyword-relevant.

Uses rank_bm25 (Okapi BM25) which tokenizes on whitespace by default.
We lowercase and do minimal cleanup to improve matching.
"""

from __future__ import annotations

import re
from rank_bm25 import BM25Okapi

from schemas.document import Document


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def _doc_text(doc: Document) -> str:
    """Extract the searchable text from a Document."""
    parts: list[str] = []
    if doc.title:
        parts.append(doc.title)
    if doc.abstract:
        parts.append(doc.abstract)
    if doc.full_text and doc.full_text != doc.abstract:
        parts.append(doc.full_text)
    return " ".join(parts)


def rank_bm25(query: str, docs: list[Document]) -> list[float]:
    """
    Score each document against the query using BM25.

    Args:
        query: free-text search query
        docs: list of Documents to score

    Returns:
        list of float scores, same length and order as docs.
        Higher = more relevant. Zero if doc has no text.
    """
    if not docs:
        return []

    corpus = [_tokenize(_doc_text(d)) for d in docs]
    query_tokens = _tokenize(query)

    # Handle edge case: all docs empty
    if all(len(tokens) == 0 for tokens in corpus):
        return [0.0] * len(docs)

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query_tokens)
    return scores.tolist()
