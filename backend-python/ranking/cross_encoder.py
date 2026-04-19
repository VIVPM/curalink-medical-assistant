"""
MedCPT cross-encoder reranker via HF Inference API (huggingface_hub).

Uses text_classification endpoint for sequence-pair scoring.
No local model weights, no PyTorch.
"""

from __future__ import annotations

import os
from huggingface_hub import InferenceClient

from schemas.document import Document

DEFAULT_MODEL = "ncbi/MedCPT-Cross-Encoder"
HF_TOKEN = os.getenv("HF_TOKEN", "")


class MedCPTReranker:
    """Calls MedCPT cross-encoder via HF Inference API."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.client = InferenceClient(model=model_name, token=HF_TOKEN)

    def _doc_text(self, doc: Document) -> str:
        parts: list[str] = []
        if doc.title:
            parts.append(doc.title)
        if doc.abstract:
            parts.append(doc.abstract)
        text = " ".join(parts)
        # MedCPT max is 512 tokens; ~4 chars/token, reserve ~60 tokens for query+SEP
        return text[:1800]

    def rerank(
        self, query: str, docs: list[Document], batch_size: int = 10
    ) -> list[float]:
        """
        Score each doc against the query with the cross-encoder.
        Sends docs in batches to reduce API calls.
        Returns list of float scores, same length/order as docs.
        """
        if not docs:
            return []

        all_scores: list[float] = []

        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            # Try batch call first
            try:
                pairs = [f"{query} [SEP] {self._doc_text(doc)}" for doc in batch]
                # HF text_classification can accept a list of strings
                results = self.client.text_classification(pairs)

                # Results could be list of lists or list of dicts
                if results and isinstance(results[0], list):
                    # Each item is a list of classifications
                    for result in results:
                        if result and len(result) > 0:
                            all_scores.append(result[0].score)
                        else:
                            all_scores.append(0.5)
                elif results and hasattr(results[0], 'score'):
                    # Flat list of classifications (one per input)
                    for result in results:
                        all_scores.append(result.score)
                else:
                    # Unknown format, fallback scores
                    all_scores.extend([0.5] * len(batch))

            except Exception as e:
                print(f"[cross-encoder] batch failed, falling back to per-doc: {e}")
                # Fallback: score one by one
                for doc in batch:
                    try:
                        pair_text = f"{query} [SEP] {self._doc_text(doc)}"
                        result = self.client.text_classification(pair_text)
                        if result and len(result) > 0:
                            all_scores.append(result[0].score)
                        else:
                            all_scores.append(0.5)
                    except Exception as e2:
                        print(f"[cross-encoder] HF API error: {e2}")
                        all_scores.append(0.5)

        return all_scores
