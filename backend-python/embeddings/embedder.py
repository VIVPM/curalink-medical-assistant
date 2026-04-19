"""
Embedder — HF Inference API via huggingface_hub InferenceClient.

Calls HF's hosted feature-extraction pipeline. No local model weights,
no PyTorch, no sentence-transformers. Deploys cleanly on Render free tier.
"""

from __future__ import annotations

import os
import time
import numpy as np
from huggingface_hub import InferenceClient

HF_TOKEN = os.getenv("HF_TOKEN", "")

MAX_RETRIES = 3
RETRY_BACKOFF = 2.0  # seconds; doubles each retry


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._dim = 768  # PubMedBERT-MS-MARCO dimension
        self.client = InferenceClient(model=model_name, token=HF_TOKEN)

    @property
    def dim(self) -> int:
        return self._dim

    def _normalize(self, vec: list[float]) -> list[float]:
        """L2-normalize a vector for cosine similarity."""
        arr = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()

    def _pool(self, result) -> list[float]:
        """Mean-pool token-level embeddings to sentence embedding."""
        arr = np.array(result, dtype=np.float32)
        if arr.ndim == 2:
            # Token-level: (seq_len, hidden_dim) -> mean pool
            return arr.mean(axis=0).tolist()
        return arr.tolist()

    def _call_with_retry(self, text):
        """Call HF feature_extraction with retry + exponential backoff."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return self.client.feature_extraction(text)
            except Exception as e:
                if attempt == MAX_RETRIES:
                    raise
                wait = RETRY_BACKOFF * (2 ** (attempt - 1))
                print(f"[embedder] attempt {attempt}/{MAX_RETRIES} failed: {e}. "
                      f"Retrying in {wait:.1f}s...")
                time.sleep(wait)

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text. Returns an L2-normalized vector."""
        result = self._call_with_retry(text)
        vec = self._pool(result)
        return self._normalize(vec)

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[list[float]]:
        """
        Embed a list of texts in batches. Returns list of L2-normalized vectors.
        Sends each batch as a single HF API call (list of texts).
        Failed batches fall back to per-text calls.
        """
        if not texts:
            return []

        zero_vec = [0.0] * self._dim
        all_vecs: list[list[float]] = []
        failed = 0

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                # Send entire batch as one API call
                results = self._call_with_retry(batch)

                # results should be list of embeddings
                arr = np.array(results, dtype=np.float32)
                if arr.ndim == 3:
                    # (batch, seq_len, hidden_dim) -> mean pool each
                    for j in range(arr.shape[0]):
                        vec = arr[j].mean(axis=0).tolist()
                        all_vecs.append(self._normalize(vec))
                elif arr.ndim == 2:
                    # (batch, hidden_dim) - already pooled
                    for j in range(arr.shape[0]):
                        all_vecs.append(self._normalize(arr[j].tolist()))
                else:
                    # Single text came back, shouldn't happen in batch
                    vec = self._pool(results)
                    all_vecs.append(self._normalize(vec))
            except Exception as e:
                print(f"[embedder] batch call failed, falling back to per-text: {e}")
                # Fallback: embed one by one
                for text in batch:
                    try:
                        result = self._call_with_retry(text)
                        vec = self._pool(result)
                        all_vecs.append(self._normalize(vec))
                    except Exception as e2:
                        print(f"[embedder] WARN: skipping text: {e2}")
                        all_vecs.append(zero_vec)
                        failed += 1

        if failed:
            print(f"[embedder] {failed}/{len(texts)} texts failed, "
                  f"replaced with zero vectors")
        return all_vecs
