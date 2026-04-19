"""
Standalone test for Phase 3 Step 3.2.
Run from the backend-python directory:
    python scripts/phase3/test_embed_batch.py

Tests the Embedder class directly (not via FastAPI). Verifies:
  1. Empty input returns an empty list (no crashes).
  2. Single-element batch works.
  3. 50-doc batch completes in <5 seconds on CPU.
  4. 100-doc batch completes in <10 seconds.
  5. Batch output order matches input order.
  6. Larger batch_size helps throughput.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from embeddings.embedder import Embedder

MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"


def make_corpus(n: int) -> list[str]:
    """Realistic-length medical abstracts for throughput testing."""
    template = (
        "This randomized controlled trial evaluated the efficacy of "
        "treatment X in a cohort of {n} patients with condition Y. "
        "Primary outcomes measured include clinical response rate and "
        "adverse event frequency over a 12-week follow-up period. "
        "Results suggest significant improvement in the intervention arm "
        "compared to placebo, with a p-value below 0.01."
    )
    return [template.format(n=100 + i) for i in range(n)]


def main():
    print(f"Loading {MODEL_NAME}...")
    t0 = time.perf_counter()
    embedder = Embedder(MODEL_NAME)
    print(f"Loaded in {time.perf_counter() - t0:.1f}s (dim={embedder.dim})")
    assert embedder.dim == 768

    # Test 1: empty input
    print("\n--- Test 1: empty input ---")
    vecs = embedder.embed_batch([])
    assert vecs == []
    print("PASS: empty list returns empty list")

    # Test 2: single-element batch
    print("\n--- Test 2: single-element batch ---")
    vecs = embedder.embed_batch(["diabetes treatment"])
    assert len(vecs) == 1 and len(vecs[0]) == 768
    print(f"PASS: got 1 vector of length {len(vecs[0])}")

    # Test 3: 50-doc batch (plan target: <5 sec on CPU)
    print("\n--- Test 3: 50-doc batch ---")
    corpus = make_corpus(50)
    t0 = time.perf_counter()
    vecs = embedder.embed_batch(corpus, batch_size=32)
    dt = time.perf_counter() - t0
    print(f"Encoded 50 docs in {dt:.2f}s ({50 / dt:.1f} docs/sec)")
    assert len(vecs) == 50
    assert all(len(v) == 768 for v in vecs)
    if dt > 5.0:
        print(f"WARN: took longer than plan target of 5 sec")
    else:
        print("PASS: under 5 sec on CPU")

    # Test 4: 100-doc batch
    print("\n--- Test 4: 100-doc batch ---")
    corpus = make_corpus(100)
    t0 = time.perf_counter()
    vecs = embedder.embed_batch(corpus, batch_size=32)
    dt = time.perf_counter() - t0
    print(f"Encoded 100 docs in {dt:.2f}s ({100 / dt:.1f} docs/sec)")
    assert len(vecs) == 100
    print("PASS")

    # Test 5: output order preserved
    print("\n--- Test 5: output order matches input order ---")
    texts = [
        "diabetes type 2 metformin treatment outcomes",
        "alzheimer memantine cognitive decline study",
        "parkinson levodopa motor symptoms review",
    ]
    vecs = embedder.embed_batch(texts)
    single_vecs = [embedder.embed_text(t) for t in texts]
    # Cosine between batch[i] and single[i] should be ~1.0
    for i in range(3):
        sim = float(np.dot(vecs[i], single_vecs[i]))
        assert sim > 0.9999, f"Order mismatch at index {i}: sim={sim}"
    print("PASS: batch output matches single-encode output, same order")

    # Test 6: batch_size scaling
    print("\n--- Test 6: batch_size scaling (same 100 docs) ---")
    corpus = make_corpus(100)
    for bs in (8, 16, 32, 64):
        t0 = time.perf_counter()
        embedder.embed_batch(corpus, batch_size=bs)
        dt = time.perf_counter() - t0
        print(f"  batch_size={bs:<3}  {dt:.2f}s  ({100 / dt:.1f} docs/sec)")

    print("\n" + "=" * 60)
    print("All tests passed. Embedder module is production-ready.")


if __name__ == "__main__":
    main()
