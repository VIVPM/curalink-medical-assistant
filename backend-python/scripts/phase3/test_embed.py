"""
Standalone test for Phase 3 Step 3.1.
Run from the backend-python directory:
    python scripts/phase3/test_embed.py

Loads PubMedBERT-MS-MARCO directly (bypassing FastAPI) and confirms:
  1. Model downloads/loads successfully.
  2. Output dimension is 768.
  3. Embeddings are L2-normalized (cosine similarity works directly).
  4. Similar medical concepts score higher than unrelated ones.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"


def main():
    print(f"Loading {MODEL_NAME}...")
    t0 = time.perf_counter()
    model = SentenceTransformer(MODEL_NAME)
    print(f"Loaded in {time.perf_counter() - t0:.1f}s")

    # Test 1: single embedding shape + normalization
    text = "Vitamin D supplementation in Parkinson's disease patients"
    vec = model.encode(text, normalize_embeddings=True, convert_to_numpy=True)
    print(f"\n--- Test 1: single embedding ---")
    print(f"Text:   {text!r}")
    print(f"Shape:  {vec.shape}")
    print(f"Norm:   {np.linalg.norm(vec):.6f}  (should be ~1.0 if normalized)")
    print(f"Sample: {vec[:5].tolist()}")
    assert vec.shape == (768,), f"Expected (768,), got {vec.shape}"
    assert abs(np.linalg.norm(vec) - 1.0) < 0.01, "Not normalized"
    print("PASS: shape 768, L2-normalized")

    # Test 2: semantic similarity sanity check
    print(f"\n--- Test 2: semantic similarity ---")
    query = "vitamin D therapy for parkinson"
    candidates = [
        ("Vitamin D supplementation improves motor symptoms in PD", "related"),
        ("Cholecalciferol levels correlate with Parkinson severity", "related"),
        ("Dopamine replacement with levodopa in parkinsonism", "somewhat"),
        ("Weather patterns in Antarctica during winter", "unrelated"),
        ("Financial markets and cryptocurrency trends", "unrelated"),
    ]

    q_vec = model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
    cand_texts = [c[0] for c in candidates]
    c_vecs = model.encode(cand_texts, normalize_embeddings=True, convert_to_numpy=True)

    # Cosine = dot product for normalized vectors
    scores = (c_vecs @ q_vec).tolist()

    print(f"Query: {query!r}\n")
    ranked = sorted(zip(scores, candidates), reverse=True)
    for score, (text, label) in ranked:
        print(f"  {score:.4f}  [{label:<10}] {text}")

    # Sanity: top hit should be a "related" one
    top_label = ranked[0][1][1]
    assert top_label == "related", f"Top result was {top_label!r}, expected 'related'"
    print("\nPASS: related docs scored higher than unrelated")

    # Test 3: batch encoding speed
    print(f"\n--- Test 3: batch encoding speed ---")
    batch_size = 50
    batch = [f"Medical abstract number {i} about diabetes." for i in range(batch_size)]
    t0 = time.perf_counter()
    batch_vecs = model.encode(
        batch, normalize_embeddings=True, convert_to_numpy=True, batch_size=32
    )
    dt = time.perf_counter() - t0
    print(f"Encoded {batch_size} docs in {dt:.2f}s "
          f"({batch_size / dt:.1f} docs/sec)")
    assert batch_vecs.shape == (batch_size, 768)
    print("PASS: batch encoding works")

    print("\n" + "=" * 60)
    print("All tests passed. Embedding dim is 768.")


if __name__ == "__main__":
    main()
