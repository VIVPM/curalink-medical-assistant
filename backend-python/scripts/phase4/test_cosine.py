"""
Standalone test for Phase 4 Step 4.2 — Cosine similarity scoring.
Run from the backend-python directory:
    python scripts/phase4/test_cosine.py

Fetches ~50 docs, scores with cosine similarity via PubMedBERT, and verifies
semantic matches rank high even when exact keywords are absent.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio
import os
import time
from dotenv import load_dotenv

from embeddings.embedder import Embedder
from sources.pubmed import fetch_pubmed
from sources.openalex import fetch_openalex
from sources.trials import fetch_trials
from sources.normalizer import normalize_pubmed, normalize_openalex, normalize_trial
from sources.merger import merge_and_dedupe, filter_complete
from ranking.cosine import rank_cosine

load_dotenv()

BIENCODER_MODEL = os.getenv("BIENCODER_MODEL", "pritamdeka/S-PubMedBert-MS-MARCO")


async def fetch_docs():
    query = "parkinson disease treatment"
    disease = "parkinson"

    pubmed_raw, openalex_raw, trials_raw = await asyncio.gather(
        fetch_pubmed(query, limit=25),
        fetch_openalex(query, limit=25),
        fetch_trials(disease=disease, limit=10),
    )

    pubmed_docs = [normalize_pubmed(r) for r in pubmed_raw]
    openalex_docs = [normalize_openalex(r) for r in openalex_raw]
    trial_docs = [normalize_trial(r, disease_context=disease) for r in trials_raw]

    deduped = merge_and_dedupe([pubmed_docs, openalex_docs, trial_docs])
    return filter_complete(deduped)


def main():
    print("=" * 60)
    print("Phase 4 Step 4.2: Cosine Similarity Scoring Test")
    print("=" * 60)

    # 1. Load embedder
    print(f"\n[1/4] Loading embedder: {BIENCODER_MODEL}")
    t0 = time.perf_counter()
    embedder = Embedder(BIENCODER_MODEL)
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s (dim={embedder.dim})")

    # 2. Fetch docs
    print("\n[2/4] Fetching docs (25 pubmed + 25 openalex + 10 trials)")
    t0 = time.perf_counter()
    docs = asyncio.run(fetch_docs())
    print(f"  Got {len(docs)} complete docs in {time.perf_counter() - t0:.1f}s")

    if len(docs) < 10:
        print("  FAIL: too few docs to test ranking")
        return

    # 3. Score with a PARAPHRASED query (no exact keyword overlap)
    # BM25 would struggle here; cosine should catch the semantic match
    query = "supplementing with Vit D for neurological movement disorder"
    print(f"\n[3/4] Running cosine scoring with paraphrased query:")
    print(f"  \"{query}\"")
    t0 = time.perf_counter()
    scores = rank_cosine(query, docs, embedder)
    dt = time.perf_counter() - t0
    print(f"  Scored {len(scores)} docs in {dt * 1000:.0f}ms")

    # Pair and sort
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

    # 4. Verify results
    print(f"\n[4/4] Top 10 results:")
    print("-" * 60)
    for i, (score, doc) in enumerate(ranked[:10]):
        title = (doc.title or "")[:65]
        print(f"  {i+1:2d}. [{score:.4f}] {title}")

    print("-" * 60)

    # Check score range
    max_score = ranked[0][0]
    min_score = ranked[-1][0]
    print(f"\n  Score range: {min_score:.4f} to {max_score:.4f}")

    # Top 5 avg vs bottom 5 avg
    top5_avg = sum(s for s, _ in ranked[:5]) / 5
    bot5_avg = sum(s for s, _ in ranked[-5:]) / 5
    print(f"  Top 5 avg: {top5_avg:.4f}")
    print(f"  Bottom 5 avg: {bot5_avg:.4f}")

    if top5_avg > bot5_avg:
        print("  PASS: top docs score higher than bottom docs")
    else:
        print("  FAIL: scoring order looks wrong")

    # Scores should be in [0, 1] range (normalized embeddings)
    all_in_range = all(0.0 <= s <= 1.01 for s in scores)
    if all_in_range:
        print("  PASS: all scores in [0, 1] range (normalized)")
    else:
        print("  WARN: some scores outside [0, 1]")

    if len(scores) == len(docs):
        print(f"  PASS: scores length ({len(scores)}) matches docs ({len(docs)})")
    else:
        print(f"  FAIL: length mismatch scores={len(scores)} docs={len(docs)}")

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
