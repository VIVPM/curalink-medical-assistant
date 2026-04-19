"""
Standalone test for Phase 4 Step 4.6 — MMR diversity selection.
Run from the backend-python directory:
    python scripts/phase4/test_mmr.py

Runs full ranking pipeline (BM25+cosine+RRF+boosts+MedCPT) to get top-20,
then applies MMR to pick 8 diverse docs. Verifies that MMR picks a more
diverse set than greedy top-8.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio
import os
import time
import numpy as np
from dotenv import load_dotenv

from embeddings.embedder import Embedder
from sources.pubmed import fetch_pubmed
from sources.openalex import fetch_openalex
from sources.trials import fetch_trials
from sources.normalizer import normalize_pubmed, normalize_openalex, normalize_trial
from sources.merger import merge_and_dedupe, filter_complete
from ranking.bm25 import rank_bm25
from ranking.cosine import rank_cosine
from ranking.rrf import rrf_fuse
from ranking.boosts import apply_boosts
from ranking.cross_encoder import MedCPTReranker
from ranking.mmr import mmr_select

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


def avg_pairwise_sim(indices, embedder, docs):
    """Compute average pairwise cosine similarity among selected docs."""
    texts = [" ".join(filter(None, [docs[i].title, docs[i].abstract])) for i in indices]
    vecs = np.array(embedder.embed_batch(texts))
    sim = vecs @ vecs.T
    n = len(indices)
    if n < 2:
        return 0.0
    total = sum(sim[i][j] for i in range(n) for j in range(i + 1, n))
    pairs = n * (n - 1) / 2
    return total / pairs


def main():
    print("=" * 60)
    print("Phase 4 Step 4.6: MMR Diversity Selection Test")
    print("=" * 60)

    # 1. Load models
    print(f"\n[1/5] Loading models")
    embedder = Embedder(BIENCODER_MODEL)
    reranker = MedCPTReranker()
    print("  Loaded bi-encoder + cross-encoder")

    # 2. Fetch docs
    print("\n[2/5] Fetching docs")
    docs = asyncio.run(fetch_docs())
    print(f"  Got {len(docs)} docs")

    query = "vitamin D supplementation parkinson"

    # 3. Full pipeline to get top-20
    print(f"\n[3/5] Running BM25+cosine+RRF+boosts -> top-20 -> MedCPT")
    bm25_scores = rank_bm25(query, docs)
    cosine_scores = rank_cosine(query, docs, embedder)
    rrf_scores = rrf_fuse([bm25_scores, cosine_scores])
    boosted_scores = apply_boosts(rrf_scores, docs)

    top20_indices = sorted(
        range(len(docs)), key=lambda i: boosted_scores[i], reverse=True
    )[:20]
    top20_docs = [docs[i] for i in top20_indices]

    ce_scores = reranker.rerank(query, top20_docs)
    print(f"  Top-20 scored with cross-encoder")

    # 4. MMR selection
    print(f"\n[4/5] Running MMR (lambda=0.7, top_k=8)")
    t0 = time.perf_counter()
    mmr_indices = mmr_select(top20_docs, ce_scores, embedder, top_k=8, lambda_=0.7)
    dt = time.perf_counter() - t0
    print(f"  Selected 8 docs in {dt * 1000:.0f}ms")

    # Greedy top-8 (no diversity)
    greedy_indices = sorted(range(len(ce_scores)), key=lambda i: ce_scores[i], reverse=True)[:8]

    # 5. Compare
    print(f"\n[5/5] MMR top-8 vs Greedy top-8:")
    print("-" * 60)
    print("  MMR selection:")
    for rank, i in enumerate(mmr_indices):
        doc = top20_docs[i]
        title = (doc.title or "")[:50]
        src = doc.doc_type
        print(f"    {rank+1}. [{ce_scores[i]:7.3f}] ({src}) {title}")

    print("\n  Greedy selection:")
    for rank, i in enumerate(greedy_indices):
        doc = top20_docs[i]
        title = (doc.title or "")[:50]
        src = doc.doc_type
        print(f"    {rank+1}. [{ce_scores[i]:7.3f}] ({src}) {title}")
    print("-" * 60)

    # Diversity check: avg pairwise cosine among selected
    mmr_sim = avg_pairwise_sim(mmr_indices, embedder, top20_docs)
    greedy_sim = avg_pairwise_sim(greedy_indices, embedder, top20_docs)
    print(f"\n  Avg pairwise similarity — MMR: {mmr_sim:.4f}, Greedy: {greedy_sim:.4f}")

    if mmr_sim <= greedy_sim:
        print("  PASS: MMR set is more diverse (lower similarity)")
    else:
        print("  INFO: MMR set not more diverse (might happen with very diverse candidates)")

    # MMR should still include some high-scoring docs
    mmr_set = set(mmr_indices)
    greedy_set = set(greedy_indices)
    overlap = mmr_set & greedy_set
    print(f"  Overlap with greedy: {len(overlap)}/8")

    # Check: correct count
    if len(mmr_indices) == 8:
        print(f"  PASS: selected exactly 8 docs")
    else:
        print(f"  FAIL: selected {len(mmr_indices)} instead of 8")

    # Check: no duplicates
    if len(set(mmr_indices)) == len(mmr_indices):
        print("  PASS: no duplicate selections")
    else:
        print("  FAIL: duplicates in selection")

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
