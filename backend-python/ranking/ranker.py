"""
Unified ranking pipeline — Stage 4 of the orchestrator.

Takes ~150 deduplicated Documents + a query and returns the final top 14
diverse, well-ranked docs (8 publications + 6 trials) ready for the
Context Builder (Stage 5).

Three-stage funnel:
  1. Hybrid scoring: BM25 + cosine -> RRF fusion + recency/credibility boosts -> shortlist
  2. Precision pass: MedCPT cross-encoder -> reranked shortlist
  3. Diversity pass: MMR -> final top_k (with min 6 trial guarantee)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


from schemas.document import Document
from embeddings.embedder import Embedder
from ranking.bm25 import rank_bm25
from ranking.cosine import rank_cosine
from ranking.rrf import rrf_fuse
from ranking.boosts import apply_boosts
from ranking.cross_encoder import MedCPTReranker


@dataclass
class RankingResult:
    """Output of the ranking pipeline."""
    top_docs: list[Document]
    timings_ms: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)


def run_ranking(
    query: str,
    docs: list[Document],
    embedder: Embedder,
    reranker: MedCPTReranker,
    top_k: int = 14,
    rrf_k: int = 60,
    mmr_lambda: float = 0.7,
    min_trials: int = 6,
    min_pubs: int = 8,
) -> RankingResult:
    """
    Full Stage 4 ranking pipeline.

    Args:
        query: expanded search query
        docs: deduplicated Documents from Stage 2+3
        embedder: loaded PubMedBERT bi-encoder
        reranker: loaded MedCPT cross-encoder
        top_k: final number of docs to return (default 14)
        rrf_k: RRF constant (default 60)
        mmr_lambda: MMR relevance/diversity trade-off (default 0.7)
        min_trials: minimum trial docs in final output (default 6)
        min_pubs: minimum publication docs in final output (default 8)

    Returns:
        RankingResult with top_docs, timings, and counts.
    """
    timings: dict = {}
    input_count = len(docs)

    if not docs:
        return RankingResult(top_docs=[], timings_ms={}, counts={"input": 0})


    t0 = time.perf_counter()
    bm25_scores_all = rank_bm25(query, docs)
    timings["bm25_ms"] = round((time.perf_counter() - t0) * 1000)


    pub_indices = [i for i in range(len(docs)) if docs[i].doc_type == "publication"]
    trial_indices = [i for i in range(len(docs)) if docs[i].doc_type == "trial"]


    pub_sorted = sorted(pub_indices, key=lambda i: bm25_scores_all[i], reverse=True)[:13]
    trial_sorted = sorted(trial_indices, key=lambda i: bm25_scores_all[i], reverse=True)[:10]
    bm25_top_indices = pub_sorted + trial_sorted

    shortlist = [docs[i] for i in bm25_top_indices]
    bm25_scores = [bm25_scores_all[i] for i in bm25_top_indices]


    t0 = time.perf_counter()
    cosine_scores, shortlist_vecs = rank_cosine(query, shortlist, embedder, return_vecs=True)
    timings["cosine_ms"] = round((time.perf_counter() - t0) * 1000)


    t0 = time.perf_counter()
    rrf_scores = rrf_fuse([bm25_scores, cosine_scores], k=rrf_k)
    boosted_scores = apply_boosts(rrf_scores, shortlist)
    timings["rrf_boosts_ms"] = round((time.perf_counter() - t0) * 1000)


    t0 = time.perf_counter()
    ce_scores = reranker.rerank(query, shortlist)
    timings["cross_encoder_ms"] = round((time.perf_counter() - t0) * 1000)


    ce_min = min(ce_scores) if ce_scores else 0
    ce_max = max(ce_scores) if ce_scores else 1
    ce_range = ce_max - ce_min if ce_max != ce_min else 1
    ce_norm = [(s - ce_min) / ce_range for s in ce_scores]

    b_min = min(boosted_scores) if boosted_scores else 0
    b_max = max(boosted_scores) if boosted_scores else 1
    b_range = b_max - b_min if b_max != b_min else 1
    b_norm = [(s - b_min) / b_range for s in boosted_scores]

    combined_scores = [0.4 * b + 0.6 * c for b, c in zip(b_norm, ce_norm)]


    t0 = time.perf_counter()


    sl_pub_indices = [i for i, d in enumerate(shortlist) if d.doc_type == "publication"]
    sl_trial_indices = [i for i, d in enumerate(shortlist) if d.doc_type == "trial"]


    sl_pub_sorted = sorted(sl_pub_indices, key=lambda i: combined_scores[i], reverse=True)
    sl_trial_sorted = sorted(sl_trial_indices, key=lambda i: combined_scores[i], reverse=True)


    picked_pub = sl_pub_sorted[:min_pubs]
    picked_trial = sl_trial_sorted[:min_trials]


    picked_indices = list(dict.fromkeys(picked_pub + picked_trial))


    if len(picked_indices) < top_k:
        remaining = [i for i in range(len(shortlist)) if i not in picked_indices]
        remaining_sorted = sorted(remaining, key=lambda i: combined_scores[i], reverse=True)
        for idx in remaining_sorted:
            if len(picked_indices) >= top_k:
                break
            picked_indices.append(idx)

    timings["selection_ms"] = round((time.perf_counter() - t0) * 1000)

    final_docs = [shortlist[i] for i in picked_indices]

    timings["total_ms"] = sum(timings.values())


    final_source_counts = {"pubmed": 0, "openalex": 0, "trial": 0, "multi": 0}
    for doc in final_docs:
        if doc.doc_type == "trial":
            final_source_counts["trial"] += 1
        elif len(doc.sources) > 1:
            final_source_counts["multi"] += 1
        elif "openalex" in doc.sources:
            final_source_counts["openalex"] += 1
        else:
            final_source_counts["pubmed"] += 1

    return RankingResult(
        top_docs=final_docs,
        timings_ms=timings,
        counts={
            "input": input_count,
            "bm25_shortlist": len(shortlist),
            "after_selection": len(final_docs),
            "final_sources": final_source_counts,
        },
    )
