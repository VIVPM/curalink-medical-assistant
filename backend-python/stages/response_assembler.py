"""
Stage 7 — Response Assembler.

Takes raw LLM output (with [docN] anchors) + doc_anchors dict from Stage 5,
produces the final user-facing JSON with resolved citations, snippets,
and pipeline metadata.

Pure translation layer — no re-ranking, no generation, no fetching.
~15ms total.

Five jobs:
  1. Citation anchor resolution ([docN] -> real metadata)
  2. Supporting snippet extraction (best sentences from abstract)
  3. Final schema assembly (matches brief's Structured Output)
  4. Hallucination flags (unverified citations)
  5. Graceful abstain formatting
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from schemas.document import Document


@dataclass
class AssembledResponse:
    user_facing_json: dict
    pipeline_meta: dict
    warnings: list[str] = field(default_factory=list)


def _extract_snippet(abstract: str, finding: str, max_sentences: int = 2) -> str:
    """
    Pull 1-2 sentences from abstract that best support the finding.
    Simple word-overlap ranking. Fallback: first 2 sentences.
    """
    if not abstract:
        return ""

    sentences = re.split(r'(?<=[.!?])\s+', abstract.strip())
    if not sentences:
        return ""

    if not finding:
        return " ".join(sentences[:max_sentences])

    finding_words = set(finding.lower().split())

    scored = []
    for sent in sentences:
        sent_words = set(sent.lower().split())
        overlap = len(finding_words & sent_words)
        scored.append((overlap, sent))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [s for _, s in scored[:max_sentences]]
    return " ".join(top)


def _resolve_source(anchor: str, doc: Document, finding: str = "") -> dict:
    """Convert a Document into source_details dict with snippet."""
    return {
        "title": doc.title or "",
        "authors": doc.authors[:5] if doc.authors else [],
        "year": doc.year,
        "platform": ", ".join(doc.sources) if doc.sources else "unknown",
        "url": doc.url or "",
        "snippet": _extract_snippet(doc.abstract or "", finding),
    }


def _resolve_trial(anchor: str, doc: Document) -> dict:
    """Build a trial entry from the Document."""
    contact = {}
    if doc.contacts:
        c = doc.contacts[0]
        contact = {
            "name": c.get("name", ""),
            "email": c.get("email", ""),
            "phone": c.get("phone", ""),
        }

    location = ""
    if doc.locations:
        loc = doc.locations[0]
        facility = loc.get("facility", "")
        country = loc.get("country", "")
        location = f"{facility}, {country}" if country else facility

    age_range = ""
    min_a = (doc.min_age or "").strip()
    max_a = (doc.max_age or "").strip()
    if min_a or max_a:
        parts = []
        if min_a:
            parts.append(f"From {min_a}")
        if max_a:
            parts.append(f"To {max_a}")
        age_range = " — ".join(parts)

    return {
        "nct_id": doc.nct_id or "",
        "title": doc.title or "",
        "status": doc.status or "",
        "eligibility_summary": doc.eligibility_criteria or "",
        "age_range": age_range,
        "start_date": doc.start_date or "",
        "location": location,
        "contact": contact,
    }


def assemble_response(
    llm_output: dict,
    doc_anchors: dict,
    stage_timings: dict | None = None,
    retrieval_counts: dict | None = None,
) -> AssembledResponse:
    """
    Stage 7: assemble the final user-facing response.

    Args:
        llm_output: parsed JSON from Stage 6 (overview, insights, trials, abstain_reason)
        doc_anchors: {"doc1": Document, ...} from Stage 5
        stage_timings: per-stage timing dict for pipelineMeta
        retrieval_counts: source counts for pipelineMeta

    Returns:
        AssembledResponse with fully resolved citations and metadata.
    """
    warnings: list[str] = []
    citation_stats = {"total": 0, "verified": 0, "unverified": 0}

    # --- Handle abstain case ---
    abstain_reason = llm_output.get("abstain_reason")
    if abstain_reason:
        return AssembledResponse(
            user_facing_json={
                "overview": llm_output.get("overview",
                    "Based on the retrieved research, I cannot directly answer this question."),
                "insights": [],
                "trials": [],
                "abstain_reason": abstain_reason,
                "suggestion": "Try rephrasing your question or providing more specific context.",
                "pipelineMeta": {
                    "stage_timings_ms": stage_timings or {},
                    "retrieval_counts": retrieval_counts or {},
                    "warnings": warnings,
                    "citation_stats": citation_stats,
                },
            },
            pipeline_meta=stage_timings or {},
            warnings=warnings,
        )

    # --- Resolve insights (publications only) ---
    resolved_insights = []
    for ins in llm_output.get("insights", []):
        finding = ins.get("finding", "")
        source_details = []
        unverified = False

        for anchor in ins.get("sources", []):
            citation_stats["total"] += 1
            doc = doc_anchors.get(anchor)
            if not doc:
                citation_stats["unverified"] += 1
                unverified = True
                continue
            if doc.doc_type != "publication":
                citation_stats["verified"] += 1
                continue
            source_details.append(_resolve_source(anchor, doc, finding))
            citation_stats["verified"] += 1

        if not source_details:
            continue

        resolved_insights.append({
            "finding": finding,
            "evidence_type": ins.get("evidence_type", "other"),
            "source_details": source_details,
            "unverified": unverified,
        })

    # --- Resolve trials (trial docs only) ---
    resolved_trials = []
    for trial in llm_output.get("trials", []):
        for anchor in trial.get("sources", []):
            citation_stats["total"] += 1
            doc = doc_anchors.get(anchor)
            if not doc:
                citation_stats["unverified"] += 1
                continue
            if doc.doc_type != "trial":
                citation_stats["verified"] += 1
                continue
            trial_entry = _resolve_trial(anchor, doc)
            trial_entry["relevance"] = trial.get("relevance", "")
            trial_entry["source_details"] = [_resolve_source(anchor, doc)]
            resolved_trials.append(trial_entry)
            citation_stats["verified"] += 1

    # --- Assemble final JSON ---
    user_facing = {
        "overview": llm_output.get("overview", ""),
        "insights": resolved_insights,
        "trials": resolved_trials,
        "recommendations": llm_output.get("recommendations", []),
        "follow_up_questions": llm_output.get("follow_up_questions", []),
        "abstain_reason": None,
        "pipelineMeta": {
            "stage_timings_ms": stage_timings or {},
            "retrieval_counts": retrieval_counts or {},
            "warnings": warnings,
            "citation_stats": citation_stats,
        },
    }

    return AssembledResponse(
        user_facing_json=user_facing,
        pipeline_meta=stage_timings or {},
        warnings=warnings,
    )
