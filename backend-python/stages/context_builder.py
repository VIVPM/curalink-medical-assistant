"""
Stage 5 — Context Builder.

Takes top 6-8 docs from Stage 4 + static context + chat history,
produces ONE finished prompt payload for Stage 6 (LLM Reasoner).

Token budget (Llama 3.1 8B, 8192 ctx):
  System + rules + schema:  ~500
  Static context:           ~150
  Chat history:             ~800  (last 5 turns, assistant condensed to ~150 tok)
  Retrieved docs:           ~5500 (equal share ~700 per doc)
  User message:             ~150
  Safety buffer:            ~300
  Reserved for output:      ~800
"""

from __future__ import annotations

from dataclasses import dataclass, field

from schemas.document import Document


GROUNDING_RULES = """\
You are a medical research assistant. Answer the user's question using ONLY the retrieved documents below.

RULES:
1. Answer ONLY using the documents provided. Do NOT use general medical knowledge.
2. Every factual claim MUST cite its source using [doc1], [doc2], etc.
3. If the documents don't answer the question, set abstain_reason — do NOT fabricate.
4. Do NOT recommend specific treatments or dosages; report what studies found.
5. If documents conflict, report both findings and note the conflict.
6. Distinguish study types: RCT, observational, review, clinical trial.
7. For trials, state status (RECRUITING, COMPLETED, etc.) clearly.
8. Keep overview to 2-3 sentences.

CRITICAL — SEPARATING INSIGHTS vs TRIALS:
- "insights" is ONLY for research publications (PubMed, OpenAlex). These are papers with authors, abstracts, and findings.
- "trials" is ONLY for clinical trials (ClinicalTrials.gov). These have NCT IDs, recruiting status, and eligibility criteria.
- Look at the document header to tell them apart:
    - Publications say: [docN] | pubmed | YEAR  or  [docN] | openalex | YEAR
    - Trials say: [docN] | ClinicalTrials.gov | NCT... | Status: ...
- NEVER put a ClinicalTrials.gov document into "insights".
- NEVER put a PubMed/OpenAlex publication into "trials".
- If only trials are retrieved, "insights" should be an empty array [].
- If only publications are retrieved, "trials" should be an empty array [].
- Include ALL relevant publications as separate insights (aim for 6-8 insights).
- Include ALL relevant trials as separate trial entries (aim for 6 trials).
- Each insight or trial should cite its own source — do not merge multiple docs into one entry.

OUTPUT FORMAT — respond with ONLY this JSON, no prose:
{
  "overview": "2-3 sentence condition/topic summary",
  "insights": [
    {"finding": "what the study found", "sources": ["doc1", "doc3"]}
  ],
  "trials": [
    {"nct_id": "NCT...", "title": "...", "relevance": "why this trial matters", "sources": ["doc5"]}
  ],
  "recommendations": [
    "actionable recommendation based on the research findings"
  ],
  "follow_up_questions": ["question 1", "question 2"],
  "abstain_reason": null
}

PERSONALIZED RECOMMENDATIONS:
- Generate 2-3 actionable recommendations based on the retrieved documents and patient context.
- Recommendations should be grounded in the research — not general advice.
- Focus on what the patient could discuss with their doctor, lifestyle considerations from studies, or relevant trials they may qualify for.
- Keep each recommendation to 1-2 sentences.
- Do NOT recommend specific treatments or dosages — suggest discussing findings with a healthcare provider.

FOLLOW-UP QUESTIONS:
- Generate exactly 2 relevant follow-up questions the user might want to ask next.
- Questions should be specific to the disease/topic and build on the retrieved documents.
- Make them actionable and different from each other (e.g., one about treatments, one about trials or prognosis).

If you cannot answer from the documents, set:
  "overview": "Based on the retrieved research, I cannot directly answer this question.",
  "insights": [],
  "trials": [],
  "abstain_reason": "explanation of why"
"""


@dataclass
class PromptPayload:
    system_prompt: str
    user_prompt: str
    token_count: int  # estimated
    doc_anchors: dict  # {"doc1": Document, "doc2": Document, ...}
    truncations: list[str] = field(default_factory=list)


def _format_publication(doc: Document, anchor: str) -> str:
    """Format a publication for the LLM context."""
    parts = [f"[{anchor}] | {doc.sources[0] if doc.sources else 'unknown'} | {doc.year or '?'}"]
    parts.append(f"Title: {doc.title}")
    if doc.authors:
        parts.append(f"Authors: {', '.join(doc.authors[:5])}")
    if doc.abstract:
        parts.append(f"Abstract: {doc.abstract}")
    return "\n".join(parts)


def _format_trial(doc: Document, anchor: str) -> str:
    """Format a clinical trial for the LLM context."""
    parts = [
        f"[{anchor}] | ClinicalTrials.gov | {doc.nct_id or '?'} | Status: {doc.status or '?'}"
    ]
    parts.append(f"Title: {doc.title}")
    if doc.abstract:
        parts.append(f"Brief Summary: {doc.abstract}")
    if doc.eligibility_criteria:
        elig = doc.eligibility_criteria
        # Truncation priority: keep first 400 tokens (~1600 chars) of eligibility
        if len(elig) > 1600:
            elig = elig[:1600].rsplit(".", 1)[0] + "."
        parts.append(f"Eligibility: {elig}")
    if doc.primary_outcomes:
        parts.append(f"Primary Outcomes: {'; '.join(doc.primary_outcomes[:3])}")
    if doc.locations:
        locs = doc.locations[:3]
        loc_strs = []
        for loc in locs:
            name = loc.get("facility", loc.get("city", "Unknown"))
            country = loc.get("country", "")
            loc_strs.append(f"{name}, {country}" if country else name)
        parts.append(f"Locations: {'; '.join(loc_strs)}")
    return "\n".join(parts)


def _truncate_to_budget(text: str, max_chars: int) -> tuple[str, bool]:
    """Sentence-aware truncation. Returns (text, was_truncated)."""
    if len(text) <= max_chars:
        return text, False
    truncated = text[:max_chars]
    # Cut at last sentence boundary
    last_period = truncated.rfind(".")
    if last_period > max_chars * 0.5:
        truncated = truncated[: last_period + 1]
    return truncated, True


def _format_chat_history(chat_history: list[dict] | None) -> str:
    """Format last 5 turns. Condense assistant answers to ~150 tokens."""
    if not chat_history:
        return ""

    turns = chat_history[-5:]
    parts = ["PREVIOUS CONVERSATION:"]
    for msg in turns:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "assistant":
            # Condense to ~150 tokens (~600 chars)
            if len(content) > 600:
                content = content[:600].rsplit(".", 1)[0] + "..."
        parts.append(f"{role.upper()}: {content}")

    return "\n".join(parts)


def build_context(
    top_docs: list[Document],
    user_message: str,
    static_context: dict,
    chat_history: list[dict] | None = None,
    per_doc_chars: int = 1600,  # ~400 tokens (14 docs fit in context)
) -> PromptPayload:
    """
    Stage 5: build the grounded prompt for the LLM.

    Args:
        top_docs: final ranked docs from Stage 4 (6-8 docs)
        user_message: current user question
        static_context: {disease, intent, location, patientName}
        chat_history: list of {role, content} dicts
        per_doc_chars: max chars per doc (~4 chars per token)

    Returns:
        PromptPayload ready for Stage 6.
    """
    doc_anchors: dict = {}
    truncations: list[str] = []

    # Format static context
    static_parts = ["PATIENT CONTEXT:"]
    disease = static_context.get("disease", "")
    if disease:
        static_parts.append(f"  Disease of Interest: {disease}")
    intent = static_context.get("intent", "")
    if intent:
        static_parts.append(f"  Additional Query/Intent: {intent}")
    location = static_context.get("location", "")
    if location:
        static_parts.append(f"  Location: {location}")
    static_block = "\n".join(static_parts)

    # Format chat history
    history_block = _format_chat_history(chat_history)

    # Format docs with anchors
    doc_blocks: list[str] = []
    for i, doc in enumerate(top_docs):
        anchor = f"doc{i + 1}"
        doc_anchors[anchor] = doc

        if doc.doc_type == "trial":
            formatted = _format_trial(doc, anchor)
        else:
            formatted = _format_publication(doc, anchor)

        formatted, was_truncated = _truncate_to_budget(formatted, per_doc_chars)
        if was_truncated:
            truncations.append(f"{anchor} truncated to {per_doc_chars} chars")

        doc_blocks.append(formatted)

    docs_block = "\n\n".join(doc_blocks)

    # Assemble user prompt
    user_prompt_parts = [static_block]
    if history_block:
        user_prompt_parts.append(history_block)
    user_prompt_parts.append(f"RETRIEVED DOCUMENTS:\n{docs_block}")
    user_prompt_parts.append(f"USER QUESTION:\n{user_message}")

    user_prompt = "\n\n".join(user_prompt_parts)

    # Rough token estimate (~4 chars per token)
    total_chars = len(GROUNDING_RULES) + len(user_prompt)
    token_estimate = total_chars // 4

    return PromptPayload(
        system_prompt=GROUNDING_RULES,
        user_prompt=user_prompt,
        token_count=token_estimate,
        doc_anchors=doc_anchors,
        truncations=truncations,
    )
