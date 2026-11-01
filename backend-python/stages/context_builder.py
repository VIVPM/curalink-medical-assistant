"""Build the grounded prompt from complete selected records, without text truncation.

Token counts are character-based estimates; provider usage gives actual total input tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from schemas.document import Document


GROUNDING_RULES = """\
You are a patient-facing medical information assistant, not a paper-summary service. Answer the CURRENT question directly in plain language, using the supplied evidence where available.

RULES:
1. Start with the answer to the patient's specific concern. Explain necessary medical terms. Never replace an answer with a list of studies.
2. Research findings, treatment benefits, numerical claims and trial details MUST be supported by the supplied documents and cite [doc1], [doc2], etc. Do not invent evidence or citations.
3. If evidence is partial, answer the supported parts and clearly explain what remains unknown. Do not claim there are no documents when some are supplied. If evidence is absent, do not invent medical facts; explain the limitation and an appropriate next step rather than only saying retrieval failed. Emergency guidance and clarification must not depend on research availability.
4. Do not diagnose, prescribe, change doses or tell patients to stop medicines. Explain supported treatment options as topics for their care team. No guarantees of cure, permanent benefit or absence of side effects.
5. Explain important evidence conflicts and population limitations. Do not generalize specialized findings to this patient.
6. A registry protocol, planned outcome or COMPLETED status is not evidence of treatment benefit. Report results only when actual results are supplied.
7. Include trial details only when the user asks about trials or a trial directly answers the question. Never add trial listings just because they were retrieved. A saved status is not verified current recruitment or eligibility.
8. Keep the overview to 2-3 short sentences that ANSWER THE QUESTION, not a generic condition summary. Honor requests for shorter wording, lists or a particular language. Resolve references from actual history; if the medicine, study or treatment is unidentified, ask a brief clarification instead of guessing.

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
- Include only insights that materially answer the current question; usually 0-3 concise insights are enough.
- There is no minimum number of insights or trials. Do not summarize every record or fill quotas.
- Merge overlapping findings with their supporting citations. Keep trials separate from publication insights.

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
    {"text": "research-informed point to discuss with a clinician", "sources": ["doc2"]}
  ],
  "follow_up_questions": ["question 1", "question 2"],
  "abstain_reason": null
}

PERSONALIZED RECOMMENDATIONS:
- Every recommendation MUST cite one or more retrieved documents using its "sources" field.
- Generate 0-2 points only when they directly address the patient's question; otherwise return [].
- Frame every point as something to discuss with a qualified healthcare professional, never as a directive.
- Recommendations should be grounded in the research — not general advice.
- Focus on relevant study findings, lifestyle considerations reported by studies, or trials worth reviewing.
- Do not diagnose, select a treatment, state that someone qualifies for a trial, or provide dosages.
- Keep each recommendation to 1-2 sentences.

FOLLOW-UP QUESTIONS:
- Generate 0-2 brief follow-up questions only when useful; return [] for emergencies, clarification, or a request for a short restatement.
- Stay with the patient's actual concern; do not introduce trials or treatments to fill a template.

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
    token_count: int
    doc_anchors: dict
    truncations: list[str] = field(default_factory=list)


def _format_publication(doc: Document, anchor: str) -> str:
    """Format a publication for the LLM context."""
    parts = [f"[{anchor}] | {doc.sources[0] if doc.sources else 'unknown'} | {doc.year or '?'}"]
    parts.append(f"Title: {doc.title}")
    if doc.authors:
        parts.append(f"Authors: {', '.join(doc.authors)}")
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
        parts.append(f"Eligibility: {doc.eligibility_criteria}")
    if doc.primary_outcomes:
        parts.append(f"Primary Outcomes: {'; '.join(doc.primary_outcomes)}")
    if doc.locations:
        locs = doc.locations
        loc_strs = []
        for loc in locs:
            name = loc.get("facility", loc.get("city", "Unknown"))
            country = loc.get("country", "")
            loc_strs.append(f"{name}, {country}" if country else name)
        parts.append(f"Locations: {'; '.join(loc_strs)}")
    return "\n".join(parts)


def _format_chat_history(chat_history: list[dict] | None) -> str:
    """Format last 6 turns. Summarize older ones as topic hints to cap token growth."""
    if not chat_history:
        return ""

    MAX_RECENT = 6
    parts = ["PREVIOUS CONVERSATION:"]

    if len(chat_history) > MAX_RECENT:

        older = chat_history[:-MAX_RECENT]
        topics = []
        for m in older:
            if m.get("role") == "user":
                first = (m.get("content") or "").split(".")[0].strip()[:80]
                if first:
                    topics.append(first)
        if topics:
            parts.append(f"[earlier: discussed {'; '.join(topics[:5])}]")
        chat_history = chat_history[-MAX_RECENT:]

    for msg in chat_history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "assistant":

            if len(content) > 600:
                content = content[:600].rsplit(".", 1)[0] + "..."
        parts.append(f"{role.upper()}: {content}")

    return "\n".join(parts)


def build_context(
    top_docs: list[Document],
    user_message: str,
    static_context: dict,
    chat_history: list[dict] | None = None,
) -> PromptPayload:
    """
    Stage 5: build the grounded prompt for the LLM.

    Args:
        top_docs: final ranked docs from Stage 4 (6-8 docs)
        user_message: current user question
        static_context: {disease, intent, location}
        chat_history: list of {role, content} dicts

    Returns:
        PromptPayload ready for Stage 6.
    """
    doc_anchors: dict = {}
    truncations: list[str] = []


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


    history_block = _format_chat_history(chat_history)


    doc_blocks: list[str] = []
    for i, doc in enumerate(top_docs):
        anchor = f"doc{i + 1}"
        doc_anchors[anchor] = doc

        if doc.doc_type == "trial":
            formatted = _format_trial(doc, anchor)
        else:
            formatted = _format_publication(doc, anchor)

        doc_blocks.append(formatted)

    docs_block = "\n\n".join(doc_blocks)


    user_prompt_parts = [static_block]
    if history_block:
        user_prompt_parts.append(history_block)
    user_prompt_parts.append(f"RETRIEVED DOCUMENTS:\n{docs_block}")
    user_prompt_parts.append(f"USER QUESTION:\n{user_message}")

    user_prompt = "\n\n".join(user_prompt_parts)


    total_chars = len(GROUNDING_RULES) + len(user_prompt)
    token_estimate = total_chars // 4

    return PromptPayload(
        system_prompt=GROUNDING_RULES,
        user_prompt=user_prompt,
        token_count=token_estimate,
        doc_anchors=doc_anchors,
        truncations=truncations,
    )
