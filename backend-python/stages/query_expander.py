"""
Stage 1 — Query Expander.

Takes user message + static form context + chat history, outputs
search-optimized queries for Stage 2 retrievers.

Single LLM call with strict JSON-output prompt. ~1-2s on HF Inference API.

Six jobs:
  1. Context injection — inject static disease into every variant
  2. Coreference resolution — resolve "it", "that" using chat history
  3. Medical synonym expansion — DBS -> deep brain stimulation
  4. Multi-query generation — 2-4 search variants
  5. Intent classification — 7 intents, used for trial status filter
  6. Entity extraction — diseases, drugs, procedures
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from llm_backend import LLMBackend

SYSTEM_PROMPT = """\
You are a medical search query optimizer. Your job is to rewrite a user's \
message into search-optimized queries for PubMed, OpenAlex, and ClinicalTrials.gov.

You will receive:
- STATIC CONTEXT: the patient's disease, intent, and location (locked for the session)
- CHAT HISTORY: prior conversation turns (if any)
- USER MESSAGE: the current question

Your tasks:
1. Inject the disease from static context into every expanded query
2. Resolve pronouns ("it", "that", "the treatment") using chat history
3. Expand medical abbreviations (DBS -> deep brain stimulation, MI -> myocardial infarction)
4. Generate 2-4 search query variants (different angles on the same question)
5. Classify the user's intent
6. Extract medical entities (diseases, drugs, procedures)

RULES:
- Always include the disease_focus in every expanded_query
- Include both lay terms AND clinical terms where applicable
- Do NOT answer the question — only rewrite it for search
- Output ONLY valid JSON, no prose before or after

Output this exact JSON schema:
{
  "expanded_queries": ["query1", "query2", "query3"],
  "entities": ["entity1", "entity2"],
  "intent": "one of: treatment_overview, drug_interaction_safety, clinical_trials_search, side_effects, prognosis, diagnosis_criteria, general_info",
  "disease_focus": "the primary disease from static context",
  "keywords": ["keyword1", "keyword2"],
  "skip_retrieval": false
}

Set skip_retrieval=true ONLY for greetings, thanks, or clearly non-medical messages.\
"""


@dataclass
class ExpanderResult:
    expanded_queries: list[str]
    entities: list[str]
    intent: str
    disease_focus: str
    keywords: list[str]
    skip_retrieval: bool = False
    raw_json: dict = field(default_factory=dict)
    timing_ms: int = 0


def _build_user_prompt(
    user_message: str,
    static_context: dict,
    chat_history: list[dict] | None = None,
) -> str:
    """Build the user prompt with context sections."""
    parts: list[str] = []

    # Static context
    disease = static_context.get("disease", "")
    intent = static_context.get("intent", "")
    location = static_context.get("location", "")
    patient = static_context.get("patientName", "")

    parts.append("STATIC CONTEXT:")
    parts.append(f"  Disease of Interest: {disease}")
    if intent:
        parts.append(f"  Additional Query/Intent: {intent}")
    if location:
        parts.append(f"  Location: {location}")
    if patient:
        parts.append(f"  Patient Name: {patient}")

    # Chat history
    if chat_history:
        parts.append("\nCHAT HISTORY (most recent last):")
        for msg in chat_history[-5:]:  # last 5 turns
            role = msg.get("role", "user")
            content = msg.get("content", "")
            # Condense assistant answers
            if role == "assistant" and len(content) > 150:
                content = content[:150] + "..."
            parts.append(f"  {role}: {content}")

    parts.append(f"\nUSER MESSAGE:\n{user_message}")

    return "\n".join(parts)


def _parse_response(raw: str, disease_fallback: str) -> dict:
    """Parse JSON from LLM response. Handles common issues."""
    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    return json.loads(text)


def _heuristic_fallback(
    user_message: str, static_context: dict
) -> dict:
    """Fallback when LLM fails to produce valid JSON."""
    disease = static_context.get("disease", "")
    intent = static_context.get("intent", "")
    combined = f"{user_message} {disease}"
    if intent:
        combined += f" {intent}"

    return {
        "expanded_queries": [combined.strip()],
        "entities": [disease] if disease else [],
        "intent": "general_info",
        "disease_focus": disease,
        "keywords": [w.lower() for w in user_message.split()[:5]],
        "skip_retrieval": False,
    }


VALID_INTENTS = {
    "treatment_overview", "drug_interaction_safety", "clinical_trials_search",
    "side_effects", "prognosis", "diagnosis_criteria", "general_info",
}


async def expand_query(
    user_message: str,
    static_context: dict,
    chat_history: list[dict] | None,
    llm: LLMBackend,
) -> ExpanderResult:
    """
    Stage 1: expand user message into search-optimized queries.

    Args:
        user_message: current user question
        static_context: {disease, intent, location, patientName}
        chat_history: list of {role, content} dicts (may be None/empty)
        llm: LLMBackend instance

    Returns:
        ExpanderResult with expanded queries, entities, intent, etc.
    """
    disease = static_context.get("disease", "")
    user_prompt = _build_user_prompt(user_message, static_context, chat_history)

    t0 = time.perf_counter()
    parsed = None

    # Try LLM call, retry once on JSON parse failure
    for attempt in range(2):
        try:
            prompt = user_prompt
            if attempt == 1:
                prompt += "\n\nIMPORTANT: Output ONLY valid JSON. No prose."

            raw = await llm.generate(
                prompt,
                system_prompt=SYSTEM_PROMPT,
                max_tokens=400,
                temperature=0.2,
                json_mode=True,
            )
            parsed = _parse_response(raw, disease)
            break
        except Exception:
            if attempt == 0:
                continue
            # Second failure — use heuristic fallback
            parsed = _heuristic_fallback(user_message, static_context)

    if parsed is None:
        parsed = _heuristic_fallback(user_message, static_context)

    timing_ms = round((time.perf_counter() - t0) * 1000)

    # Validate and sanitize
    expanded = parsed.get("expanded_queries", [])
    if not expanded:
        combined = f"{user_message} {disease}".strip()
        expanded = [combined]
    # Cap at 4 variants
    expanded = expanded[:4]
    # Ensure disease is in every query
    for i, q in enumerate(expanded):
        if disease.lower() not in q.lower():
            expanded[i] = f"{q} {disease}"

    intent = parsed.get("intent", "general_info")
    if intent not in VALID_INTENTS:
        intent = "general_info"

    return ExpanderResult(
        expanded_queries=expanded,
        entities=parsed.get("entities", []),
        intent=intent,
        disease_focus=parsed.get("disease_focus", disease),
        keywords=parsed.get("keywords", []),
        skip_retrieval=parsed.get("skip_retrieval", False),
        raw_json=parsed,
        timing_ms=timing_ms,
    )
