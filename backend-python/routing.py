"""Shared routing preparation for regular, streamed and background requests."""

import json
import re
from dataclasses import dataclass
from typing import Literal

from token_budget import TokenBudget

from llm_backend import LLMBackend, get_tier_backend
from stages.history_context import bounded_history, build_history_context
from stages.llm_reasoner import run_reasoner
from stages.query_router import RoutingDecision, classify_query, routing_enabled
from stages.response_assembler import assemble_response


@dataclass
class RoutedRequest:
    static_context: dict
    chat_history: list[dict]
    expansion_llm: LLMBackend
    answer_llm: LLMBackend
    decision: RoutingDecision | None = None
    response: dict | None = None

    def metadata(self) -> dict:
        if self.decision is None:
            return {}
        return {
            **self.decision.model_dump(),
            "answer_model": getattr(self.answer_llm, "model", "") if self.decision.route in {"rag", "conversation"} else None,
            "retrieval_used": self.decision.route == "rag",
        }


def _message_response(message: str) -> dict:
    return {
        "overview": message, "insights": [], "trials": [], "recommendations": [],
        "follow_up_questions": [], "abstain_reason": None,
        "pipelineMeta": {"warnings": [], "retrieval_counts": {}},
    }


_SYMPTOM = r"(?:trouble breathing|difficulty breathing|struggling to breathe|barely breathe|chest pain|chest pressure|severe bleeding|bleeding heavily|(?:a )?swollen tongue|tongue swelling)"
_URGENT = re.compile(
    r"\b(?:overdose|suicidal|kill myself|self[- ]harm|anaphylaxis|can't breathe|cannot breathe|unable to breathe|"
    r"trouble breathing|difficulty breathing|struggling to breathe|barely breathe|extremely breathless|"
    r"bleeding heavily|severe bleeding|unconscious|unresponsive|"
    r"(?:tongue|throat)\s+(?:is\s+|has\s+been\s+)?(?:swelling|swollen)|"
    r"(?:swollen|swelling)\s+(?:tongue|throat))\b"
)
_CHEST = re.compile(r"\b(?:chest pain|chest pressure|pressure in (?:my |the )?chest)\b")
_HYPOTHETICAL = re.compile(
    r"^\s*(?:hypothetically|for a story|suppose|imagine|what if|if someone|if a person|if i|what are the signs|what are symptoms)\b"
)
_NEW_PATIENT_FACT = tuple(map(re.compile, (
    r"\b(?:i|we|he|she|they)\s+(?:need|use|am using|are using)\s+.{0,35}\bmore(?: often)?\b",
    r"\b(?:my|his|her|their|our)\s+\w+(?:\s+\w+){0,5}\s+(?:has|have|is|are)\s+(?:gotten|become|been|getting)?\s*(?:worse|worsening|more frequent)\b",
    r"\b(?:i|we|he|she|they)\s+(?:started|developed)\s+(?:(?:a|an|the|my|our)\s+)?(?:new|different)\b",
    r"\b(?:i|we|he|she|they)\s+(?:now|recently|lately)\s+(?:have|has|developed|started)\b",
    r"\b(?:it|my|his|her|their)\s+(?:is|has been)\s+(?:now|recently|lately)\s+(?:waking|keeping)\s+(?:me|him|her|them)\b",
)))


def _active_urgent_clause(user_message: str) -> bool:
    text = user_message.lower().replace("’", "'")
    text = re.sub(r'"[^"\n]*"|“[^”\n]*”|‘[^’\n]*’|(?<!\w)\'(?:[^\'\n]|\'(?=\w))*\'(?!\w)', " ", text)
    for clause in re.split(r"[.!?;\n]+", text):
        if _HYPOTHETICAL.search(clause):
            continue
        clause = re.sub(rf"\b(?:no|not having|don't have|do not have|without)\s+(?:any\s+)?{_SYMPTOM}(?:\s+or\s+{_SYMPTOM})*", " ", clause)
        clause = re.sub(r"\b(?:am|is|are|was|were)\s+not\s+(?:struggling to breathe|bleeding heavily|swelling|suicidal|unconscious|unresponsive|having\s+(?:a\s+)?(?:seizure|trouble breathing|difficulty breathing))", " ", clause)
        clause = re.sub(r"\b(?:did|do|does|have|has)\s+not\s+(?:overdose|overdosed)\b", " ", clause)
        clause = re.sub(r"\b(?:tongue|throat)\s+is\s+not\s+(?:swelling|swollen)", " ", clause)
        if _URGENT.search(clause) or (_CHEST.search(clause) and re.search(r"\b(?:now|crushing|sweating|jaw)\b", clause)):
            return True
    return False


def _reports_new_patient_fact(user_message: str) -> bool:
    text = user_message.lower().replace("’", "'")
    return any(pattern.search(text) for pattern in _NEW_PATIENT_FACT)


def immediate_response(user_message: str, history: list[dict]) -> tuple[Literal["emergency", "clarification"], str] | None:
    """Handle explicit urgent symptoms and absent references before any provider call."""
    text = user_message.lower().replace("’", "'")
    if _active_urgent_clause(user_message):
        return ("emergency", "This could be a medical emergency. Call your local emergency number now and follow the dispatcher's instructions. Do not wait for research results or your next appointment. I cannot diagnose the cause here.")
    reference = bool(re.search(r"\b(?:that trial|that study|the (?:medicine|drug|treatment|one) you (?:mentioned|recommended)|trial you showed)\b", text))
    if reference and not any(m.get("role") == "assistant" and m.get("content", "").strip() for m in history):
        return ("clarification", "Which medicine, treatment or study do you mean? Please share its name or link; I don't have an earlier identified item to refer to.")
    return None


async def route_request(user_message: str, static_context: dict, history: list[dict], base_llm: LLMBackend, *, decision: RoutingDecision | None = None, budget: TokenBudget | None = None) -> RoutedRequest:
    direct = immediate_response(user_message, history)
    if direct is not None:
        route, message = direct
        response = _message_response(message)
        routed = RoutedRequest(dict(static_context), bounded_history(history), base_llm, base_llm,
                               RoutingDecision(route=route, fallback="pre_retrieval_guard"), response)
        response["pipelineMeta"]["routing"] = routed.metadata()
        return routed
    if not routing_enabled():
        return RoutedRequest(static_context, history, base_llm, base_llm)
    history = bounded_history(history)
    if budget is not None and decision is None:
        budget.consume_text(json.dumps({"intake": static_context, "history": history, "question": user_message}))
    decision = decision or await classify_query(user_message, static_context, history)
    if decision.route == "conversation" and _reports_new_patient_fact(user_message):
        decision.route = "rag"
        decision.fallback = "new_patient_fact"
    context = dict(static_context)
    if decision.topic == "new_topic":
        context.update(disease="", intent="")
    routed = RoutedRequest(
        context, history, get_tier_backend(base_llm, "8b"),
        get_tier_backend(base_llm, decision.model_tier), decision,
    )
    if decision.route == "smalltalk":
        routed.response = _message_response("I'm here to help with medical research. Ask a research question whenever you're ready.")
    elif decision.route == "out_of_scope":
        routed.response = _message_response("I can help explain medical research and clinical-trial listings. Please ask a question in that scope.")
    elif decision.route == "conversation":
        payload = build_history_context(history, user_message, context)
        if payload is not None:
            if budget is not None:
                budget.consume_text(payload.user_prompt)
                budget.consume_text(payload.system_prompt)
            result = await run_reasoner(payload, routed.answer_llm)
            if budget is not None:
                budget.consume_text(json.dumps(result.llm_output))
            if not result.llm_output.get("abstain_reason") and result.llm_output.get("overview"):
                routed.response = assemble_response(result.llm_output, payload.doc_anchors).user_facing_json
        if routed.response is None:
            decision.route = "rag"
            decision.fallback = "history_insufficient"
    if routed.response is not None:
        routed.response["pipelineMeta"]["routing"] = routed.metadata()
    return routed
