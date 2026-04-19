"""
Stage 6 — LLM Reasoner.

Takes the PromptPayload from Stage 5, sends it to the LLM (HF Inference),
parses the structured JSON response, and validates against the schema.
Retries once on schema failure.

Output: raw parsed dict with overview, insights, trials, abstain_reason.
Stage 7 (Response Assembler) handles citation resolution.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from llm_backend import LLMBackend
from stages.context_builder import PromptPayload


@dataclass
class ReasonerResult:
    llm_output: dict  # parsed JSON from LLM
    raw_text: str = ""
    timing_ms: int = 0
    retried: bool = False
    parse_error: str | None = None


REQUIRED_KEYS = {"overview", "insights", "trials", "abstain_reason"}


def _validate_schema(parsed: dict) -> list[str]:
    """Check the LLM output has the required structure. Returns list of issues."""
    issues: list[str] = []

    for key in REQUIRED_KEYS:
        if key not in parsed:
            issues.append(f"missing key: {key}")

    if "insights" in parsed and not isinstance(parsed["insights"], list):
        issues.append("insights must be a list")

    if "trials" in parsed and not isinstance(parsed["trials"], list):
        issues.append("trials must be a list")

    if "insights" in parsed and isinstance(parsed["insights"], list):
        for i, ins in enumerate(parsed["insights"]):
            if not isinstance(ins, dict):
                issues.append(f"insights[{i}] not a dict")
            elif "finding" not in ins:
                issues.append(f"insights[{i}] missing 'finding'")
            elif "sources" not in ins:
                issues.append(f"insights[{i}] missing 'sources'")

    return issues


def _parse_llm_response(raw: str) -> dict:
    """Parse JSON from LLM. Handles markdown fences and common LLM quirks."""
    import re
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fix trailing commas before ] or }
    text = re.sub(r',\s*([}\]])', r'\1', text)

    # If JSON is truncated, try to close it
    open_braces = text.count('{') - text.count('}')
    open_brackets = text.count('[') - text.count(']')
    # Trim any trailing partial value (e.g. truncated string)
    if open_braces > 0 or open_brackets > 0:
        # Remove trailing partial string/value after last complete entry
        text = re.sub(r',\s*"[^"]*$', '', text)  # trailing incomplete key
        text = re.sub(r',\s*$', '', text)
        text += ']' * open_brackets + '}' * open_braces

    return json.loads(text)


def _fallback_output(error_msg: str) -> dict:
    """Produce a valid schema when LLM fails completely."""
    return {
        "overview": "I was unable to generate a structured response from the retrieved research.",
        "insights": [],
        "trials": [],
        "abstain_reason": f"LLM response parsing failed: {error_msg}",
    }


async def run_reasoner(
    payload: PromptPayload,
    llm: LLMBackend,
    max_tokens: int = 4000,
    temperature: float = 0.2,
) -> ReasonerResult:
    """
    Stage 6: send the grounded prompt to the LLM and parse structured output.

    Args:
        payload: PromptPayload from Stage 5
        llm: LLMBackend instance
        max_tokens: max output tokens
        temperature: LLM temperature

    Returns:
        ReasonerResult with parsed JSON output.
    """
    t0 = time.perf_counter()
    retried = False
    parse_error = None

    for attempt in range(2):
        try:
            prompt = payload.user_prompt
            if attempt == 1:
                prompt += "\n\nIMPORTANT: You MUST respond with ONLY valid JSON matching the schema. No prose."
                retried = True

            raw = await llm.generate(
                prompt,
                system_prompt=payload.system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                json_mode=True,
            )

            parsed = _parse_llm_response(raw)
            issues = _validate_schema(parsed)

            if issues and attempt == 0:
                parse_error = f"schema issues: {issues}"
                continue

            # Fill in missing keys with defaults
            parsed.setdefault("overview", "")
            parsed.setdefault("insights", [])
            parsed.setdefault("trials", [])
            parsed.setdefault("abstain_reason", None)

            timing_ms = round((time.perf_counter() - t0) * 1000)
            return ReasonerResult(
                llm_output=parsed,
                raw_text=raw,
                timing_ms=timing_ms,
                retried=retried,
                parse_error=parse_error if issues else None,
            )

        except Exception as e:
            parse_error = f"{type(e).__name__}: {e}"
            if attempt == 0:
                continue

    # Both attempts failed
    timing_ms = round((time.perf_counter() - t0) * 1000)
    return ReasonerResult(
        llm_output=_fallback_output(parse_error or "unknown error"),
        raw_text="",
        timing_ms=timing_ms,
        retried=True,
        parse_error=parse_error,
    )
