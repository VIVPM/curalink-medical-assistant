"""Opt-in Gemini generation for evaluation only; production providers are unchanged."""

MODEL = "gemini-3.5-flash-lite"
SOURCES = {"type": "array", "items": {"type": "string"}}
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "overview": {"type": "string"},
        "insights": {"type": "array", "items": {"type": "object", "properties": {
            "finding": {"type": "string"}, "sources": SOURCES}, "required": ["finding", "sources"]}},
        "trials": {"type": "array", "items": {"type": "object", "properties": {
            "nct_id": {"type": "string"}, "title": {"type": "string"},
            "relevance": {"type": "string"}, "sources": SOURCES},
            "required": ["nct_id", "title", "relevance", "sources"]}},
        "recommendations": {"type": "array", "items": {"type": "object", "properties": {
            "text": {"type": "string"}, "sources": SOURCES}, "required": ["text", "sources"]}},
        "follow_up_questions": {"type": "array", "items": {"type": "string"}},
        "abstain_reason": {"type": ["string", "null"]},
    },
    "required": ["overview", "insights", "trials", "recommendations", "follow_up_questions", "abstain_reason"],
}
RATES = {"input_per_million": 0.30, "output_per_million": 2.50,
         "source": "https://ai.google.dev/gemini-api/docs/pricing", "checked_on": "2026-11-01"}


def usage_record(metadata):
    if metadata is None or metadata.prompt_token_count is None or metadata.candidates_token_count is None:
        return {"cost_usd": None, "status": "usage_unavailable"}
    inputs = metadata.prompt_token_count
    outputs = metadata.candidates_token_count
    thoughts = metadata.thoughts_token_count or 0
    if any(type(value) is not int or value < 0 for value in (inputs, outputs, thoughts)):
        raise ValueError("Invalid Gemini usage counters")
    return {"input_tokens": inputs, "output_tokens": outputs, "thinking_tokens": thoughts,
            "cost_usd": (inputs * RATES["input_per_million"] + (outputs + thoughts) * RATES["output_per_million"]) / 1e6,
            "status": "estimated", "rate_provenance": RATES}


def answer_generation_cost(records):
    answers = [record for record in records if record.get("purpose") == "answer"]
    if any(record["cost_usd"] is None for record in answers):
        return None
    return sum(record["cost_usd"] for record in answers)


class EvalGeminiBackend:
    model = MODEL
    use_cache = False

    def __init__(self, api_key):
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is required")
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.usage = []

    def _config(self, system_prompt, max_tokens, temperature, json_mode, answer=False):
        from google.genai import types
        return types.GenerateContentConfig(
            system_instruction=system_prompt, max_output_tokens=max_tokens,
            temperature=temperature,
            response_mime_type="application/json" if json_mode else "text/plain",
            response_json_schema=ANSWER_SCHEMA if answer and json_mode else None,
        )

    async def generate(self, prompt, *, system_prompt=None, max_tokens=800, temperature=0.2, json_mode=False):
        from stages.context_builder import GROUNDING_RULES
        answer = bool(system_prompt and system_prompt.startswith(GROUNDING_RULES))
        response = None
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model, contents=prompt,
                config=self._config(system_prompt, max_tokens, temperature, json_mode, answer=answer),
            )
            return response.text or ""
        finally:
            self.usage.append({**usage_record(response.usage_metadata if response is not None else None),
                               "purpose": "answer" if answer else "query_expansion"})

    async def generate_stream(self, prompt, *, system_prompt=None, max_tokens=800, temperature=0.2, json_mode=False):
        usage = None
        try:
            stream = await self.client.aio.models.generate_content_stream(
                model=self.model, contents=prompt,
                config=self._config(system_prompt, max_tokens, temperature, json_mode, answer=True),
            )
            async for chunk in stream:
                if chunk.usage_metadata is not None:
                    usage = chunk.usage_metadata
                if chunk.text:
                    yield chunk.text
        finally:
            # Streaming counters are cumulative: record the last, never sum every chunk.
            self.usage.append({**usage_record(usage), "purpose": "answer"})

    async def aclose(self):
        await self.client.aio.aclose()
