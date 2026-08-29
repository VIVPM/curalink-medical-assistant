"""
LLM backend abstraction — multi-provider.

LLM_MODEL env var selects the provider:
  CLOUDFLARE  -> Cloudflare Workers AI gpt-oss-20b (OpenAI-compatible, httpx)
  <anything else> -> HuggingFace Inference API (huggingface_hub, value is the model id)

No fallback: whatever LLM_MODEL is set to, that provider is used. If the
required env vars for that provider are missing, startup fails.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from abc import ABC, abstractmethod
from typing import AsyncIterator

logger = logging.getLogger(__name__)

_LLM_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "3"))
_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0


class LLMBackend(ABC):
    """Abstract interface every LLM backend must satisfy."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 800,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> str:
        """Return the full generated string (non-streaming)."""

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 800,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> AsyncIterator[str]:
        """Yield tokens as they arrive from the model."""


# ---------------------------------------------------------------------------
# HuggingFace Inference API
# ---------------------------------------------------------------------------
class HFBackend(LLMBackend):
    """HuggingFace Inference API backend using huggingface_hub InferenceClient."""

    def __init__(self, token: str, model: str):
        self.token = token
        self.model = model
        from huggingface_hub import InferenceClient
        self.client = InferenceClient(model=model, token=token)
        self._sem = asyncio.Semaphore(_LLM_CONCURRENCY)

    async def _call(self, fn):
        async with self._sem:
            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    return await asyncio.to_thread(fn)
                except Exception as e:
                    if attempt == _MAX_RETRIES:
                        raise
                    wait = _RETRY_BACKOFF * (2 ** (attempt - 1)) * (0.5 + random.random())
                    logger.warning("[hf] attempt %d/%d failed: %s. Retrying in %.1fs...",
                                   attempt, _MAX_RETRIES, e, wait)
                    await asyncio.sleep(wait)

    def _build_messages(self, prompt: str, system_prompt: str | None) -> list[dict]:
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    async def generate(self, prompt, *, system_prompt=None, max_tokens=800,
                       temperature=0.2, json_mode=False) -> str:
        sys = system_prompt or ""
        if json_mode:
            sys += "\n\nYou MUST respond with ONLY valid JSON. No prose before or after."

        # Prompt cache: hit = skip the LLM call entirely
        from redis_cache import get_prompt_cache, set_prompt_cache
        cached = get_prompt_cache(self.model, sys, prompt)
        if cached is not None:
            logger.info("[hf] prompt cache hit (%d chars)", len(cached))
            return cached

        messages = self._build_messages(prompt, sys)

        import observability as obs
        with obs.llm_generation(self.model, prompt) as span:
            resp = await self._call(
                lambda: self.client.chat_completion(
                    messages=messages, max_tokens=max_tokens,
                    temperature=max(temperature, 0.01),
                )
            )
            text = resp.choices[0].message.content or ""
            obs.set_generation_output(span, text)

        set_prompt_cache(self.model, sys, prompt, text)
        return text

    async def generate_stream(self, prompt, *, system_prompt=None, max_tokens=800,
                              temperature=0.2, json_mode=False) -> AsyncIterator[str]:
        sys = system_prompt or ""
        if json_mode:
            sys += "\n\nYou MUST respond with ONLY valid JSON. No prose before or after."
        messages = self._build_messages(prompt, sys)

        def _stream():
            tokens = []
            for chunk in self.client.chat_completion(
                messages=messages, max_tokens=max_tokens,
                temperature=max(temperature, 0.01), stream=True,
            ):
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    tokens.append(delta)
            return tokens

        import observability as obs
        with obs.llm_generation(self.model, prompt) as span:
            tokens = await self._call(_stream)
            obs.set_generation_output(span, "".join(tokens))
        for token in tokens:
            yield token


# ---------------------------------------------------------------------------
# Cloudflare Workers AI (OpenAI-compatible /chat/completions)
# ---------------------------------------------------------------------------
class CloudflareBackend(LLMBackend):
    """Cloudflare Workers AI backend via httpx (OpenAI wire format)."""

    def __init__(self, account_id: str, api_token: str):
        import httpx as _httpx
        self.model = "@cf/openai/gpt-oss-20b"
        self._max_tokens = 4096
        self._base = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
        self._headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }
        self._sem = asyncio.Semaphore(_LLM_CONCURRENCY)
        self._httpx = _httpx

    def _messages(self, prompt: str, system_prompt: str | None) -> list[dict]:
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    async def generate(self, prompt, *, system_prompt=None, max_tokens=800,
                       temperature=0.2, json_mode=False) -> str:
        sys = system_prompt or ""
        if json_mode:
            sys += "\n\nYou MUST respond with ONLY valid JSON. No prose before or after."

        # Prompt cache
        from redis_cache import get_prompt_cache, set_prompt_cache
        cached = get_prompt_cache(self.model, sys, prompt)
        if cached is not None:
            logger.info("[cf] prompt cache hit (%d chars)", len(cached))
            return cached

        messages = self._messages(prompt, sys)

        import observability as obs
        with obs.llm_generation(self.model, prompt) as span:
            async with self._sem:
                for attempt in range(1, _MAX_RETRIES + 1):
                    try:
                        async with self._httpx.AsyncClient(timeout=90) as client:
                            r = await client.post(
                                f"{self._base}/chat/completions",
                                headers=self._headers,
                                json={
                                    "model": self.model,
                                    "messages": messages,
                                    "temperature": temperature,
                                    "max_tokens": self._max_tokens,
                                },
                            )
                            r.raise_for_status()
                            text = (r.json()["choices"][0]["message"].get("content") or "").strip()
                            obs.set_generation_output(span, text)
                            set_prompt_cache(self.model, sys, prompt, text)
                            return text
                    except Exception as e:
                        if attempt == _MAX_RETRIES:
                            raise
                        wait = _RETRY_BACKOFF * (2 ** (attempt - 1)) * (0.5 + random.random())
                        logger.warning("[cf] attempt %d/%d failed: %s. Retrying in %.1fs...",
                                       attempt, _MAX_RETRIES, e, wait)
                        await asyncio.sleep(wait)

    async def generate_stream(self, prompt, *, system_prompt=None, max_tokens=800,
                              temperature=0.2, json_mode=False) -> AsyncIterator[str]:
        sys = system_prompt or ""
        if json_mode:
            sys += "\n\nYou MUST respond with ONLY valid JSON. No prose before or after."
        messages = self._messages(prompt, sys)

        import observability as obs
        with obs.llm_generation(self.model, prompt) as span:
            full = []
            async with self._httpx.AsyncClient(timeout=120) as client:
                async with client.stream(
                    "POST", f"{self._base}/chat/completions",
                    headers=self._headers,
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": self._max_tokens,
                        "stream": True,
                    },
                ) as r:
                    r.raise_for_status()
                    async for line in r.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            delta = json.loads(data)["choices"][0]["delta"].get("content")
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
                        if delta:
                            full.append(delta)
                            yield delta
            obs.set_generation_output(span, "".join(full))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def get_llm_backend() -> LLMBackend:
    """Return the LLM backend selected by LLM_MODEL env var.

    LLM_MODEL=CLOUDFLARE  -> CloudflareBackend (needs CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN)
    LLM_MODEL=<other>     -> HFBackend (needs HF_TOKEN; value is the HF model id)
    """
    raw = os.getenv("LLM_MODEL")
    if not raw:
        raise RuntimeError(
            "LLM_MODEL must be set. Use CLOUDFLARE or a HuggingFace model id."
        )

    provider = raw.strip().upper()

    if provider == "CLOUDFLARE":
        account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
        api_token = os.getenv("CLOUDFLARE_API_TOKEN")
        missing = [k for k, v in (("CLOUDFLARE_ACCOUNT_ID", account_id),
                                   ("CLOUDFLARE_API_TOKEN", api_token)) if not v]
        if missing:
            raise RuntimeError(
                f"LLM_MODEL=CLOUDFLARE also needs {', '.join(missing)} in .env."
            )
        logger.info("LLM provider: Cloudflare Workers AI (@cf/openai/gpt-oss-20b)")
        return CloudflareBackend(account_id=account_id, api_token=api_token)

    # Default: treat LLM_MODEL as a HuggingFace model id
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(f"LLM_MODEL={raw} (HuggingFace) requires HF_TOKEN in .env.")
    logger.info("LLM provider: HuggingFace (%s)", raw)
    return HFBackend(token=token, model=raw)
