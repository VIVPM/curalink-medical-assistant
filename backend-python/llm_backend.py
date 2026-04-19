"""
LLM backend abstraction.

One interface, multiple implementations. Pipeline code calls `LLMBackend`;
the concrete backend is picked at startup via the `LLM_BACKEND` env var.

For the hackathon we ship only GroqBackend. OllamaBackend is a stub so
the offline-fallback path is obvious when someone wants to add it later.
"""

from __future__ import annotations

import asyncio
import json
import os
from abc import ABC, abstractmethod
from typing import AsyncIterator

import httpx
from groq import AsyncGroq
from huggingface_hub import InferenceClient


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


class GroqBackend(LLMBackend):
    def __init__(self, api_key: str, model: str):
        self.client = AsyncGroq(api_key=api_key)
        self.model = model

    def _messages(self, prompt: str, system_prompt: str | None = None) -> list[dict]:
        msgs: list[dict] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def _response_format(self, json_mode: bool) -> dict | None:
        return {"type": "json_object"} if json_mode else None

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 800,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> str:
        kwargs = {
            "model": self.model,
            "messages": self._messages(prompt, system_prompt),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = await self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    async def generate_stream(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 800,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> AsyncIterator[str]:
        kwargs = {
            "model": self.model,
            "messages": self._messages(prompt, system_prompt),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        stream = await self.client.chat.completions.create(**kwargs)
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


class HFBackend(LLMBackend):
    """HuggingFace Inference API backend using huggingface_hub InferenceClient."""

    def __init__(self, token: str, model: str):
        self.token = token
        self.model = model
        self.client = InferenceClient(model=model, token=token)

    def _build_messages(self, prompt: str, system_prompt: str | None) -> list[dict]:
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 800,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> str:
        sys = system_prompt or ""
        if json_mode:
            sys += "\n\nYou MUST respond with ONLY valid JSON. No prose before or after."

        messages = self._build_messages(prompt, sys)

        # InferenceClient is sync — run in thread to not block async loop
        resp = await asyncio.to_thread(
            self.client.chat_completion,
            messages=messages,
            max_tokens=max_tokens,
            temperature=max(temperature, 0.01),
        )
        return resp.choices[0].message.content or ""

    async def generate_stream(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 800,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> AsyncIterator[str]:
        sys = system_prompt or ""
        if json_mode:
            sys += "\n\nYou MUST respond with ONLY valid JSON. No prose before or after."

        messages = self._build_messages(prompt, sys)

        # Streaming is sync in huggingface_hub — collect in thread, yield tokens
        def _stream():
            tokens = []
            for chunk in self.client.chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                stream=True,
            ):
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    tokens.append(delta)
            return tokens

        tokens = await asyncio.to_thread(_stream)
        for token in tokens:
            yield token


class OllamaBackend(LLMBackend):
    """Stub for offline fallback."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("OllamaBackend is a stub.")

    async def generate(self, *args, **kwargs) -> str:
        raise NotImplementedError

    async def generate_stream(self, *args, **kwargs) -> AsyncIterator[str]:
        raise NotImplementedError
        yield


def get_llm_backend() -> LLMBackend:
    """
    Factory selected by the LLM_BACKEND env var.
    Supported: groq, hf, ollama (stub).
    """
    backend_name = os.getenv("LLM_BACKEND", "groq").lower()

    if backend_name == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        model = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set in .env")
        return GroqBackend(api_key=api_key, model=model)

    if backend_name == "hf":
        token = os.getenv("HF_TOKEN")
        model = os.getenv("LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
        if not token:
            raise RuntimeError("HF_TOKEN not set in .env")
        return HFBackend(token=token, model=model)

    if backend_name == "ollama":
        return OllamaBackend()

    raise RuntimeError(
        f"Unknown LLM_BACKEND={backend_name!r}. Supported: groq, hf, ollama (stub)."
    )
