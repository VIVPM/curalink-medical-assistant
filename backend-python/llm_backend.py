"""
LLM backend abstraction.

Single implementation — HuggingFace Inference API. The abstract interface is
kept so alternative backends can be plugged in later without touching pipeline
code.
"""

from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod
from typing import AsyncIterator

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


def get_llm_backend() -> LLMBackend:
    """Factory — currently returns HFBackend only."""
    token = os.getenv("HF_TOKEN")
    model = os.getenv("LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    if not token:
        raise RuntimeError("HF_TOKEN not set in .env")
    return HFBackend(token=token, model=model)
