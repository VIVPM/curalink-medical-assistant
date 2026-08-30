"""
Per-job token budget — caps total tokens (input + output) per pipeline run.

Prevents prompt injection or oversized queries from inflating LLM cost.
Estimate: ~1 token per 4 chars (rough, good enough for budgeting).
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

MAX_TOKENS_PER_JOB = int(os.getenv("MAX_TOKENS_PER_JOB", "50000"))


class TokenBudgetExceeded(Exception):
    pass


class TokenBudget:
    def __init__(self, max_tokens: int | None = None):
        self.max_tokens = max_tokens or MAX_TOKENS_PER_JOB
        self.used = 0

    def estimate(self, text: str) -> int:
        """Rough token estimate: 1 token ≈ 4 chars."""
        return max(1, len(text) // 4)

    def consume(self, tokens: int) -> None:
        self.used += tokens
        if self.used > self.max_tokens:
            logger.warning("[budget] exceeded: %d / %d", self.used, self.max_tokens)
            raise TokenBudgetExceeded(
                f"Token budget exceeded: {self.used}/{self.max_tokens}"
            )

    def consume_text(self, text: str) -> None:
        """Estimate tokens from text and consume."""
        self.consume(self.estimate(text))

    def remaining(self) -> int:
        return max(0, self.max_tokens - self.used)

    def summary(self) -> dict:
        return {
            "used": self.used,
            "max": self.max_tokens,
            "remaining": self.remaining(),
        }
