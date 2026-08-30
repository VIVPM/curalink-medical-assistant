"""
Circuit breaker + auto-fallback for LLM providers.

States: closed (normal) → open (failing, fast-reject) → half-open (probing).
After FAILURE_THRESHOLD consecutive failures the circuit opens, rejecting calls
instantly (503) until RESET_TIMEOUT_S elapses, then allows one probe request.
A success closes the circuit; another failure re-opens it.

Auto-fallback: when the primary provider's circuit is open, ResilientLLM
transparently routes to the fallback provider (if configured).
"""

from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger(__name__)

FAILURE_THRESHOLD = int(os.getenv("BREAKER_FAILURE_THRESHOLD", "5"))
RESET_TIMEOUT_S = int(os.getenv("BREAKER_RESET_TIMEOUT", "60"))


class CircuitOpen(Exception):
    """Raised when a call is rejected because the circuit is open."""


class CircuitBreaker:
    def __init__(self, name: str, failure_threshold: int = FAILURE_THRESHOLD,
                 reset_timeout: int = RESET_TIMEOUT_S):
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self._failures = 0
        self._state = "closed"
        self._opened_at: float = 0

    @property
    def state(self) -> str:
        if self._state == "open" and time.time() - self._opened_at >= self.reset_timeout:
            self._state = "half-open"
        return self._state

    def allow(self) -> bool:
        s = self.state
        return s in ("closed", "half-open")

    def record_success(self):
        if self._state != "closed":
            logger.info("[breaker:%s] closed (recovered)", self.name)
        self._failures = 0
        self._state = "closed"

    def record_failure(self):
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._state = "open"
            self._opened_at = time.time()
            logger.warning("[breaker:%s] OPEN after %d consecutive failures",
                           self.name, self._failures)

    def status(self) -> dict:
        return {"name": self.name, "state": self.state, "failures": self._failures}


class ResilientLLM:
    """Wraps a primary + optional fallback LLMBackend with circuit breakers.

    Usage:
        resilient = ResilientLLM(primary_backend, fallback_backend)
        text = await resilient.generate(prompt)
    """

    def __init__(self, primary, fallback=None):
        self.primary = primary
        self.fallback = fallback
        self._breakers = {
            "primary": CircuitBreaker(primary.__class__.__name__),
        }
        if fallback:
            self._breakers["fallback"] = CircuitBreaker(fallback.__class__.__name__)

    async def generate(self, prompt, **kwargs) -> str:
        # Try primary
        pb = self._breakers["primary"]
        if pb.allow():
            try:
                result = await self.primary.generate(prompt, **kwargs)
                pb.record_success()
                return result
            except Exception as e:
                pb.record_failure()
                logger.warning("[resilient] primary failed: %s", e)
                if not self.fallback:
                    raise

        # Fallback
        if self.fallback:
            fb = self._breakers["fallback"]
            if fb.allow():
                try:
                    logger.info("[resilient] falling back to %s",
                                self.fallback.__class__.__name__)
                    result = await self.fallback.generate(prompt, **kwargs)
                    fb.record_success()
                    return result
                except Exception:
                    fb.record_failure()
                    raise

        raise CircuitOpen("All providers unavailable — primary: %s" % pb.status())

    async def generate_stream(self, prompt, **kwargs):
        # Try primary
        pb = self._breakers["primary"]
        if pb.allow():
            try:
                async for token in self.primary.generate_stream(prompt, **kwargs):
                    yield token
                pb.record_success()
                return
            except Exception as e:
                pb.record_failure()
                logger.warning("[resilient] primary stream failed: %s", e)
                if not self.fallback:
                    raise

        # Fallback
        if self.fallback:
            fb = self._breakers["fallback"]
            if fb.allow():
                try:
                    async for token in self.fallback.generate_stream(prompt, **kwargs):
                        yield token
                    fb.record_success()
                    return
                except Exception:
                    fb.record_failure()
                    raise

        raise CircuitOpen("All providers unavailable")

    def status(self) -> dict:
        return {k: b.status() for k, b in self._breakers.items()}
