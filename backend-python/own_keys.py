"""Validate user-provided inference credentials without storing or logging them."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Literal

import httpx
from fastapi import HTTPException
from pydantic import BaseModel, Field


class OwnKeys(BaseModel):
    provider: Literal["huggingface", "cloudflare"]
    hfToken: str = Field(..., min_length=1, max_length=1024)
    cfAccountId: str = Field("", max_length=64)
    cfToken: str = Field("", max_length=1024)


@dataclass
class ProviderCredentials:
    provider: str
    hf_token: str
    cf_account_id: str = ""
    cf_token: str = ""


def provider_name() -> str:
    """Return the currently selected LLM provider."""
    return "cloudflare" if os.getenv("LLM_MODEL", "").strip().upper() == "CLOUDFLARE" else "huggingface"


def credentials_from_headers(headers) -> ProviderCredentials | None:
    """Read ephemeral credentials from a trusted internal pipeline request."""
    provider = headers.get("x-provider-name", "")
    hf = headers.get("x-provider-hf-key", "")
    account = headers.get("x-provider-cf-account", "")
    cf = headers.get("x-provider-cf-key", "")
    if not any((provider, hf, account, cf)):
        return None
    try:
        keys = OwnKeys(provider=provider, hfToken=hf, cfAccountId=account, cfToken=cf)
    except ValueError:
        raise HTTPException(status_code=400, detail="Complete API credentials are required") from None
    if keys.provider == "cloudflare" and (
        not re.fullmatch(r"[a-fA-F0-9]{32}", keys.cfAccountId) or not keys.cfToken
    ):
        raise HTTPException(status_code=400, detail="Hugging Face and Cloudflare credentials are required")
    return ProviderCredentials(keys.provider, keys.hfToken, keys.cfAccountId, keys.cfToken)


async def job_credential_headers(tenant: str, provider: str) -> dict[str, str]:
    """Resolve an account/provider reference at execution time; never queue secrets."""
    if provider == "free":
        return {}
    base_url = os.getenv("NODE_API_URL", "").rstrip("/")
    if not base_url:
        raise RuntimeError("NODE_API_URL is required for saved-provider background jobs")
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(
                f"{base_url}/internal/job-credentials",
                headers={"X-Internal-API-Key": os.environ["INTERNAL_API_KEY"]},
                json={"tenant": tenant, "provider": provider},
            )
        if response.status_code != 200:
            raise ValueError("Credential reference is unavailable")
        keys = OwnKeys.model_validate(response.json())
        if keys.provider != provider:
            raise ValueError("Saved provider changed")
        headers = {
            "x-provider-name": keys.provider, "x-provider-hf-key": keys.hfToken,
            "x-provider-cf-account": keys.cfAccountId, "x-provider-cf-key": keys.cfToken,
        }
        credentials_from_headers(headers)
        return headers
    except (httpx.HTTPError, ValueError, HTTPException):
        raise RuntimeError("Saved provider credentials unavailable. Update Settings and submit a new job.") from None


async def validate_keys(keys: OwnKeys) -> str:
    """Verify the HF token and selected provider's credentials."""
    if keys.provider == "cloudflare" and (
        not re.fullmatch(r"[a-fA-F0-9]{32}", keys.cfAccountId) or not keys.cfToken
    ):
        raise HTTPException(status_code=400, detail="Hugging Face and Cloudflare credentials are required")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            hf = await client.get(
                "https://huggingface.co/api/whoami-v2",
                headers={"Authorization": f"Bearer {keys.hfToken}"},
            )
            if hf.status_code != 200:
                raise HTTPException(status_code=400, detail="Hugging Face token is invalid")
            if keys.provider == "cloudflare":
                model = os.getenv("CF_LLM_8B_MODEL", "@cf/meta/llama-3.1-8b-instruct-fp8")
                cf = await client.post(
                    f"https://api.cloudflare.com/client/v4/accounts/{keys.cfAccountId}/ai/run/{model}",
                    headers={"Authorization": f"Bearer {keys.cfToken}"},
                    json={"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1},
                )
                if cf.status_code == 429:
                    raise HTTPException(status_code=429, detail="Cloudflare usage limit reached")
                if cf.status_code >= 500:
                    raise HTTPException(status_code=503, detail="Cloudflare verification is unavailable; try again later")
                if cf.status_code != 200:
                    raise HTTPException(status_code=400, detail="Cloudflare account, token or model access is invalid")
                try:
                    result = cf.json()
                except ValueError:
                    raise HTTPException(status_code=503, detail="Cloudflare returned an invalid verification response") from None
                if not isinstance(result, dict) or result.get("success") is not True:
                    raise HTTPException(status_code=400, detail="Cloudflare account, token or model access is invalid")
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="Provider verification is unavailable; try again later") from None
    return keys.provider
