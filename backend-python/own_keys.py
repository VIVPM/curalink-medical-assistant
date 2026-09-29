"""Validate user-provided inference credentials without storing or logging them."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import httpx
from fastapi import HTTPException
from pydantic import BaseModel, Field


class OwnKeys(BaseModel):
    hfToken: str = Field(..., min_length=1, max_length=1024)
    cfAccountId: str = Field("", max_length=64)
    cfToken: str = Field("", max_length=1024)


@dataclass
class ProviderCredentials:
    hf_token: str
    cf_account_id: str = ""
    cf_token: str = ""


def provider_name() -> str:
    """Return the currently selected LLM provider."""
    return "cloudflare" if os.getenv("LLM_MODEL", "").strip().upper() == "CLOUDFLARE" else "huggingface"


def credentials_from_headers(headers) -> ProviderCredentials | None:
    """Read ephemeral credentials from a trusted internal pipeline request."""
    hf = headers.get("x-provider-hf-key", "")
    account = headers.get("x-provider-cf-account", "")
    cf = headers.get("x-provider-cf-key", "")
    if not any((hf, account, cf)):
        return None
    try:
        keys = OwnKeys(hfToken=hf, cfAccountId=account, cfToken=cf)
    except ValueError:
        raise HTTPException(status_code=400, detail="Complete API credentials are required") from None
    if provider_name() == "cloudflare" and (
        not re.fullmatch(r"[a-fA-F0-9]{32}", keys.cfAccountId) or not keys.cfToken
    ):
        raise HTTPException(status_code=400, detail="Hugging Face and Cloudflare credentials are required")
    return ProviderCredentials(keys.hfToken, keys.cfAccountId, keys.cfToken)


async def validate_keys(keys: OwnKeys) -> str:
    """Verify the HF token and, when active, the Cloudflare account token."""
    provider = provider_name()
    if provider == "cloudflare" and (not keys.cfAccountId or not keys.cfToken):
        raise HTTPException(status_code=400, detail="Hugging Face and Cloudflare credentials are required")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            hf = await client.get(
                "https://huggingface.co/api/whoami-v2",
                headers={"Authorization": f"Bearer {keys.hfToken}"},
            )
            if hf.status_code != 200:
                raise HTTPException(status_code=400, detail="Hugging Face token is invalid")
            if provider == "cloudflare":
                cf = await client.get(
                    f"https://api.cloudflare.com/client/v4/accounts/{keys.cfAccountId}/tokens/verify",
                    headers={"Authorization": f"Bearer {keys.cfToken}"},
                )
                if cf.status_code != 200 or not cf.json().get("success") or cf.json().get("result", {}).get("status") != "active":
                    raise HTTPException(status_code=400, detail="Cloudflare account or token is invalid")
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="Provider verification is unavailable; try again later") from None
    return provider
