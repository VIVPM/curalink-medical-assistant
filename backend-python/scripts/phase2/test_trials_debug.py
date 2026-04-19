"""
Debug script for the ClinicalTrials 403 issue.

Reproduces the EXACT request twice - once through httpx (failing), once
through the same URL via subprocess+curl (known working) - so we can see
what's different.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio
import subprocess
from urllib.parse import urlencode

import httpx

BASE = "https://clinicaltrials.gov/api/v2/studies"

# Simple test: exactly what the working curl used
SIMPLE_PARAMS = {
    "query.cond": "diabetes",
    "pageSize": "5",
    "format": "json",
}


async def try_httpx(params: dict, label: str) -> None:
    print(f"\n--- httpx: {label} ---")
    url = f"{BASE}?{urlencode(params)}"
    print(f"URL: {url}")
    try:
        async with httpx.AsyncClient(
            timeout=10.0,
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
        ) as client:
            resp = await client.get(BASE, params=params)
            print(f"Status: {resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                print(f"Got {len(data.get('studies', []))} studies")
            else:
                print(f"Body: {resp.text[:300]}")
    except Exception as e:
        print(f"Exception: {type(e).__name__}: {e}")


def try_curl(params: dict, label: str) -> None:
    print(f"\n--- curl: {label} ---")
    url = f"{BASE}?{urlencode(params)}"
    print(f"URL: {url}")
    try:
        result = subprocess.run(
            [
                "curl.exe",
                "-sS",
                "-H",
                "User-Agent: Mozilla/5.0",
                "-o",
                "NUL",
                "-w",
                "HTTP %{http_code} | size %{size_download}",
                url,
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        print(f"curl stdout: {result.stdout}")
        if result.stderr:
            print(f"curl stderr: {result.stderr}")
    except Exception as e:
        print(f"curl failed: {e}")


async def main():
    # Test 1: no filter (the known-working shape)
    await try_httpx(SIMPLE_PARAMS, "simple, no filter")
    try_curl(SIMPLE_PARAMS, "simple, no filter")

    # Test 2: with status filter (the failing shape)
    params_with_filter = {
        **SIMPLE_PARAMS,
        "filter.overallStatus": "RECRUITING,ACTIVE_NOT_RECRUITING,COMPLETED",
    }
    await try_httpx(params_with_filter, "with status filter")
    try_curl(params_with_filter, "with status filter")


if __name__ == "__main__":
    asyncio.run(main())
