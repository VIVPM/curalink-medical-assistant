"""
Geocode a location string to (lat, lon) using Nominatim (OpenStreetMap).
Free, no API key. Cached in-memory to avoid repeated calls for the same location.
"""

from __future__ import annotations

import httpx

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_cache: dict[str, tuple[float, float] | None] = {}


async def geocode(location: str) -> tuple[float, float] | None:
    """
    Convert a location string to (lat, lon).
    Returns None if geocoding fails or location is empty.
    Caches results in memory.
    """
    if not location or not location.strip():
        return None

    key = location.strip().lower()
    if key in _cache:
        return _cache[key]

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                NOMINATIM_URL,
                params={
                    "q": location,
                    "format": "json",
                    "limit": "1",
                },
                headers={"User-Agent": "curalink (medical-research-assistant)"},
            )
            resp.raise_for_status()
            results = resp.json()

            if results:
                lat = float(results[0]["lat"])
                lon = float(results[0]["lon"])
                _cache[key] = (lat, lon)
                return (lat, lon)

    except Exception:
        pass

    _cache[key] = None
    return None
