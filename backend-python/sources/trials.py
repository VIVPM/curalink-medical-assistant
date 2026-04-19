"""
ClinicalTrials.gov v2 fetcher.

No API key, no auth, no formal rate limit. The only complexity is the
deeply nested `protocolSection` response structure.

Location filtering uses `filter.geo=distance(lat,lon,100mi)`. Geocoding
happens once per session upstream (Nominatim); this fetcher just takes
(lat, lon) pre-resolved.
"""

from __future__ import annotations

import asyncio
from typing import Any

from curl_cffi import requests as curl_requests

TRIALS_URL = "https://clinicaltrials.gov/api/v2/studies"
DEFAULT_TIMEOUT = 10.0
MAX_RETRIES = 2

# ClinicalTrials.gov is behind Cloudflare which blocks Python's native HTTP
# client TLS fingerprints. curl_cffi impersonates a real Chrome TLS handshake
# so the WAF lets us through.
BROWSER_IMPERSONATE = "chrome120"

# Intent-driven status filter. Set by Stage 1's query expander output.
# ClinicalTrials.gov v2 API expects comma-separated values here (NOT pipe-separated).
STATUS_FILTERS: dict[str, str] = {
    "clinical_trials_search": "RECRUITING,ACTIVE_NOT_RECRUITING,ENROLLING_BY_INVITATION",
    "treatment_overview": "RECRUITING,ACTIVE_NOT_RECRUITING,COMPLETED",
    "default": "RECRUITING,ACTIVE_NOT_RECRUITING,COMPLETED",
}



def _get(d: Any, path: str, default: Any = None) -> Any:
    """Safe nested dict lookup via dotted path, e.g. 'a.b.c'."""
    cur: Any = d
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
            if cur is None:
                return default
        else:
            return default
    return cur if cur is not None else default


def _parse_outcomes(outcomes_list: list[dict] | None) -> list[str]:
    if not outcomes_list:
        return []
    return [o.get("measure") for o in outcomes_list if o.get("measure")]


def _parse_locations(locations_list: list[dict] | None, cap: int = 10) -> list[dict]:
    """Extract structured locations, capped to avoid blowing up response size."""
    if not locations_list:
        return []
    result: list[dict] = []
    for loc in locations_list[:cap]:
        geo = loc.get("geoPoint") or {}
        result.append(
            {
                "facility": loc.get("facility"),
                "city": loc.get("city"),
                "state": loc.get("state"),
                "country": loc.get("country"),
                "status": loc.get("status"),
                "lat": geo.get("lat"),
                "lon": geo.get("lon"),
            }
        )
    return result


def _parse_contacts(contacts_list: list[dict] | None) -> list[dict]:
    if not contacts_list:
        return []
    return [
        {
            "name": c.get("name"),
            "phone": c.get("phone"),
            "email": c.get("email"),
            "role": c.get("role"),
        }
        for c in contacts_list
    ]


def _parse_year(start_date_struct: dict | None) -> int | None:
    if not start_date_struct:
        return None
    date_str = start_date_struct.get("date")
    if not date_str:
        return None
    try:
        return int(date_str.split("-")[0])
    except (ValueError, IndexError):
        return None


def _parse_study(study: dict) -> dict:
    proto = study.get("protocolSection") or {}
    nct_id = _get(proto, "identificationModule.nctId")
    start_struct = _get(proto, "statusModule.startDateStruct")
    start_date = start_struct.get("date") if start_struct else None

    return {
        "nct_id": nct_id,
        "title": _get(proto, "identificationModule.briefTitle"),
        "status": _get(proto, "statusModule.overallStatus"),
        "brief_summary": _get(proto, "descriptionModule.briefSummary"),
        "detailed_description": _get(proto, "descriptionModule.detailedDescription"),
        "eligibility_criteria": _get(proto, "eligibilityModule.eligibilityCriteria"),
        "min_age": _get(proto, "eligibilityModule.minimumAge"),
        "max_age": _get(proto, "eligibilityModule.maximumAge"),
        "start_date": start_date,
        "primary_outcomes": _parse_outcomes(
            _get(proto, "outcomesModule.primaryOutcomes")
        ),
        "secondary_outcomes": _parse_outcomes(
            _get(proto, "outcomesModule.secondaryOutcomes")
        ),
        "locations": _parse_locations(
            _get(proto, "contactsLocationsModule.locations")
        ),
        "contacts": _parse_contacts(
            _get(proto, "contactsLocationsModule.centralContacts")
        ),
        "year": _parse_year(start_struct),
        "url": f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else None,
    }


async def fetch_trials(
    disease: str,
    location: tuple[float, float] | None = None,
    radius_miles: int = 100,
    limit: int = 40,
    intent: str = "default",
    extra_terms: str | None = None,
) -> list[dict]:
    """
    Fetch clinical trials matching a disease from ClinicalTrials.gov v2.

    Args:
        disease: primary condition, goes into query.cond (e.g. "diabetes").
        location: optional (lat, lon) tuple. None skips the geo filter.
        radius_miles: search radius when `location` is given. Default 100mi.
        limit: max studies to return. Default 40.
        intent: "clinical_trials_search" | "treatment_overview" | "default".
        extra_terms: optional free-text keywords added to query.term.
    """
    status_filter = STATUS_FILTERS.get(intent, STATUS_FILTERS["default"])

    params: dict[str, str] = {
        "query.cond": disease,
        "filter.overallStatus": status_filter,
        "pageSize": str(limit),
        "format": "json",
    }
    if extra_terms:
        params["query.term"] = extra_terms
    if location is not None:
        lat, lon = location
        params["filter.geo"] = f"distance({lat},{lon},{radius_miles}mi)"

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            # curl_cffi's async API mirrors requests. impersonate="chrome120"
            # makes the TLS fingerprint look like real Chrome, bypassing
            # Cloudflare bot detection that blocks httpx/requests.
            resp = await asyncio.to_thread(
                curl_requests.get,
                TRIALS_URL,
                params=params,
                impersonate=BROWSER_IMPERSONATE,
            )
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"trials HTTP {resp.status_code}: {resp.text[:200]}"
                )
            data = resp.json()
            studies = data.get("studies", [])
            return [_parse_study(s) for s in studies]
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                await asyncio.sleep(0.5 * (attempt + 1))
                continue
            raise

    raise last_error if last_error else RuntimeError("trials fetch failed")
