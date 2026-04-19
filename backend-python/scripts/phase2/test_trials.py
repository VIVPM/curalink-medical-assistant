"""
Standalone test script for Step 2.3.
Run from the backend-python directory:
    python scripts/test_trials.py

Tests two scenarios:
  1. "diabetes" with no location filter
  2. "diabetes" filtered to within 100 miles of Toronto, Canada

Manual verification:
  - Both return 20+ plausible trials with structured sections
  - Scenario 2 results are geographically near Toronto (not Tokyo)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio
from dotenv import load_dotenv

from sources.trials import fetch_trials

load_dotenv()

# Toronto coordinates (hardcoded for the test; Phase 2.4+ will geocode dynamically)
TORONTO = (43.6532, -79.3832)


def print_trial(i: int, t: dict) -> None:
    title = (t["title"] or "<no title>")[:80]
    status = t["status"] or "?"
    year = t["year"] or "?"
    brief = (t["brief_summary"] or "<no brief summary>")[:180]
    eligibility = (t["eligibility_criteria"] or "<no eligibility>")[:150]
    primary = ", ".join((t["primary_outcomes"] or [])[:2])[:120]
    loc_count = len(t["locations"] or [])
    first_loc = ""
    if t["locations"]:
        loc0 = t["locations"][0]
        first_loc = f"{loc0.get('city') or '?'}, {loc0.get('country') or '?'}"

    print(f"\n[{i}] {title}")
    print(f"    NCT: {t['nct_id']} | Status: {status} | Start: {year}")
    print(f"    Brief: {brief}...")
    print(f"    Eligibility: {eligibility}...")
    if primary:
        print(f"    Primary outcome: {primary}")
    if loc_count:
        print(f"    Locations: {loc_count} sites (first: {first_loc})")


async def run_case(name: str, **kwargs):
    print("\n" + "=" * 70)
    print(f"CASE: {name}")
    print("=" * 70)

    trials = await fetch_trials(**kwargs)
    print(f"Got {len(trials)} trials\n")

    if not trials:
        print("No results.")
        return

    for i, t in enumerate(trials[:5], 1):  # print first 5 only to keep output short
        print_trial(i, t)
    if len(trials) > 5:
        print(f"\n... ({len(trials) - 5} more trials not shown)")

    with_brief = sum(1 for t in trials if t["brief_summary"])
    with_eligibility = sum(1 for t in trials if t["eligibility_criteria"])
    with_locations = sum(1 for t in trials if t["locations"])
    statuses: dict[str, int] = {}
    for t in trials:
        s = t["status"] or "unknown"
        statuses[s] = statuses.get(s, 0) + 1

    print("\n" + "-" * 70)
    print(f"Summary: {with_brief}/{len(trials)} have brief summaries, "
          f"{with_eligibility}/{len(trials)} have eligibility criteria, "
          f"{with_locations}/{len(trials)} have locations")
    print(f"Status breakdown: {statuses}")


async def main():
    # Case 1: no location filter
    await run_case(
        "diabetes, no location filter",
        disease="diabetes",
        limit=20,
    )

    # Case 2: near Toronto
    await run_case(
        "diabetes, within 100mi of Toronto",
        disease="diabetes",
        location=TORONTO,
        radius_miles=100,
        limit=20,
    )


if __name__ == "__main__":
    asyncio.run(main())
