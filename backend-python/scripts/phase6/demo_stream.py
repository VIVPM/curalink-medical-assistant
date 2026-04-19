"""
Live SSE streaming demo — watch tokens arrive in real-time.
Run from the backend-python directory:
    python scripts/phase6/demo_stream.py

Shows pipeline stages and LLM tokens printing live as they arrive.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import httpx
import json
import time

FASTAPI_URL = "http://localhost:8000"


def main():
    body = {
        "static": {
            "disease": "Parkinson's disease",
            "intent": "Deep Brain Stimulation",
            "location": "Toronto, Canada",
            "patientName": "John Smith",
        },
        "current": {"userMessage": "Can I take Vitamin D?"},
    }

    print("=" * 60)
    print("  LIVE STREAMING DEMO — Curalink Pipeline")
    print("=" * 60)
    print()

    t0 = time.perf_counter()
    token_count = 0

    with httpx.stream(
        "POST",
        f"{FASTAPI_URL}/pipeline/stream",
        json=body,
        timeout=120.0,
    ) as resp:
        current_event = None

        for line in resp.iter_lines():
            if line.startswith("event: "):
                current_event = line[7:]
            elif line.startswith("data: ") and current_event:
                data = line[6:]

                if current_event == "status":
                    info = json.loads(data)
                    elapsed = time.perf_counter() - t0
                    print(f"\n  [{elapsed:5.1f}s] Stage: {info.get('stage', '?')} — {info.get('message', '')}")

                elif current_event == "token":
                    if token_count == 0:
                        elapsed = time.perf_counter() - t0
                        print(f"\n  [{elapsed:5.1f}s] LLM generating:\n")
                        print("  ", end="", flush=True)
                    token = json.loads(data)
                    print(token, end="", flush=True)
                    token_count += 1

                elif current_event == "metadata":
                    elapsed = time.perf_counter() - t0
                    meta = json.loads(data)
                    print(f"\n\n  [{elapsed:5.1f}s] === RESPONSE ASSEMBLED ===")
                    print(f"\n  Overview: {meta.get('overview', '')[:200]}")
                    insights = meta.get("insights", [])
                    print(f"\n  Insights ({len(insights)}):")
                    for i, ins in enumerate(insights):
                        finding = ins.get("finding", "")[:100]
                        sources = [s.get("title", "?")[:50] for s in ins.get("source_details", [])]
                        verified = "unverified" if ins.get("unverified") else "verified"
                        print(f"    {i+1}. [{verified}] {finding}")
                        for s in sources:
                            print(f"       -> {s}")

                    trials = meta.get("trials", [])
                    print(f"\n  Trials ({len(trials)}):")
                    for t in trials:
                        print(f"    - {t.get('nct_id', '?')} | {t.get('status', '?')} | {t.get('title', '')[:60]}")

                    pm = meta.get("pipelineMeta", {})
                    timings = pm.get("stage_timings_ms", {})
                    print(f"\n  Timings: {timings}")
                    citations = pm.get("citation_stats", {})
                    print(f"  Citations: {citations}")

                elif current_event == "done":
                    elapsed = time.perf_counter() - t0
                    print(f"\n  [{elapsed:5.1f}s] Done. ({token_count} tokens streamed)")

                current_event = None

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
