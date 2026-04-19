"""
Test for Phase 6 Step 6.5 — SSE streaming end-to-end.
Run from the backend-python directory:
    python scripts/phase6/test_sse_stream.py

Prerequisites:
  - FastAPI running on localhost:8000
  - Express running on localhost:4000

Tests both FastAPI direct streaming and Express SSE proxy.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import httpx
import json
import time

FASTAPI_URL = "http://localhost:8000"
EXPRESS_URL = "http://localhost:4000"


def test_fastapi_stream():
    """Test FastAPI /pipeline/stream directly."""
    print("\n--- Test 1: FastAPI /pipeline/stream (direct) ---")

    body = {
        "static": {
            "disease": "Parkinson's disease",
            "intent": "DBS",
            "location": "Toronto",
            "patientName": "John",
        },
        "current": {"userMessage": "What are the latest treatments?"},
    }

    events = {"status": [], "token": [], "metadata": None, "done": False}
    token_text = ""

    t0 = time.perf_counter()
    first_token_time = None

    try:
        with httpx.stream(
            "POST",
            f"{FASTAPI_URL}/pipeline/stream",
            json=body,
            timeout=120.0,
        ) as resp:
            if resp.status_code != 200:
                print(f"  FAIL: status {resp.status_code}")
                return False

            current_event = None
            for line in resp.iter_lines():
                if line.startswith("event: "):
                    current_event = line[7:]
                elif line.startswith("data: ") and current_event:
                    data = line[6:]
                    if current_event == "status":
                        try:
                            events["status"].append(json.loads(data))
                        except json.JSONDecodeError:
                            events["status"].append(data)
                    elif current_event == "token":
                        if first_token_time is None:
                            first_token_time = time.perf_counter() - t0
                        try:
                            token_text += json.loads(data)
                        except json.JSONDecodeError:
                            token_text += data
                        events["token"].append(data)
                    elif current_event == "metadata":
                        try:
                            events["metadata"] = json.loads(data)
                        except json.JSONDecodeError:
                            events["metadata"] = data
                    elif current_event == "done":
                        events["done"] = True
                    current_event = None
    except Exception as e:
        print(f"  FAIL: connection error: {e}")
        return False

    total_time = time.perf_counter() - t0

    # Report
    print(f"  Total time: {total_time:.1f}s")
    if first_token_time:
        print(f"  First token: {first_token_time:.1f}s")

    print(f"  Status events: {len(events['status'])}")
    for s in events["status"]:
        if isinstance(s, dict):
            print(f"    - {s.get('stage', '?')}: {s.get('message', '')}")

    print(f"  Token events: {len(events['token'])}")
    if token_text:
        print(f"  Token preview: {token_text[:100]}...")

    # Checks
    all_pass = True

    if len(events["status"]) >= 4:
        print("  PASS: received status events for pipeline stages")
    else:
        print(f"  FAIL: expected >=4 status events, got {len(events['status'])}")
        all_pass = False

    if len(events["token"]) > 0:
        print(f"  PASS: received {len(events['token'])} token events (LLM streaming works)")
    else:
        print("  FAIL: no token events received")
        all_pass = False

    if events["metadata"]:
        meta = events["metadata"]
        has_overview = bool(meta.get("overview"))
        has_insights = isinstance(meta.get("insights"), list)
        has_pipeline = bool(meta.get("pipelineMeta"))
        print(f"  PASS: metadata received (overview={has_overview}, insights={has_insights}, pipelineMeta={has_pipeline})")
        if not (has_overview and has_insights and has_pipeline):
            print("  WARN: metadata may be incomplete")
    else:
        print("  FAIL: no metadata event received")
        all_pass = False

    if events["done"]:
        print("  PASS: received done event")
    else:
        print("  FAIL: no done event")
        all_pass = False

    return all_pass


def test_express_stream():
    """Test Express /api/chat/stream (SSE proxy)."""
    print("\n--- Test 2: Express /api/chat/stream (SSE proxy) ---")

    # First create a session
    try:
        resp = httpx.post(
            f"{EXPRESS_URL}/api/session",
            json={
                "disease": "Parkinson's disease",
                "intent": "DBS",
                "location": "Toronto",
                "patientName": "Test",
            },
            timeout=10.0,
        )
        session_id = resp.json()["session"]["_id"]
        print(f"  Created session: {session_id}")
    except Exception as e:
        print(f"  FAIL: could not create session: {e}")
        return False

    body = {"sessionId": session_id, "message": "Latest treatments?"}
    events_count = {"status": 0, "token": 0, "metadata": False, "done": False}

    t0 = time.perf_counter()

    try:
        with httpx.stream(
            "POST",
            f"{EXPRESS_URL}/api/chat/stream",
            json=body,
            timeout=120.0,
        ) as resp:
            if resp.status_code != 200:
                text = resp.read().decode()
                print(f"  FAIL: status {resp.status_code}: {text[:200]}")
                return False

            current_event = None
            for line in resp.iter_lines():
                if line.startswith("event: "):
                    current_event = line[7:]
                elif line.startswith("data: ") and current_event:
                    if current_event == "status":
                        events_count["status"] += 1
                    elif current_event == "token":
                        events_count["token"] += 1
                    elif current_event == "metadata":
                        events_count["metadata"] = True
                    elif current_event == "done":
                        events_count["done"] = True
                    current_event = None
    except Exception as e:
        print(f"  FAIL: connection error: {e}")
        return False

    total_time = time.perf_counter() - t0
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Status events: {events_count['status']}, Token events: {events_count['token']}")

    all_pass = True

    if events_count["status"] >= 4:
        print("  PASS: status events proxied through Express")
    else:
        print(f"  WARN: expected >=4 status events, got {events_count['status']}")

    if events_count["token"] > 0:
        print(f"  PASS: {events_count['token']} token events proxied")
    else:
        print("  FAIL: no token events")
        all_pass = False

    if events_count["metadata"]:
        print("  PASS: metadata event received")
    else:
        print("  FAIL: no metadata event")
        all_pass = False

    if events_count["done"]:
        print("  PASS: done event received")
    else:
        print("  WARN: no done event (stream may have closed early)")

    # Verify Mongo persistence
    try:
        resp = httpx.get(f"{EXPRESS_URL}/api/session/{session_id}", timeout=10.0)
        messages = resp.json().get("messages", [])
        roles = [m["role"] for m in messages]
        if "user" in roles and "assistant" in roles:
            print(f"  PASS: messages saved to Mongo ({len(messages)} messages)")
        else:
            print(f"  WARN: messages in Mongo: {roles}")
    except Exception as e:
        print(f"  WARN: could not verify Mongo: {e}")

    return all_pass


def main():
    print("=" * 60)
    print("Phase 6 Step 6.5: SSE Streaming End-to-End Test")
    print("=" * 60)

    pass1 = test_fastapi_stream()
    pass2 = test_express_stream()

    print("\n" + "=" * 60)
    if pass1 and pass2:
        print("All tests passed. SSE streaming works end-to-end.")
    elif pass1:
        print("FastAPI streaming works. Express proxy had issues.")
    else:
        print("Some tests failed — check output above.")


if __name__ == "__main__":
    main()
