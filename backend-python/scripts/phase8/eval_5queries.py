"""
Phase 8 Step 8.1 — 5-query eval set.
Run from the backend-python directory:
    python scripts/phase8/eval_5queries.py

Prerequisites:
  - Express running on localhost:4000
  - FastAPI running on localhost:8000

Creates sessions, sends questions, captures full responses,
and writes everything to eval_results.txt for review.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import httpx
import json
import time
from datetime import datetime

EXPRESS_URL = "http://localhost:4000"
TIMEOUT = 180.0  # pipeline can take a while on CPU

# --- Auth ---
AUTH_EMAIL = "eval@curalink.test"
AUTH_PASS = "eval123456"
AUTH_NAME = "Eval Bot"


def get_token():
    """Sign up or login to get a JWT token."""
    # Try login first
    resp = httpx.post(
        f"{EXPRESS_URL}/api/auth/login",
        json={"email": AUTH_EMAIL, "password": AUTH_PASS},
        timeout=10.0,
    )
    data = resp.json()
    if data.get("ok"):
        return data["token"]

    # Sign up
    resp = httpx.post(
        f"{EXPRESS_URL}/api/auth/signup",
        json={"name": AUTH_NAME, "email": AUTH_EMAIL, "password": AUTH_PASS},
        timeout=10.0,
    )
    data = resp.json()
    if data.get("ok"):
        return data["token"]

    raise RuntimeError(f"Auth failed: {data}")


def headers(token):
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }


# --- Eval Queries ---
EVAL_SET = [
    {
        "name": "Q1: Lung Cancer + Immunotherapy (New York)",
        "session": {
            "disease": "Lung Cancer",
            "intent": "Immunotherapy",
            "location": "New York, USA",
            "patientName": "Jane Doe",
        },
        "questions": [
            "How effective is immunotherapy for non-small cell lung cancer?",
            "Which checkpoint inhibitors work best for this?",
            "What happens when immunotherapy is combined with radiation?",
            "Are there trials recruiting near New York?",
            "What biomarkers predict a good response to immunotherapy?",
        ],
    },
    {
        "name": "Q2: Heart Disease + Statin Therapy (Chicago, Illinois)",
        "session": {
            "disease": "Coronary Artery Disease",
            "intent": "Statin therapy",
            "location": "Chicago, Illinois, USA",
            "patientName": "Michael Johnson",
        },
        "questions": [
            "How effective are statins at preventing heart attacks?",
            "What are the muscle-related side effects?",
            "Is there a link between statins and cognitive decline?",
            "Are there trials near Chicago for newer cholesterol drugs?",
            "What role does diet play alongside statin treatment?",
        ],
    },
    {
        "name": "Q3: Parkinson's + DBS (Boston, Massachusetts)",
        "session": {
            "disease": "Parkinson's disease",
            "intent": "Deep Brain Stimulation",
            "location": "Boston, Massachusetts, USA",
            "patientName": "John Smith",
        },
        "questions": [
            "What are the latest advances in DBS for Parkinson's?",
            "Can I take Vitamin D alongside DBS treatment?",
            "How does adaptive DBS compare to conventional stimulation?",
            "Are there DBS trials near Boston?",
            "What are the long term outcomes after DBS surgery?",
        ],
    },
    {
        "name": "Q4: Alzheimer's + Early Diagnosis (Los Angeles, California)",
        "session": {
            "disease": "Alzheimer's disease",
            "intent": "early diagnosis",
            "location": "Los Angeles, California, USA",
            "patientName": "Robert Williams",
        },
        "questions": [
            "What blood biomarkers can detect Alzheimer's before symptoms appear?",
            "How reliable is the p-tau217 test?",
            "What role does the APOE gene play in risk?",
            "Any early diagnosis studies near LA?",
            "How does amyloid PET compare to blood-based screening?",
        ],
    },
]


def create_session(token, form):
    resp = httpx.post(
        f"{EXPRESS_URL}/api/session",
        json=form,
        headers=headers(token),
        timeout=10.0,
    )
    data = resp.json()
    if not data.get("ok"):
        raise RuntimeError(f"Session create failed: {data}")
    return data["session"]["_id"]


def send_message(token, session_id, message, max_retries=2):
    """Send message via non-streaming endpoint and return full response.
    Retries on pipeline failure (likely HF rate limit)."""
    for attempt in range(max_retries + 1):
        resp = httpx.post(
            f"{EXPRESS_URL}/api/chat",
            json={"sessionId": session_id, "message": message},
            headers=headers(token),
            timeout=TIMEOUT,
        )
        data = resp.json()
        if data.get("ok"):
            return data.get("response", {})

        error = data.get("error", "unknown error")
        detail = data.get("detail", "")

        # Retry on pipeline/rate-limit errors
        if attempt < max_retries:
            print(f"    RETRY ({attempt+1}/{max_retries}): {error} — waiting 30s...")
            time.sleep(30)
            continue

        return {"error": error, "detail": detail}
    return {"error": "max retries exceeded"}


def format_response(resp, indent=4):
    """Format a pipeline response for the eval report."""
    lines = []
    prefix = " " * indent

    if "error" in resp:
        lines.append(f"{prefix}ERROR: {resp['error']}")
        if resp.get("detail"):
            lines.append(f"{prefix}DETAIL: {str(resp['detail'])[:500]}")
        return "\n".join(lines)

    # Overview
    overview = resp.get("overview", "")
    lines.append(f"{prefix}OVERVIEW: {overview}")

    # Abstain
    abstain = resp.get("abstain_reason")
    if abstain:
        lines.append(f"{prefix}ABSTAIN: {abstain}")
        return "\n".join(lines)

    # Insights
    insights = resp.get("insights", [])
    lines.append(f"{prefix}INSIGHTS ({len(insights)}):")
    for i, ins in enumerate(insights):
        finding = ins.get("finding", "")
        unverified = " [UNVERIFIED]" if ins.get("unverified") else ""
        sources = ins.get("source_details", [])
        lines.append(f"{prefix}  {i+1}. {finding}{unverified}")
        for src in sources:
            platform = src.get("platform", "?")
            year = src.get("year", "?")
            title = (src.get("title", "?"))[:80]
            lines.append(f"{prefix}     -> [{platform} {year}] {title}")
            snippet = src.get("snippet", "")
            if snippet:
                lines.append(f"{prefix}        Snippet: \"{snippet[:120]}...\"")

    # Trials
    trials = resp.get("trials", [])
    lines.append(f"{prefix}TRIALS ({len(trials)}):")
    for t in trials:
        nct = t.get("nct_id", "?")
        status = t.get("status", "?")
        title = (t.get("title", ""))[:70]
        location = t.get("location", "")
        relevance = t.get("relevance", "")
        lines.append(f"{prefix}  - {nct} | {status} | {title}")
        if location:
            lines.append(f"{prefix}    Location: {location}")
        if relevance:
            lines.append(f"{prefix}    Relevance: {relevance}")

    # Pipeline meta
    meta = resp.get("pipelineMeta", {})
    timings = meta.get("stage_timings_ms", {})
    counts = meta.get("retrieval_counts", {})
    citations = meta.get("citation_stats", {})
    warnings = meta.get("warnings", [])

    lines.append(f"{prefix}PIPELINE META:")
    lines.append(f"{prefix}  Timings: {timings}")
    lines.append(f"{prefix}  Retrieval: {counts}")
    lines.append(f"{prefix}  Citations: {citations}")
    if warnings:
        lines.append(f"{prefix}  Warnings: {warnings}")

    return "\n".join(lines)


def run_eval():
    output_file = Path(__file__).parent / "eval_results.txt"
    report = []

    report.append("=" * 70)
    report.append("CURALINK — 5-QUERY EVAL SET")
    report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 70)

    print("Authenticating...")
    token = get_token()
    print(f"  Token obtained\n")

    total_start = time.perf_counter()

    for qi, eval_case in enumerate(EVAL_SET):
        name = eval_case["name"]
        session_form = eval_case["session"]
        questions = eval_case["questions"]

        print(f"[{qi+1}/5] {name}")
        report.append(f"\n{'='*70}")
        report.append(f"EVAL {qi+1}: {name}")
        report.append(f"{'='*70}")
        report.append(f"  Disease: {session_form['disease']}")
        report.append(f"  Intent: {session_form.get('intent', '')}")
        report.append(f"  Location: {session_form.get('location', '')}")
        report.append(f"  Patient: {session_form.get('patientName', '')}")

        # Create session
        session_id = create_session(token, session_form)
        report.append(f"  Session ID: {session_id}")

        for qi2, question in enumerate(questions):
            print(f"  Q{qi2+1}: \"{question}\"")
            report.append(f"\n  --- Question {qi2+1}: \"{question}\" ---")

            t0 = time.perf_counter()
            resp = send_message(token, session_id, question)
            dt = time.perf_counter() - t0

            print(f"    Response in {dt:.1f}s")
            report.append(f"  Response time: {dt:.1f}s")
            report.append(format_response(resp))

            # Quality checks
            checks = []
            if "error" in resp:
                checks.append("FAIL: pipeline error")
            else:
                overview = resp.get("overview", "")
                insights = resp.get("insights", [])
                trials = resp.get("trials", [])
                abstain = resp.get("abstain_reason")
                citations = resp.get("pipelineMeta", {}).get("citation_stats", {})

                if abstain:
                    checks.append(f"WARN: abstained — {abstain}")
                if not overview:
                    checks.append("FAIL: no overview")
                elif len(overview) < 30:
                    checks.append("WARN: overview too short")
                else:
                    checks.append("PASS: overview present")

                if len(insights) >= 2:
                    checks.append(f"PASS: {len(insights)} insights")
                elif len(insights) == 1:
                    checks.append(f"WARN: only 1 insight")
                elif not abstain:
                    checks.append("FAIL: no insights")

                if len(trials) >= 1:
                    checks.append(f"PASS: {len(trials)} trials")
                else:
                    checks.append("INFO: no trials (may be expected)")

                verified = citations.get("verified", 0)
                unverified = citations.get("unverified", 0)
                if unverified > 0:
                    checks.append(f"WARN: {unverified} unverified citations")
                if verified > 0:
                    checks.append(f"PASS: {verified} verified citations")
                else:
                    checks.append("FAIL: no verified citations")

                # Check disease context is in the response
                disease_lower = session_form["disease"].lower()
                overview_lower = overview.lower()
                if any(word in overview_lower for word in disease_lower.split()):
                    checks.append("PASS: disease mentioned in overview")
                else:
                    checks.append("WARN: disease not mentioned in overview")

            report.append(f"\n  CHECKS:")
            for c in checks:
                report.append(f"    {c}")
                status = c.split(":")[0]
                if status in ("FAIL", "WARN"):
                    print(f"    {c}")

            # Sleep between questions within a session to avoid HF rate limits
            if qi2 < len(questions) - 1:
                print(f"    Sleeping 15s (rate limit)...")
                time.sleep(15)

        # Sleep between sessions to avoid HF rate limits
        if qi < len(EVAL_SET) - 1:
            print(f"  Sleeping 30s before next session (HF rate limit)...")
            time.sleep(30)

    total_time = time.perf_counter() - total_start

    report.append(f"\n{'='*70}")
    report.append(f"TOTAL TIME: {total_time:.1f}s")
    report.append(f"{'='*70}")

    # Write report
    report_text = "\n".join(report)
    output_file.write_text(report_text, encoding="utf-8")
    print(f"\nDone in {total_time:.1f}s. Report saved to: {output_file}")
    print(f"Review: {output_file}")


if __name__ == "__main__":
    run_eval()
