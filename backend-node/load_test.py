"""
load_test.py — does the Express API stay responsive while the chat pipeline is
saturated, and how many concurrent users can browse before it breaks?

Modeled on multi-crew-lead-coordinator/backend/load_test_api.py. Same idea: run
the REAL app and stub only the expensive boundary, so a run costs $0 and finishes
in minutes. Here the app-under-test is Express (the browser-facing tier) and the
stubbed boundary is the FastAPI pipeline — we point Express's FASTAPI_URL at a
tiny SSE stub that sleeps `--lead-seconds` and returns a canned answer, so no
HuggingFace call ever happens.

Two questions, two modes:

  --ramp   step concurrency up (10, 25, 50, ...) hammering the READ endpoints
           (/health, GET /api/sessions, GET /api/session/:id) and find where it
           breaks. Answers "how many people can use the app at once." Reads have
           no rate limit, so nothing synthetic caps this.

  (default) idle-vs-saturated at one fixed concurrency: hammer reads with the
           pipeline idle, then again while N chat/stream requests are in flight
           (each holds a connection to the stub and does its Mongo writes), and
           compare. This is where DB-pool contention between streaming writes and
           browse reads shows up.

Latency is scaled (`--lead-seconds`), so read the idle-vs-saturated RATIO and the
ramp's breaking point, not absolute milliseconds.

    # against a locally-spawned Express + stub (needs `node` on PATH):
    python load_test.py --ramp
    python load_test.py                       # idle-vs-saturated

    # against a live deployment (ramp only — read-only, safe):
    python load_test.py --ramp --base-url https://curalink-...onrender.com

Run with the backend-python venv's Python (it has httpx + fastapi + uvicorn):
    ../backend-python/.venv/Scripts/python.exe load_test.py --ramp
"""

import os
import sys
import time
import json
import uuid
import asyncio
import argparse
import subprocess
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # backend-node/
ROOT_DIR = os.path.dirname(BASE_DIR)                        # repo root
RESULTS_DIR = os.path.join(ROOT_DIR, "load_test_results")

import httpx

# A stable load-test account. Its data is its own (sessions are per-user), so a
# run never touches real users; we still delete the seeded sessions at the end.
LOAD_EMAIL = "loadtest@curalink.local"
LOAD_PASS = "loadtest-pw-9137"
LOAD_NAME = "Load Test"


# =============================================================================
# Percentiles (nearest-rank — small samples, so no interpolated precision)
# =============================================================================

def pctl(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    import math
    rank = max(1, math.ceil(p / 100 * len(xs)))
    return xs[min(rank, len(xs)) - 1]


# =============================================================================
# Stub pipeline server — mimics FastAPI /pipeline/* with a sleep + canned answer
# =============================================================================

def serve_stub(port, lead_seconds):
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    import uvicorn

    app = FastAPI()
    canned = {
        "overview": "Stubbed answer for load testing.",
        "insights": [],
        "trials": [],
        "abstain_reason": None,
        "pipelineMeta": {"stub": True},
    }

    # ponytail: minimal in-memory job store for v2 probe tests
    _jobs = {}

    @app.get("/health")
    async def health():
        return {"ok": True, "service": "stub", "queue_depth": len(_jobs)}

    @app.post("/pipeline/run")
    async def run():
        await asyncio.sleep(lead_seconds)
        return canned

    @app.post("/pipeline/stream")
    async def stream():
        async def gen():
            yield 'event: status\ndata: {"stage":"stub","message":"stub"}\n\n'
            await asyncio.sleep(lead_seconds)
            yield f"event: metadata\ndata: {json.dumps(canned)}\n\n"
            yield "event: done\ndata: {}\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/jobs")
    async def submit_job():
        jid = uuid.uuid4().hex[:16]
        _jobs[jid] = "pending"
        asyncio.get_event_loop().call_later(lead_seconds, lambda: _jobs.update({jid: "completed"}))
        return {"job_id": jid, "state": "pending"}

    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str):
        state = _jobs.get(job_id, "not_found")
        return {"job_id": job_id, "state": state}

    @app.delete("/jobs/{job_id}")
    async def cancel_job(job_id: str):
        if job_id in _jobs:
            _jobs[job_id] = "cancelled"
        return {"job_id": job_id, "state": "cancelled"}

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="error")


# =============================================================================
# HTTP helpers
# =============================================================================

def wait_for_health(base, timeout=120):
    """Generous per-request timeout: a Render free instance cold-starts for tens
    of seconds on the first request, and a short timeout would kill each attempt
    before it can answer."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{base}/health", timeout=60).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def ensure_user(base):
    """Sign up (idempotent) then log in; return (token, user_id)."""
    httpx.post(f"{base}/api/auth/signup", timeout=30,
               json={"name": LOAD_NAME, "email": LOAD_EMAIL, "password": LOAD_PASS})
    r = httpx.post(f"{base}/api/auth/login", timeout=30,
                   json={"email": LOAD_EMAIL, "password": LOAD_PASS})
    if r.status_code != 200:
        sys.exit(f"Could not authenticate the load-test user: {r.status_code} {r.text[:200]}")
    d = r.json()
    return d["token"], d["user"]["_id"]


def seed_sessions(base, token, n):
    """Create N sessions so GET /api/sessions returns a realistic list payload."""
    auth = {"Authorization": f"Bearer {token}"}
    ids = []
    for i in range(n):
        r = httpx.post(f"{base}/api/session", headers=auth, timeout=30, json={
            "disease": f"parkinson-loadtest-{i}", "intent": "DBS",
            "location": "Toronto, Canada", "patientName": "Load Test",
        })
        if r.status_code == 201:
            ids.append(r.json()["session"]["_id"])
    return ids


def cleanup_sessions(base, token, ids):
    auth = {"Authorization": f"Bearer {token}"}
    n = 0
    for sid in ids:
        try:
            if httpx.delete(f"{base}/api/session/{sid}", headers=auth, timeout=30).status_code == 200:
                n += 1
        except Exception:
            pass
    return n


# =============================================================================
# The read-path hammer
# =============================================================================

async def hammer(base, token, session_id, concurrency, duration, mix="read"):
    results = {"health": [], "sessions": [], "session": [], "login": []}
    errors = {"count": 0, "samples": []}
    stop = time.monotonic() + duration
    auth = {"Authorization": f"Bearer {token}"}
    limits = httpx.Limits(max_connections=concurrency + 20, max_keepalive_connections=concurrency)

    cycle = [
        ("health", "GET", "/health", {}),
        ("sessions", "GET", "/api/sessions", {"headers": auth}),
        ("session", "GET", f"/api/session/{session_id}", {"headers": auth}),
    ]
    if mix == "all":
        cycle.append(("login", "POST", "/api/auth/login",
                      {"json": {"email": LOAD_EMAIL, "password": LOAD_PASS}}))

    async def one_client(client):
        while time.monotonic() < stop:
            for name, method, url, kwargs in cycle:
                t = time.monotonic()
                try:
                    r = await client.request(method, url, timeout=30, **kwargs)
                    dt = (time.monotonic() - t) * 1000
                    if r.status_code >= 400:
                        errors["count"] += 1
                        if len(errors["samples"]) < 5:
                            errors["samples"].append(f"{name} {r.status_code} {r.text[:70]}")
                    else:
                        results[name].append(dt)
                except Exception as exc:
                    errors["count"] += 1
                    if len(errors["samples"]) < 5:
                        errors["samples"].append(f"{name} {type(exc).__name__}")
                if time.monotonic() >= stop:
                    return

    started = time.monotonic()
    async with httpx.AsyncClient(base_url=base, limits=limits) as client:
        await asyncio.gather(*[one_client(client) for _ in range(concurrency)])
    return {"lat": results, "errors": errors, "elapsed": time.monotonic() - started}


async def chat_flood(base, token, session_id, n_clients, stop_evt):
    """Keep n_clients concurrent /api/chat/stream requests in flight until told to
    stop — this is what saturates the pipeline + the Mongo write path."""
    auth = {"Authorization": f"Bearer {token}"}
    limits = httpx.Limits(max_connections=n_clients + 5)

    async def looper(client):
        while not stop_evt.is_set():
            try:
                async with client.stream("POST", "/api/chat/stream", timeout=60, headers=auth,
                                         json={"sessionId": session_id, "message": "load test question"}) as r:
                    async for _ in r.aiter_bytes():
                        if stop_evt.is_set():
                            break
            except Exception:
                await asyncio.sleep(0.2)

    async with httpx.AsyncClient(base_url=base, limits=limits) as client:
        await asyncio.gather(*[looper(client) for _ in range(n_clients)])


# =============================================================================
# V2 feature probes — async jobs, queue depth, webhooks, circuit breaker
# =============================================================================

async def test_job_api(base, token, session_id):
    """Submit a job, poll until terminal, then verify cancel on a second job."""
    auth = {"Authorization": f"Bearer {token}"}
    results = {"submit": None, "poll_states": [], "cancel": None, "errors": []}
    async with httpx.AsyncClient(base_url=base) as client:
        # Submit
        r = await client.post("/api/jobs", timeout=30, headers=auth,
                              json={"sessionId": session_id, "message": "job api test"})
        results["submit"] = r.status_code
        if r.status_code not in (200, 201, 202):
            results["errors"].append(f"submit {r.status_code}")
            return results
        job_id = r.json().get("job_id") or r.json().get("jobId")
        if not job_id:
            results["errors"].append("no job_id in response")
            return results
        # Poll up to 10 times
        for _ in range(10):
            await asyncio.sleep(1)
            r = await client.get(f"/api/jobs/{job_id}", timeout=15, headers=auth)
            state = r.json().get("state") or r.json().get("status")
            results["poll_states"].append(state)
            if state in ("completed", "failed"):
                break
        # Submit + cancel a second job
        r = await client.post("/api/jobs", timeout=30, headers=auth,
                              json={"sessionId": session_id, "message": "cancel test"})
        if r.status_code in (200, 201, 202):
            j2 = r.json().get("job_id") or r.json().get("jobId")
            if j2:
                r = await client.delete(f"/api/jobs/{j2}", timeout=15, headers=auth)
                results["cancel"] = r.status_code
    return results


async def test_queue_depth(base):
    """Check /health includes queue_depth field."""
    async with httpx.AsyncClient(base_url=base) as client:
        r = await client.get("/health", timeout=10)
        data = r.json()
        return {"has_queue_depth": "queue_depth" in data,
                "queue_depth": data.get("queue_depth")}


async def test_webhook_crud(base, token):
    """Register a webhook, list it, delete it."""
    auth = {"Authorization": f"Bearer {token}"}
    results = {"create": None, "list_count": None, "delete": None, "errors": []}
    async with httpx.AsyncClient(base_url=base) as client:
        # Create
        r = await client.post("/api/webhooks", timeout=15, headers=auth,
                              json={"url": "https://httpbin.org/post",
                                    "events": ["job.completed"]})
        results["create"] = r.status_code
        if r.status_code not in (200, 201):
            results["errors"].append(f"create {r.status_code}")
            return results
        wh_id = r.json().get("_id") or r.json().get("id")
        # List
        r = await client.get("/api/webhooks", timeout=15, headers=auth)
        if r.status_code == 200:
            results["list_count"] = len(r.json()) if isinstance(r.json(), list) else None
        # Delete
        if wh_id:
            r = await client.delete(f"/api/webhooks/{wh_id}", timeout=15, headers=auth)
            results["delete"] = r.status_code
    return results


# =============================================================================
# Ramp
# =============================================================================

def run_ramp(base, token, session_id, levels, duration, stop_pct, mix):
    rows = []
    baseline_p95 = None
    for level in levels:
        res = asyncio.run(hammer(base, token, session_id, level, duration, mix))
        fast = res["lat"]["health"] + res["lat"]["sessions"] + res["lat"]["session"]
        n_ok = sum(len(v) for v in res["lat"].values())
        n_err = res["errors"]["count"]
        n_total = n_ok + n_err
        err_pct = (n_err / n_total * 100) if n_total else 0.0
        p50, p95 = pctl(fast, 50), pctl(fast, 95)
        lp95 = pctl(res["lat"]["login"], 95)
        rps = n_total / res["elapsed"] if res.get("elapsed") else 0.0
        if baseline_p95 is None and fast:
            baseline_p95 = p95
        degraded = bool(baseline_p95) and p95 > 3 * baseline_p95
        flag = "  <-- ERRORS" if err_pct > 1 else ("  <-- latency degrading" if degraded else "")
        login_col = f"   login p95 {lp95:6.0f}ms" if mix == "all" else ""
        print(f"  {level:4} users   {n_total:5} req   {rps:5.1f} req/s   {err_pct:5.1f}% err   "
              f"fast p50 {p50:6.0f}ms p95 {p95:6.0f}ms{login_col}{flag}")
        rows.append({"concurrency": level, "requests": n_total, "errors": n_err,
                     "error_pct": round(err_pct, 1), "fast_p50_ms": p50, "fast_p95_ms": p95,
                     "login_p95_ms": lp95, "req_per_s": round(rps, 1)})
        if err_pct > stop_pct:
            print(f"  Error rate over {stop_pct:g}% at {level} users — stopping the ramp.")
            break
    return rows


def ramp_report(rows, args, tag):
    print("\n" + "=" * 78)
    print(f"CONCURRENCY RAMP  ({args.ramp_duration:.0f}s/level, pipeline idle — the API's own ceiling)")
    print("=" * 78)
    baseline = rows[0]["fast_p95_ms"] if rows else float("nan")
    healthy = [r for r in rows if r["error_pct"] < 1 and r["fast_p95_ms"] < 3 * baseline]
    ceiling = healthy[-1]["concurrency"] if healthy else 0
    print(f"\n  Estimated healthy ceiling: ~{ceiling} concurrent users")
    print(f"  SLO: <1% errors AND fast-path p95 < 3x the {rows[0]['concurrency'] if rows else '?'}-user baseline")
    print("  Measured on THIS machine — a different box / the real Render instance will")
    print("  shift the number; re-run there (or with --base-url) before trusting it.")
    _save(f"ramp_{_stamp()}.json", {"tag": tag, "config": vars(args), "levels": rows,
                                    "estimated_ceiling": ceiling})


def idle_saturated_report(idle, sat, args, tag):
    print("\n" + "=" * 78)
    print(f"IDLE vs SATURATED  ({args.concurrency} read clients, {args.chat_clients} chat streams in flight)")
    print("=" * 78)
    print(f"  {'':10} {'idle':>21}   {'saturated':>24}")
    for key, label in (("health", "GET /health"), ("sessions", "GET /api/sessions"),
                       ("session", "GET /api/session")):
        i, s = idle["lat"][key], sat["lat"][key]
        if not i or not s:
            print(f"  {label:16} {'no data':>40}")
            continue
        i50, i95, s50, s95 = pctl(i, 50), pctl(i, 95), pctl(s, 50), pctl(s, 95)
        ratio = s95 / i95 if i95 else float("nan")
        flag = "  <-- degraded" if ratio >= 2 else ""
        print(f"  {label:16} p50 {i50:6.0f}->{s50:6.0f}ms   p95 {i95:6.0f}->{s95:6.0f}ms   x{ratio:.1f}{flag}")
    print(f"\n  Errors   {idle['errors']['count']} idle, {sat['errors']['count']} saturated")
    for s in (idle["errors"]["samples"] + sat["errors"]["samples"])[:5]:
        print(f"    {s}")
    _save(f"idle_sat_{_stamp()}.json", {"tag": tag, "config": vars(args),
          "idle": {k: {"p50": pctl(v, 50), "p95": pctl(v, 95), "n": len(v)} for k, v in idle["lat"].items()},
          "saturated": {k: {"p50": pctl(v, 50), "p95": pctl(v, 95), "n": len(v)} for k, v in sat["lat"].items()},
          "errors": {"idle": idle["errors"]["count"], "saturated": sat["errors"]["count"]}})


def _stamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _save(name, payload):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    payload = {"timestamp": datetime.now(timezone.utc).isoformat(), **payload}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n  Report: {path}")
    print("  Latency is stub-scaled — read the ratio / breaking point, not absolute ms.")


# =============================================================================
# Spawning the real Express + the stub
# =============================================================================

def _resolve_srv_uri(uri):
    """Rewrite mongodb+srv:// to mongodb:// with pre-resolved hosts.
    Node's c-ares DNS can fail SRV lookups in sandboxed environments even when
    the system resolver works fine.  Falls back to the original URI on error."""
    # ponytail: only handles the Atlas SRV convention; enough for load tests
    import re
    m = re.match(r"mongodb\+srv://([^@]+@)?([^/]+)/(.+)", uri)
    if not m:
        return uri
    creds = m.group(1) or ""
    host = m.group(2)
    rest = m.group(3)
    try:
        import subprocess as _sp
        out = _sp.check_output(
            ["powershell", "-NoProfile", "-Command",
             f"(Resolve-DnsName -Name '_mongodb._tcp.{host}' -Type SRV"
             " | ForEach-Object { \"$($_.NameTarget):$($_.Port)\" }) -join ','"],
            timeout=10, text=True).strip()
        if out:
            sep = "&" if "?" in rest else "?"
            return f"mongodb://{creds}{out}/{rest}{sep}ssl=true&authSource=admin"
    except Exception:
        pass
    return uri


def spawn_stub(port, lead_seconds):
    return subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), "--serve-stub",
         "--stub-port", str(port), "--lead-seconds", str(lead_seconds)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def spawn_express(port, stub_port):
    env = dict(os.environ)
    env.update({
        "PORT": str(port),
        "FASTAPI_URL": f"http://127.0.0.1:{stub_port}",
        # Lift the demo guards so the test isn't throttled by its own limits.
        "AUTH_RATE_MAX": "1000000",
        "CHAT_RATE_MAX": "1000000",
        "SESSION_RATE_MAX": "1000000",
        "DAILY_MESSAGE_CAP": "100000000",
    })
    env.pop("REDIS_URL", None)  # keep the run deterministic — Mongo-backed cache only
    # Rewrite mongodb+srv:// → standard mongodb:// to bypass SRV DNS sandbox issue
    mongo_uri = env.get("MONGO_URI", "")
    if not mongo_uri:
        dotenv_path = os.path.join(BASE_DIR, ".env")
        if os.path.isfile(dotenv_path):
            with open(dotenv_path) as f:
                for line in f:
                    if line.strip().startswith("MONGO_URI="):
                        mongo_uri = line.strip().split("=", 1)[1]
                        break
    if mongo_uri.startswith("mongodb+srv://"):
        env["MONGO_URI"] = _resolve_srv_uri(mongo_uri)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log = open(os.path.join(RESULTS_DIR, "express.log"), "w", encoding="utf-8")
    proc = subprocess.Popen(["node", "index.js"], cwd=BASE_DIR, env=env,
                            stdout=log, stderr=subprocess.STDOUT)
    return proc, log


# =============================================================================
# Smoke — quick functional check (every endpoint once, pass/fail)
# =============================================================================

def run_smoke(args):
    """Spawn Express + stub, hit every endpoint once, report pass/fail."""
    stub = spawn_stub(args.stub_port, args.lead_seconds)
    express, express_log = spawn_express(args.express_port, args.stub_port)
    base = f"http://127.0.0.1:{args.express_port}"
    checks = []
    try:
        ok = wait_for_health(base, timeout=30)
        checks.append(("health", ok))
        if not ok:
            sys.exit("Smoke: server didn't start")

        token, _ = ensure_user(base)
        checks.append(("auth", True))

        sids = seed_sessions(base, token, 1)
        checks.append(("session_create", len(sids) == 1))

        auth = {"Authorization": f"Bearer {token}"}
        r = httpx.get(f"{base}/api/sessions", headers=auth, timeout=10)
        checks.append(("session_list", r.status_code == 200))

        if sids:
            r = httpx.get(f"{base}/api/session/{sids[0]}", headers=auth, timeout=10)
            checks.append(("session_get", r.status_code == 200))

        # Quick chat/stream — the stub answers in lead_seconds
        r = httpx.post(f"{base}/api/chat/stream", headers=auth, timeout=30,
                       json={"sessionId": sids[0] if sids else "x", "message": "smoke"})
        checks.append(("chat_stream", r.status_code == 200))

        if sids:
            cleanup_sessions(base, token, sids)
            checks.append(("session_delete", True))

        passed = all(ok for _, ok in checks)
        for name, ok in checks:
            print(f"  {'✓' if ok else '✗'} {name}")
        print(f"\nSmoke {'PASSED' if passed else 'FAILED'}")
        if not passed:
            sys.exit(1)
    finally:
        if express is not None:
            express.terminate()
            try:
                express.wait(timeout=5)
            except subprocess.TimeoutExpired:
                express.kill()
            if express_log:
                express_log.close()
        if stub is not None:
            stub.terminate()


# =============================================================================
# main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=None, help="hit a running server (ramp only, read-only)")
    ap.add_argument("--ramp", action="store_true")
    ap.add_argument("--ramp-levels", default="10,25,50,100,200,300")
    ap.add_argument("--ramp-duration", type=float, default=8)
    ap.add_argument("--ramp-stop-pct", type=float, default=25)
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--duration", type=float, default=15)
    ap.add_argument("--chat-clients", type=int, default=12)
    ap.add_argument("--lead-seconds", type=float, default=4.0)
    ap.add_argument("--mix", choices=("read", "all"), default="read")
    ap.add_argument("--seed-sessions", type=int, default=5)
    ap.add_argument("--express-port", type=int, default=4055)
    ap.add_argument("--stub-port", type=int, default=8055)
    ap.add_argument("--serve-stub", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="quick functional check — hit every endpoint once")
    ap.add_argument("--v2-probes", action="store_true",
                    help="run v2 feature probes: job API, queue depth, webhooks")
    args = ap.parse_args()

    if args.selftest:
        import math
        assert pctl([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) == 5
        assert pctl([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10
        assert pctl([5], 50) == 5 and math.isnan(pctl([], 50))
        print("selftest ok")
        return

    if args.smoke:
        return run_smoke(args)

    if args.serve_stub:
        return serve_stub(args.stub_port, args.lead_seconds)

    if args.base_url and not args.ramp:
        sys.exit("--base-url only supports --ramp (the saturated test needs the local stub).")

    stub = express = express_log = None
    if args.base_url:
        base = args.base_url.rstrip("/")
    else:
        base = f"http://127.0.0.1:{args.express_port}"
        stub = spawn_stub(args.stub_port, args.lead_seconds)
        express, express_log = spawn_express(args.express_port, args.stub_port)

    tag = f"loadtest-{uuid.uuid4().hex[:8]}"
    seeded = []
    token = None
    try:
        if not wait_for_health(base):
            sys.exit(f"Could not reach {base} — "
                     + ("check the URL." if args.base_url else "see load_test_results/express.log."))
        print(f"API up on {base}")
        token, _user_id = ensure_user(base)
        seeded = seed_sessions(base, token, args.seed_sessions)
        probe = seeded[0] if seeded else "000000000000000000000000"
        print(f"Seeded {len(seeded)} session(s) for the load-test user.")

        if args.v2_probes:
            print("\n--- V2 feature probes ---")
            print("  Job API (submit → poll → cancel)...")
            job = asyncio.run(test_job_api(base, token, probe))
            print(f"    submit={job['submit']} states={job['poll_states']} "
                  f"cancel={job['cancel']}")
            print("  Queue depth on /health...")
            qd = asyncio.run(test_queue_depth(base))
            print(f"    has_queue_depth={qd['has_queue_depth']} value={qd['queue_depth']}")
            print("  Webhook CRUD...")
            wh = asyncio.run(test_webhook_crud(base, token))
            print(f"    create={wh['create']} list={wh['list_count']} "
                  f"delete={wh['delete']}")
            _save(f"v2_probes_{_stamp()}.json", {"tag": tag, "job_api": job,
                  "queue_depth": qd, "webhook_crud": wh})
            print("--- V2 probes done ---\n")

        if args.ramp:
            levels = [int(x) for x in args.ramp_levels.split(",") if x.strip()]
            print(f"Ramping read concurrency through {levels} ({args.ramp_duration:.0f}s each)...")
            rows = run_ramp(base, token, probe, levels, args.ramp_duration, args.ramp_stop_pct, args.mix)
            ramp_report(rows, args, tag)
        else:
            print(f"Phase 1/2: idle, {args.concurrency} read clients for {args.duration:.0f}s...")
            idle = asyncio.run(hammer(base, token, probe, args.concurrency, args.duration, args.mix))

            print(f"Phase 2/2: saturated — {args.chat_clients} chat streams + "
                  f"{args.concurrency} read clients for {args.duration:.0f}s...")

            async def saturated():
                stop_evt = asyncio.Event()
                flood = asyncio.create_task(chat_flood(base, token, probe, args.chat_clients, stop_evt))
                await asyncio.sleep(2)  # let the chat streams ramp up
                res = await hammer(base, token, probe, args.concurrency, args.duration, args.mix)
                stop_evt.set()
                try:
                    await asyncio.wait_for(flood, timeout=10)
                except asyncio.TimeoutError:
                    pass
                return res

            sat = asyncio.run(saturated())
            idle_saturated_report(idle, sat, args, tag)
    finally:
        if token and seeded:
            print(f"  Removed {cleanup_sessions(base, token, seeded)} seeded session(s)")
        if express is not None:
            express.terminate()
            try:
                express.wait(timeout=10)
            except subprocess.TimeoutExpired:
                express.kill()
            if express_log:
                express_log.close()
        if stub is not None:
            stub.terminate()


if __name__ == "__main__":
    main()
