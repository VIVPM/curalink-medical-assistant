# Curalink — AI Medical Research Assistant

## Architecture

Three-service split: React (Vite) frontend → thin Express API → stateless FastAPI orchestrator.
Routing/DB in Node, entire AI pipeline in Python. Deployed on Render free tier + MongoDB Atlas M0.

## LLM Provider

`LLM_MODEL` env var in `backend-python/.env` selects the provider:
- `CLOUDFLARE` → Cloudflare Workers AI (`@cf/openai/gpt-oss-20b`), needs `CLOUDFLARE_ACCOUNT_ID` + `CLOUDFLARE_API_TOKEN`
- Any other value → HuggingFace Inference API (value = model id, e.g. `meta-llama/Llama-3.3-70B-Instruct`), needs `HF_TOKEN`

One provider active at a time. Factory: `get_llm_backend()` in `llm_backend.py`.

## Key Files

| File | Purpose |
|------|---------|
| `backend-python/llm_backend.py` | LLMBackend ABC, HFBackend, CloudflareBackend, factory |
| `backend-python/redis_cache.py` | Embedding cache + prompt-level LLM cache (Upstash Redis) |
| `backend-python/main.py` | FastAPI app, /pipeline/run, /pipeline/stream, graceful shutdown |
| `backend-python/stages/` | 7-stage RAG pipeline (query expansion → response assembly) |
| `backend-node/index.js` | Express server, health, CORS, graceful shutdown, DELETE /api/account |
| `backend-node/routes/chat.js` | POST /chat/stream, idempotency keys, Retry-After headers |
| `backend-node/models/Session.js` | Session schema + 90-day TTL index |
| `backend-python/circuit_breaker.py` | CircuitBreaker + ResilientLLM auto-fallback wrapper |
| `backend-python/job_manager.py` | Async job queue, state machine, backpressure, per-tenant fairness |
| `backend-python/egress_allowlist.py` | Egress allowlist (SSRF protection) for httpx clients |
| `backend-python/checkpoint.py` | Per-step pipeline checkpointing in Redis |
| `backend-python/event_buffer.py` | SSE event buffering + Last-Event-ID replay |
| `backend-python/pii_redactor.py` | PII redaction for observability spans |
| `backend-python/token_budget.py` | Per-job token budget enforcement |
| `backend-node/routes/jobs.js` | Async job submit/poll/cancel proxy to FastAPI |
| `backend-node/routes/webhooks.js` | Webhook CRUD + HMAC-signed dispatch |
| `backend-node/models/AuditLog.js` | Audit log schema (1-year TTL) |
| `backend-node/models/Webhook.js` | Webhook registration schema |
| `backend-node/middleware/audit.js` | Audit logging middleware (fire-and-forget) |
| `.github/workflows/ci.yml` | CI: lint, syntax, build, Docker images, gated Render deploy |

## Caching Layers

1. **Mongo query-result cache** — SHA-256(disease|intent|message+history), 24h TTL, skips entire pipeline
2. **Semantic query cache** — cosine ≥0.97 on first-turn embeddings in Redis, skips pipeline
3. **Embedding cache** — per (model, text) in Redis, 7-day TTL
4. **Prompt-level LLM cache** — hash(model+system_prompt+user_prompt) in Redis, 24h TTL

## V1 Features (curalink-v0-v1-practice branch)

- Retry-After headers (429 + 402)
- Graceful shutdown (SIGTERM/SIGINT, 30s drain)
- Idempotency-Key dedup (5-min window)
- History summarization (6 recent + topic hints)
- User account deletion (cascade)
- 90-day data retention (TTL index)
- Structured output validation + repair
- Jittered retries (both backends)
- Prompt-level LLM cache in Redis
- Redis TLS via certifi CA bundle

## V2 Scale Features (curalink-v0-v1-practice branch)

- Circuit breaker + auto-fallback (ResilientLLM wraps primary + fallback provider)
- Async job API: POST /jobs → 202, GET /jobs/{id}, DELETE /jobs/{id}
- Job state machine: pending → running → completed | failed | cancelled (Redis-backed)
- Queue + backpressure: asyncio.Queue with configurable size, 503 when full
- Per-step checkpointing in Redis (resume from last stage on retry)
- SSE event buffering + Last-Event-ID replay
- Queue-depth metric on /health (for autoscaling triggers)
- Token-aware rate limits (DAILY_TOKEN_CAP env)
- PII redaction on observability spans (email/phone/SSN)
- Webhooks (CRUD + HMAC-signed dispatch on job.completed/failed)
- Audit log (AuditLog model + middleware, 1-year TTL)
- Per-job token budget (MAX_TOKENS_PER_JOB env, default 50k)
- Per-tenant fairness (round-robin dispatcher across per-user queues)
- Egress allowlist (SSRF protection on all httpx clients)
- Cost + TTFT dashboards (llm_ttft_seconds histogram + llm_tokens_total counter)
- Correlation IDs (X-Request-Id propagated Express → FastAPI → spans)

## Commands

```bash
# Dev
cd backend-python && uvicorn main:app --reload --port 8000
cd backend-node && npm run dev
cd frontend && npm run dev

# Test
cd backend-python && python -m py_compile main.py
cd backend-node && node --check index.js
cd frontend && npm run lint && npm run build
```

## Environment

- `.env` files in `backend-python/` and `backend-node/` (gitignored)
- Redis: Upstash (`rediss://...huge-garfish-223261.upstash.io:6379`)
- `CLOUDFLARE_MAX_TOKENS` is hardcoded to 4096 in `llm_backend.py`, not an env var

## Planning Docs (gitignored)

- `upgrade_roadmap.txt` — full roadmap (Parts 1-4)
- `v2_scale_roadmap.txt` — v2 items with emoji status table
