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
| `backend-python/redis_cache.py` | Embedding cache (Upstash Redis) |
| `backend-python/main.py` | FastAPI app, /pipeline/run, /pipeline/stream, `X-Internal-API-Key` middleware |
| `backend-python/stages/` | 7-stage RAG pipeline (query expansion → response assembly); recommendations must cite sources |
| `backend-python/semantic_cache.py` | Tenant-isolated semantic query cache (`semq:<userId>:<hash>`) |
| `backend-python/observability.py` | Content-free LLM/HTTP telemetry and metrics exporters |
| `backend-node/index.js` | Express server, health, CORS, credits, DELETE /api/account |
| `backend-node/routes/chat.js` | POST /chat, /chat/stream (SSE proxy), tenant cache keys, urgent-use diversion |
| `backend-node/cache.js` | Query cache (Redis or Mongo fallback) + per-user cache deletion |
| `frontend/src/components/LegalPage.jsx` | Privacy Notice + Terms of Use |
| `backend-node/load_test.py` | Load test harness (spawns Express + stub FastAPI) |
| `backend-node/middleware/auth.js` | Access (1h) and refresh (7d) JWTs; `authVersion` revokes refresh on logout |
| `backend-node/own_keys.js` | Parses, validates, and forwards user-supplied provider keys |
| `backend-python/own_keys.py` | Verifies HF / Cloudflare keys; builds per-request inference clients |
| `frontend/src/session.js` | Tab-scoped `sessionStorage` tokens, auto-refresh, own-key headers |
| `frontend/src/components/ApiKeySettings.jsx` | Own-API-key settings popup |
| `.github/workflows/ci.yml` | CI: lint, unit tests, syntax, build, Docker images, gated Render deploy |

## Caching Layers

1. **Query-result cache** — `query:<userId>:SHA-256(user|disease|intent|location|message|history)`, 24h TTL, Redis with Mongo fallback, skips entire pipeline
2. **Semantic query cache** — cosine ≥0.97 on first-turn embeddings in Redis, bucketed per user + hashed disease/intent/location, skips pipeline
3. **Embedding cache** — per (model, text) in Redis, 7-day TTL

## Sessions and Own API Keys

- Login is tab-scoped (`sessionStorage`); closing the tab logs out. Access tokens renew every 45 min and on 401 while the tab is open; logout increments `User.authVersion` to revoke refresh tokens
- Auth rate limit applies only to signup/login; `/me` session checks are not throttled and only a 401 clears the session
- Users may add their own keys in Settings: HF token always (embeddings + MedCPT + HF LLM); Cloudflare account ID + token also when `LLM_MODEL=CLOUDFLARE`
- Keys are verified via `/keys/validate` (HF `whoami-v2`, Cloudflare account token verify), stored only in the browser tab, forwarded as `X-Provider-*` headers, never persisted or logged
- Own-key requests skip the daily message cap and the exact/semantic caches, and mark user messages `ownKey: true` so they never count toward the free quota
- Landing hero metrics: 98% scope-routing accuracy (`abstain_correct`), 93% citation coverage (`citations_grounded`), 2.3 s p95 API latency at 100 concurrent users (Render, stubbed pipeline)

## Commands

```bash
# Dev
cd backend-python && uvicorn main:app --reload --port 8000
cd backend-node && npm run dev
cd frontend && npm run dev

# Test
cd backend-python && python -m ruff check . ../backend-node/load_test.py && python -m compileall -q . && python -m unittest discover -p "test_*.py"
cd backend-node && npm test
cd frontend && npm run lint && npm run build

# Load test (spawns Express + stub FastAPI, zero HF cost)
cd backend-node && python load_test.py --ramp         # concurrency ceiling
cd backend-node && python load_test.py                 # idle vs saturated
cd backend-node && python load_test.py --smoke         # quick functional check
cd backend-node && python load_test.py --v2-probes     # job API, queue depth, webhooks
cd backend-node && python load_test.py --selftest      # CI smoke test

# Pipeline quality eval (needs FastAPI running on :8000, hits real LLM — $0 on free tier)
cd backend-python && python eval_harness.py            # full 50-query eval (needs INTERNAL_API_KEY)
cd backend-python && python eval_harness.py --query 0  # single query by index
cd backend-python && python eval_harness.py --selftest # validate eval set only
```

## Environment

- `.env` files in `backend-python/` and `backend-node/` (gitignored)
- `INTERNAL_API_KEY` is required and must match in both backend services
- Redis: Upstash (`rediss://...`)
- `CLOUDFLARE_MAX_TOKENS` is hardcoded to 4096 in `llm_backend.py`, not an env var

## Branch Strategy

- `main` — v0 core + bug fixes + public-beta safeguards (de-identified context, tenant-isolated caches,
  internal service auth, cited recommendations, urgent-use diversion, consent, retention/deletion, mobile UI)
- `curalink-v012` — v0 + v1 + v2 (all features)
