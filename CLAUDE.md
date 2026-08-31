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
| `backend-python/main.py` | FastAPI app, /pipeline/run, /pipeline/stream |
| `backend-python/stages/` | 7-stage RAG pipeline (query expansion → response assembly) |
| `backend-node/index.js` | Express server, health, CORS |
| `backend-node/routes/chat.js` | POST /chat/stream (SSE proxy to FastAPI) |
| `backend-node/load_test.py` | Load test harness (spawns Express + stub FastAPI) |
| `.github/workflows/ci.yml` | CI: lint, syntax, build, Docker images, gated Render deploy |

## Caching Layers

1. **Mongo query-result cache** — SHA-256(disease|intent|message+history), 24h TTL, skips entire pipeline
2. **Semantic query cache** — cosine ≥0.97 on first-turn embeddings in Redis, skips pipeline
3. **Embedding cache** — per (model, text) in Redis, 7-day TTL

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

# Load test (spawns Express + stub FastAPI, zero HF cost)
cd backend-node && python load_test.py --ramp         # concurrency ceiling
cd backend-node && python load_test.py                 # idle vs saturated
cd backend-node && python load_test.py --smoke         # quick functional check
cd backend-node && python load_test.py --selftest      # CI smoke test
```

## Environment

- `.env` files in `backend-python/` and `backend-node/` (gitignored)
- Redis: Upstash (`rediss://...`)
- `CLOUDFLARE_MAX_TOKENS` is hardcoded to 4096 in `llm_backend.py`, not an env var

## Branch Strategy

- `main` — v0 core + bug fixes
- `curalink-v0-v1-practice` — v0 + v1 reliability features
- `curalink-v2` — v0 + v1 + v2 scale features

## Planning Docs (gitignored)

- `upgrade_roadmap.txt` — full roadmap (Parts 1-4)
- `scale_readiness_audit.txt` — 8-layer reference architecture benchmark
