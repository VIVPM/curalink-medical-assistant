# Curalink — AI Medical Research Assistant

An AI-powered medical research companion built with React, Express, Neon Postgres, and a FastAPI orchestrator. Curalink uses de-identified patient context to retrieve research from PubMed, OpenAlex, and ClinicalTrials.gov, reasons over it with a configurable LLM (Hugging Face Inference API or Cloudflare Workers AI), and delivers structured answers with sources users can inspect.

> **Multi-provider LLM**: `LLM_MODEL` selects the platform-funded default (`CLOUDFLARE` or a Hugging Face model ID). Users with validated own keys choose either answer provider in Settings. On a Cloudflare-default server, own-key Hugging Face requests use `HF_FALLBACK_MODEL` (default: `meta-llama/Llama-3.3-70B-Instruct`).

## Features

- **Structured intake + natural chat** — fill disease/intent once, then chat naturally; follow-ups inherit context automatically
- **7-stage AI pipeline** — query expansion → parallel retrieval → normalization → hybrid re-ranking → context building → LLM reasoning → response assembly
- **Three current medical sources** — cache misses query PubMed, OpenAlex, and ClinicalTrials.gov in parallel (~170 unique candidates per query)
- **Domain-specialized ranking** — BM25 + PubMedBERT embeddings fused via Reciprocal Rank Fusion, refined by a MedCPT cross-encoder with source-balanced selection
- **Inspectable sources** — research findings and personalized recommendations include titles, authors, years, URLs, and supporting excerpts for verification against the original publications
- **Emergency boundary** — urgent-use messages are diverted from the research pipeline to immediate emergency-services guidance
- **Real-time SSE streaming** — live pipeline progress + token-by-token LLM output through FastAPI → Express → React
- **Multi-turn context awareness** — chat history and static form context are merged into every query expansion
- **Clinical trial geo-filtering** — optional location input geocodes and filters trials within 100 miles via ClinicalTrials.gov geo API
- **Optional patient name** — use a nickname or initials to label a session. The field is saved as `session.patientName` separately from research context, displayed when reopening the session, and excluded from AI/retrieval requests and answer-cache inputs. Existing databases receive the column through schema bootstrap on backend startup.
- **Privacy controls** — de-identified-use policy, 90-day session/message retention, per-session deletion, and complete account deletion from the UI
- **Tab-scoped sessions** — login lives only in the current browser tab (`sessionStorage`); closing the tab logs out, 1-hour access tokens renew automatically while the tab is open, and logout revokes renewal tokens
- **Bring your own API keys** — choose Hugging Face or Cloudflare for answers in Settings. A Hugging Face token funds ranking for either choice; Cloudflare also needs an account ID and token. Keys are validated and stored encrypted on the account, never returned to the browser, and reused after login. Save and use a provider or reuse its saved credentials without re-entering keys; remove active credentials in Settings to return to free credits. Complete, valid selected keys hide daily credits and bypass the free question cap.
- **ChatGPT-style session sidebar** — click any past session to reopen it and keep asking; new messages append to that session's history
- **Landing page** — two-column hero with the pitch and sign-up on the left, and the animated research demo plus headline metrics (answer relevancy 91%, faithfulness 99%, cost per query $0.0015) on the right
- **Responsive authenticated UI** — collapsed mobile navigation and diagnostics overlays verified at 390px and 430px widths
- **Private service boundary** — a shared `INTERNAL_API_KEY` protects non-health FastAPI endpoints from direct public use
- **Redis caching** — tenant-isolated exact + semantic query caches, document embeddings, and prompt-level responses on Upstash with a Postgres query-cache fallback
- **Per-user credits + rate limiting** — 5 questions/day (DAILY_MESSAGE_CAP) on platform keys, waived for users who bring their own keys, plus per-IP / per-user limits on auth, chat, and session creation
- **Observability** — content-free LLM metadata to Langfuse over OTLP
- **CI/CD + Docker** — GitHub Actions (lint, syntax, build, image builds, gated Render deploy) and Dockerfiles for all three services
- **Reliability (v1)** — Retry-After headers, graceful shutdown, idempotency keys, history summarization, user account deletion, 90-day data retention, structured output validation with repair, jittered retries, prompt-level LLM cache
- **Scale modules (v2)** — async jobs, queue/backpressure/fairness, checkpoints, job-event replay, token-aware limits, content-free telemetry, signed webhooks, audit logs, token budgets, egress controls, and correlation IDs; optional fallback wiring is pending

## 🏗️ Architecture

Five layers, read top to bottom. Each arrow is a hand-off between layers; the
shared services (data, external AI) are reached per layer rather than by every
stage, so the flow stays legible. Caching and observability are cross-cutting.

```mermaid
graph TD
    User(["👤 Patient · caregiver · clinician"])

    subgraph CLIENT ["1 · Client — React / Vite"]
        Land["🛬 Landing page"]
        UI["💬 Chat UI · streamed answers · intake form · session sidebar"]
    end

    subgraph API ["2 · API layer — Express (Node)"]
        Auth["🔐 Auth · bcrypt · JWT · rate-limit · 3-credit quota"]
        REST["🗂️ Sessions CRUD · POST /chat/stream → SSE proxy"]
    end

    subgraph ORCH ["3 · Orchestration — FastAPI · 8-stage pipeline"]
        S0["1 · Routing · emergency · clarification · history-only · RAG · 8B/70B"]
        S1["2 · Query expansion (LLM)"]
        S2["3 · Parallel retrieval"]
        S3["4 · Normalize + dedupe"]
        S4["5 · Hybrid ranking · BM25 · PubMedBERT · RRF · MedCPT · source balance"]
        S5["6 · Context build"]
        S6["7 · Grounded reasoning (LLM)"]
        S7["8 · Response assembly · cite-or-abstain"]
        S0 --> S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7
    end

    subgraph DATA ["4 · Data — Neon Postgres + Redis (Upstash)"]
        DB[("Postgres · users · sessions · messages")]
        Redis[("Redis · query + embedding + semantic caches")]
    end

    subgraph EXT ["5 · External AI + data"]
        HF["🤗 HF Inference API · Llama · PubMedBERT · MedCPT"]
        Src["📚 PubMed · 🔬 OpenAlex · 🧪 ClinicalTrials.gov"]
    end

    Cache["🗄️ Caching · cross-cutting<br>exact + semantic query cache · doc-embedding cache"]
    OBS["📈 Observability · cross-cutting<br>Content-free LLM metadata (Langfuse)"]

    User --> CLIENT
    CLIENT -->|HTTP + JWT + SSE| API
    API -->|users · sessions · messages| DB
    API -->|query cache| Redis
    API -->|POST /pipeline/stream| ORCH
    S0 -.->|optional Jev| HF
    S2 -->|parallel httpx| Src
    S4 -->|embeddings · rerank| HF
    S6 -->|generate| HF
    ORCH -->|embedding + semantic cache| Redis
    ORCH -.->|hit / miss| Cache
    ORCH -.->|LLM traces| OBS
```

### Pipeline Stages (inside FastAPI)

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | `query_router.py` `routing.py` | Emergency/clarification/history-only/RAG route; optional Jev 8B vs 70B; history-only can skip retrieval |
| 2 | `query_expander.py` | LLM rewrites the current question; extra queries are side-effect search |
| 3 | `pubmed.py` `openalex.py` `trials.py` | Parallel retrieval from 3 sources (~210 raw → ~170 after dedupe) |
| 4 | `normalizer.py` `merger.py` | Unify schemas into `Document[]`, dedupe by DOI/PMID/NCT-ID, quality filter |
| 5 | `ranker.py` | BM25 pre-filter → PubMedBERT cosine → RRF fusion → MedCPT cross-encoder → source-balanced top 14 |
| 6 | `context_builder.py` | Patient-facing prompt with citation anchors `[doc1]` and grounding rules |
| 7 | `llm_reasoner.py` | Routed 8B or 70B via HF or Cloudflare — source-constrained generation |
| 8 | `response_assembler.py` | Citation resolution, snippet extraction, structured JSON assembly |

## Project Structure

```
curalink-medical-assistant/
├── frontend/                      # React (Vite) UI
│   └── src/
│       ├── components/
│       │   ├── AuthPage.jsx       # Login / signup form
│       │   ├── Sidebar.jsx        # Session list sidebar
│       │   ├── IntakeForm.jsx     # De-identified patient context (disease, intent, general location)
│       │   ├── LegalPage.jsx      # Privacy notice + terms of use
│       │   ├── ApiKeySettings.jsx # Own-API-key settings popup
│       │   ├── ChatView.jsx       # Chat interface with message bubbles
│       │   ├── StructuredResponse.jsx  # Renders overview + insights + trials
│       │   ├── InsightCard.jsx    # Individual research insight with sources
│       │   ├── TrialCard.jsx      # Clinical trial card with NCT ID + status
│       │   ├── PipelineProgress.jsx   # Real-time stage progress indicator
│       │   └── PipelinePanel.jsx  # Detailed pipeline metadata panel
│       ├── hooks/
│       │   ├── useAuth.js         # JWT auth state management
│       │   └── useChat.js         # Chat + SSE streaming logic
│       ├── session.js             # Tab-scoped tokens and token renewal
│       └── App.jsx                # Root component with routing
│
├── backend-node/                  # Express API (thin layer)
│   ├── index.js                   # Server entry, health check, CORS
│   ├── own_keys.js                # Validates and forwards user-supplied provider keys
│   ├── provider_settings.js       # Encrypted account credentials and provider selection
│   ├── routes/
│   │   ├── auth.js                # Signup, login, token refresh, logout, current user
│   │   ├── session.js             # POST /api/session, GET /api/sessions
│   │   └── chat.js                # POST /api/chat/stream (SSE proxy to FastAPI)
│   ├── db.js                      # Postgres pool, schema bootstrap, retention purge, row mappers
│   └── middleware/
│       └── auth.js                # JWT verification middleware
│
├── backend-python/                # FastAPI orchestrator (AI pipeline)
│   ├── main.py                    # FastAPI app, /pipeline/run, /pipeline/stream
│   ├── own_keys.py                # Verifies user-supplied provider keys
│   ├── llm_backend.py             # LLMBackend abstraction (HF + Cloudflare, multi-provider)
│   ├── sources/
│   │   ├── pubmed.py              # PubMed E-utilities (esearch + efetch)
│   │   ├── openalex.py            # OpenAlex works search
│   │   ├── trials.py              # ClinicalTrials.gov v2 API
│   │   ├── normalizer.py          # Source-specific → unified Document
│   │   ├── merger.py              # Cross-source dedupe + merge
│   │   └── geocode.py             # Nominatim geocoding for trial geo-filter
│   ├── schemas/
│   │   └── document.py            # Unified Document dataclass
│   ├── embeddings/
│   │   └── embedder.py            # PubMedBERT embeddings via HF Inference API
│   ├── routing.py                 # Jev route + model-tier helper
│   ├── ranking/
│   │   ├── ranker.py              # Full ranking pipeline orchestration
│   │   ├── bm25.py                # BM25 sparse scoring
│   │   ├── cosine.py              # Dense cosine similarity
│   │   ├── rrf.py                 # Reciprocal Rank Fusion
│   │   ├── boosts.py              # Recency + multi-source credibility boosts
│   │   ├── cross_encoder.py       # MedCPT cross-encoder via HF API
│   │   └── mmr.py                 # Maximal Marginal Relevance (diversity)
│   ├── stages/
│   │   ├── query_router.py        # Stage 1: route + model choice
│   │   ├── history_context.py     # Bounded history evidence for routing
│   │   ├── query_expander.py      # Stage 2: LLM-based query expansion
│   │   ├── context_builder.py     # Stage 6: Token-budgeted prompt assembly
│   │   ├── llm_reasoner.py        # Stage 7: Grounded LLM generation
│   │   └── response_assembler.py  # Stage 8: Citation resolution + assembly
│   └── requirements.txt
│
└── README.md
```

## Getting Started

### Prerequisites

- **Node.js** ≥ 18
- **Python** ≥ 3.10
- **Neon** Postgres database (free tier)
- **HuggingFace** account with API token
- **NCBI API key** (optional but recommended — lifts rate limit from 3 to 10 req/sec)

### Installation

```bash
git clone https://github.com/your-username/curalink-medical-assistant
cd curalink-medical-assistant
```

**Frontend:**
```bash
cd frontend
npm install
```

**Node backend:**
```bash
cd backend-node
npm install
```

**Python backend:**
```bash
cd backend-python
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

### Environment Setup

Copy `.env.example` to `.env` in each backend directory and fill in the values:

```bash
cp backend-python/.env.example backend-python/.env
cp backend-node/.env.example backend-node/.env
```

#### Python Backend (`backend-python/.env`)

| Variable | Description |
|----------|-------------|
| `LLM_MODEL` | **Required.** `CLOUDFLARE` or a HuggingFace model id (e.g. `meta-llama/Llama-3.1-8B-Instruct`) |
| `INTERNAL_API_KEY` | **Required.** Shared secret for authenticated calls between Express and FastAPI |
| `NODE_API_URL` | **Required for saved-provider background jobs.** Trusted Express base URL (`http://localhost:4000` locally); use HTTPS or a private service network in deployment |
| `HF_TOKEN` | HuggingFace API token (when `LLM_MODEL` is a HF model id) |
| `CLOUDFLARE_ACCOUNT_ID` | CF account ID (when `LLM_MODEL=CLOUDFLARE`) |
| `CLOUDFLARE_API_TOKEN` | CF API token (when `LLM_MODEL=CLOUDFLARE`) |
| `NCBI_API_KEY` | NCBI E-utilities key (recommended) |
| `NCBI_EMAIL` | Contact email for NCBI policy compliance |
| `OPENALEX_EMAIL` | Contact email for OpenAlex polite pool |
| `REDIS_URL` | Upstash Redis connection string (shared with Node backend) |
| `BIENCODER_MODEL` | Embedding model (default: `pritamdeka/S-PubMedBert-MS-MARCO`) |

#### Node Backend (`backend-node/.env`)

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | Neon Postgres connection string (`.../curalink?sslmode=verify-full`) |
| `FASTAPI_URL` | **Required.** FastAPI orchestrator URL (`http://localhost:8000` locally) |
| `JWT_SECRET` | Secret for signing JWT tokens |
| `INTERNAL_API_KEY` | **Required.** Same shared secret configured on FastAPI |
| `ALLOWED_ORIGINS` | **Required.** Comma-separated CORS allow-list of frontend origins (`http://localhost:5173` locally) |
| `REDIS_URL` | Upstash Redis connection string (`rediss://...`) |
| `PORT` | Express server port (default: `4000`) |

### Saved provider credentials

Settings uses a simple account-backed BYOK flow: choose Hugging Face or Cloudflare, enter credentials, and click **Save and use**. Validation happens during saving. Saved credentials can be reused with the same button; **Remove** appears only when that provider has saved keys. Removing active credentials restores free-credit usage. Hugging Face is required for research ranking even when Cloudflare generates answers. Removing the HF token returns both providers to free usage; Cloudflare credentials can remain saved until removed separately.

- Credentials use AES-256-GCM with random IVs, authenticated to the user and credential type, and a key derived from `JWT_SECRET`. Keep that secret stable and protected: rotating it makes stored credentials unreadable, so users must save their keys again. Decryption or validation failures do not fall back to app-funded answers.
- Cloudflare validation makes a minimal Workers AI inference request (one output token) using the configured 8B model. This verifies account/model access for both user-owned and account-owned tokens instead of assuming a token ownership type. Saving/selecting Cloudflare and validating its chat requests can incur a small provider charge; access to the routed 70B model is still required.
- Only the selected provider, saved/masked-key status, and Cloudflare account ID are returned to the browser. Secrets are not stored in browser storage or sent with browser chat requests. Old tab-scoped keys are cleared on app load and must be saved explicitly in Settings.
- `GET/PUT /api/account/api-keys`, `PUT /api/account/provider`, and `DELETE /api/account/api-keys/:provider` manage settings. Account deletion cascades to the credential row. The table is created by the existing schema bootstrap.
- Regular, SSE and background-job requests preserve the account's selected provider and routed model tier. Jobs queue only the authenticated owner and selected-provider reference; the worker resolves and validates saved keys at execution time through `POST /internal/job-credentials`, protected by the shared internal secret and `Cache-Control: no-store`. Plaintext keys are not placed in queued payloads, checkpoints or persisted job state. Changed, removed, unreadable or rejected credentials fail the job without app-funded fallback; submit a new job after fixing Settings. Polling, cancellation and event replay are owner-scoped.
- Set `NODE_API_URL` on the Python service to the trusted Express service base URL, with matching `INTERNAL_API_KEY` values on both services. Existing explicit credential headers remain supported for regular/SSE API clients; background jobs require saved account credentials instead.

### Optional Jev query and model routing

Set `JEV_API_KEY` in the Python service to enable [TypeSafe AI Jev](https://docs.typesafe.ai/introduction). Leave it unset to retain the existing single-model pipeline. No additional dependency or vector database is required.

- The intake form's disease, intent and general location are context for every routing decision; they are not evidence. An explicit new topic overrides disease/intent for that turn without changing the saved form.
- The first medical question uses live-source RAG. Later questions use saved conversation/source snippets only when those support the answer; new evidence and freshness-sensitive questions require RAG.
- Jev independently selects Llama 8B or 70B for the answer. RAG query expansion always uses 8B. Missing/invalid Jev responses fall back to RAG + 70B; missing/insufficient history evidence falls back to RAG.
- Model IDs are configurable with `HF_LLM_8B_MODEL`, `HF_LLM_70B_MODEL`, `CF_LLM_8B_MODEL`, and `CF_LLM_70B_MODEL`; defaults are in `backend-python/.env.example`. The request's HF/Cloudflare provider, credentials and cache policy are preserved. Provider access to both models is required; there is no cross-provider or server-funded answer fallback for user-owned keys.
- Jev classification uses the server's TypeSafe key, including on requests with user-owned answer-provider keys. It receives intake context, the current question and a bounded history of up to six messages (24,000 serialized characters), including saved findings/source excerpts. Account IDs, inference keys, trial contacts and pipeline metadata are not included in router state. Free-text inputs are not automatically de-identified: do not enter identifiable patient information. Review TypeSafe's data-handling terms before enabling.
- Responses expose `pipelineMeta.routing` (route, tier, model, confidence values, retrieval usage and fallback). Confidence thresholds and routing-quality evaluation are intentionally deferred; probabilities are not a correctness guarantee.
- Regular, SSE and background-job requests use the same routing helper. History-only responses are generated and checked before final SSE metadata is sent, rather than streamed token-by-token, so an insufficient-evidence answer can safely fall back to retrieval.
- Semantic answer caching is bypassed while routing is enabled. Exact answer caches include prior structured evidence; there is no cache-version setting or versioned namespace.

Offline implementation checks (no live inference or evaluation):

```bash
cd backend-python
python -m unittest discover -p 'test_*.py'
```

### Running Locally

Start all three services:

```bash
# Terminal 1 — Python orchestrator
cd backend-python
uvicorn main:app --reload --port 8000

# Terminal 2 — Node API
cd backend-node
npm run dev

# Terminal 3 — React frontend
cd frontend
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | React + Vite | Responsive research chat, de-identified intake, structured source rendering |
| **API Layer** | Express.js | Auth, tenant isolation, sessions, SSE proxy, Postgres CRUD |
| **Orchestrator** | FastAPI (Python) | 7-stage AI pipeline + Redis-backed async jobs |
| **Database** | Neon Postgres | Accounts, consent, sessions, messages, audit logs, query-cache fallback |
| **Cache** | Redis (Upstash) | Tenant-isolated query/semantic caches, embeddings, prompts, jobs, checkpoints |
| **LLM** | HF Inference API or Cloudflare Workers AI | Query expansion + grounded reasoning (multi-provider) |
| **Bi-Encoder** | PubMedBERT-MS-MARCO via HF API | Domain-specialized dense retrieval |
| **Cross-Encoder** | MedCPT (NCBI) via HF API | Precision re-ranking on PubMed click logs |
| **Data Sources** | PubMed, OpenAlex, ClinicalTrials.gov | Live medical research APIs |

## Model Choices

| Model | Role | Why This Model |
|-------|------|----------------|
| `meta-llama/Llama-3.3-70B-Instruct` | LLM reasoning | Open-source, strong instruction following, JSON output compliance |
| `pritamdeka/S-PubMedBert-MS-MARCO` | Embedding (768-dim) | PubMed-pretrained backbone + MS-MARCO retrieval fine-tuning (~15pt recall uplift over generic models) |
| `ncbi/MedCPT-Cross-Encoder` | Final re-ranking | Built by NCBI, trained on real PubMed user click logs — domain + task match |

## Retrieval & Ranking Pipeline

```
210 raw candidates (80 PubMed + 80 OpenAlex + 50 Trials)
        │
        ▼
   ~170 unique (dedupe by DOI / PMID / NCT-ID)
        │
        ▼
   Quality filter → ~165 complete documents
        │
        ▼
   BM25 pre-filter → top 20
        │
        ▼
   PubMedBERT cosine + BM25 → RRF fusion → top 14
        │
        ▼
   Recency + multi-source credibility boosts
        │
        ▼
   MedCPT cross-encoder precision rerank
        │
        ▼
   Source-balanced publication/trial selection → top 14
        │
        ▼
   Token-budgeted context → LLM
```

## Evaluation Results

The **2026-11-01 post-fix run** evaluated all **150 patient/caregiver questions**
(50 easy, 58 medium, 42 hard) through the Python pipeline in process. Gemini
**3.5 Flash-Lite** generated answers and judged them through DeepEval; production
HF/Cloudflare provider settings were not changed. Six isolated workers completed
150 captures and scores in **10.45 minutes**, with **150/150 successful requests**.

Of the **130 regular questions**, 102 receive factual correctness, completeness
and answer-relevancy scoring; the other 28 use behavior rubrics for clarification,
emergencies, uncertainty and dynamic information without fixed factual gold.
Faithfulness applies only when usable generator evidence is present. The remaining
20 designated safety cases receive toxicity (10), scope adherence (5), or
response-level PII (5). Contextual precision and recall are excluded.

Fresh retrieval in this evaluation filters the existing up-to-14 selected records
using **combined ranking score >=0.8**, then retains at most **5 publications and
6 trials** without refilling below the cutoff. This is a query-relative,
unvalidated score—not an 80% relevance probability. Full selected record text is
used without the earlier character truncation; saved history retains its existing
bounds and is not rescored by this retrieval cutoff. This selection override is
**evaluation-only**, not a change to production record selection.

### Answer quality and safety

Pass threshold: **0.7**. Means are judge scores, not percentages of clinically
correct answers. Inapplicable metrics are excluded rather than counted as zero.

| Metric | Scored | Passed | Pass rate | Mean score |
|---|---:|---:|---:|---:|
| Correctness | 102 | 84 | 82.4% | 0.852 |
| Completeness | 102 | 86 | 84.3% | 0.836 |
| Answer relevancy | 102 | 92 | 90.2% | 0.904 |
| Faithfulness | 104 | 104 | 100.0% | 0.994 |
| Behavior: appropriate clarification, uncertainty or urgent response | 28 | 22 | 78.6% | 0.811 |
| Toxicity: higher is better | 10 | 10 | 100.0% | 1.000 |
| Scope adherence | 5 | 4 | 80.0% | 0.860 |
| Response-level PII | 5 | 5 | 100.0% | 1.000 |

**116/150 cases passed all their applicable metrics; 34 failed.** Among the 102
factual-correctness cases, 54 received 1.0. The **82.4% correctness pass rate** is
84/102 at the provisional threshold—not accuracy across all 150 questions and
not independent clinical verification.

### Latency, time to first token and generation cost

| Measurement | p50 | p95 | p99 | Mean |
|---|---:|---:|---:|---:|
| Request latency | 9.61s | 15.87s | 23.33s | 8.36s |
| Application-stream TTFT | 8.86s | 14.70s | 22.56s | 7.76s |
| Answer-generation cost per query (USD) | $0.001634 | $0.002652 | $0.004098 | $0.001518 |

Total answer-generation cost: **$0.2276394 for 150 questions**, with no missing
cost measurements. This includes answer input/output/thinking usage, excluding
query expansion, judging, routing, embeddings, ranking and hosting. Cost uses the
2026-11-01 recorded Gemini rates, not a complete serving-cost invoice.

Timing starts before routing/retrieval and excludes initialization and judging.
TTFT means the first nonempty application-stream token or direct-answer chunk;
structured JSON tokens are not necessarily rendered answer words. These are
six-worker local in-process measurements, not browser latency or a capacity test.
Maximum actual generation input was **12,022 tokens**; this includes instructions,
question, history and schema, not only record text.

### Component-level timing: repeated single question

A separate diagnostic ran P001 ("My doctor said I have prediabetes.
Does that mean I already have diabetes?") **10 times sequentially in one process**.
All ten succeeded. Runs **1-2 are excluded**; the table contains arithmetic means
from **runs 3-10 only**.

| Step | Mean time, runs 3-10 |
|---|---:|
| Query/model routing (Jev) | 0.580s |
| Query expansion | 1.261s |
| Parallel source retrieval | 3.616s |
| Normalization and deduplication | 0.019s |
| Hybrid ranking, embeddings and reranking | 2.151s |
| Context building | <0.001s |
| Answer generation and streaming | 2.808s |
| Response assembly | <0.001s |
| **Total request time** | **10.438s** |

Mean application-stream TTFT for runs 3-10: **9.432s**. The first measured request
took **19.690s**, excluding initialization before the request timer. Stage timers
are rounded to milliseconds, so displayed zero values do not imply zero work;
rounding and request overhead can produce a small difference from the total.
Answer caching was disabled; embedding-cache behavior was uncontrolled, so repeats
can benefit from warmed embeddings and clients. No judge calls were made. These
figures exclude Node/frontend/proxy transit and are not a load test.
Measurement artifacts: `backend-python/evals/artifacts/component-p001-10runs-20261101T182848Z/`.

### Reproducibility and limits

Latest full-run artifacts: `backend-python/evals/artifacts/full-150-20261101T181123Z/`.
See [evaluation instructions](backend-python/evals/README.md) for the opt-in harness.
Captures contain actual generator evidence; previous runs are not overwritten.

References and relevance thresholds are provisional and need independent clinical
review. The generator and judge use the same model; conversation tests use fixed
canonical history rather than actual-answer rollout. The harness bypasses Node,
authentication and browser flows. Response PII scores do not establish backend
privacy enforcement. Emergency guardrails cover explicit patterns, not comprehensive
clinical triage. **These results do not establish clinical readiness.**

## Load testing & capacity

**Local k6, 2026-11-02.** Real Express/FastAPI and configured Postgres/Redis; **60 async test-only job slots**, queue setting **50**, and a **10.438s simulated AI/retrieval delay**. Each user logs in, browses a session, submits one job and polls every 2s with Retry-After backoff (180s deadline). Each level ramps 30s and holds 180s, then jobs drain. Local shared-IP login allowance: 10,000; chat limits apply to submissions, not polling. **Production remains at its configured worker count, not 60.** This low-memory laptop run overlapped demo services/material preparation, so it is not a clean production-capacity benchmark.

<table>
<thead>
<tr><th rowspan="2">Users</th><th rowspan="2">Requests</th><th rowspan="2">HTTP errors</th><th colspan="3">Login (s)</th><th colspan="3">Sessions (s)</th><th colspan="3">Submit (s)</th></tr>
<tr><th>p50</th><th>p95</th><th>p99</th><th>p50</th><th>p95</th><th>p99</th><th>p50</th><th>p95</th><th>p99</th></tr>
</thead>
<tbody>
<tr><td>25</td><td>1,376</td><td>0.00%</td><td>0.30</td><td>1.54</td><td>2.03</td><td>0.24</td><td>0.37</td><td>1.71</td><td>0.29</td><td>0.54</td><td>0.91</td></tr>
<tr><td>50</td><td>2,760</td><td>0.00%</td><td>0.29</td><td>1.71</td><td>1.78</td><td>0.23</td><td>0.29</td><td>1.62</td><td>0.29</td><td>0.33</td><td>0.34</td></tr>
<tr><td>75</td><td>4,130</td><td>0.00%</td><td>0.29</td><td>1.68</td><td>1.88</td><td>0.23</td><td>0.30</td><td>0.47</td><td>0.30</td><td>0.34</td><td>0.38</td></tr>
<tr><td>100</td><td>5,410</td><td>0.00%</td><td>0.30</td><td>1.74</td><td>2.10</td><td>0.24</td><td>0.56</td><td>1.28</td><td>0.30</td><td>0.40</td><td>0.44</td></tr>
<tr><td>150</td><td>6,648</td><td>0.02%</td><td>0.43</td><td>1.79</td><td>2.14</td><td>1.08</td><td>1.89</td><td>2.48</td><td>0.43</td><td>2.04</td><td>2.13</td></tr>
</tbody>
</table>

<table>
<thead>
<tr><th rowspan="2">Users</th><th colspan="3">Accepted jobs</th><th colspan="2">Job ID received</th><th colspan="3">Queue wait (s)</th></tr>
<tr><th>Created</th><th>Succeeded</th><th>Failed</th><th>Clients</th><th>Expected</th><th>p50</th><th>p95</th><th>p99</th></tr>
</thead>
<tbody>
<tr><td>25</td><td>25</td><td>25</td><td>0</td><td>25</td><td>25</td><td>0.03</td><td>0.06</td><td>0.07</td></tr>
<tr><td>50</td><td>50</td><td>50</td><td>0</td><td>50</td><td>50</td><td>0.03</td><td>0.07</td><td>0.08</td></tr>
<tr><td>75</td><td>75</td><td>75</td><td>0</td><td>75</td><td>75</td><td>0.05</td><td>0.08</td><td>0.09</td></tr>
<tr><td>100</td><td>100</td><td>100</td><td>0</td><td>100</td><td>100</td><td>0.05</td><td>0.08</td><td>0.09</td></tr>
<tr><td>150</td><td>150</td><td>150</td><td>0</td><td>150</td><td>150</td><td>0.05</td><td>0.61</td><td>0.98</td></tr>
</tbody>
</table>

**All five levels passed the configured thresholds; all 400 accepted jobs succeeded.** HTTP errors were zero through 100 users and 0.02% at 150. Queue wait is server-observed, not end-to-end latency. The 60-slot profile matches ecommerce's aggregate concurrency, not its three-process topology; simulated waits do not reproduce real inference resource demands. No live-model capacity, SSE TTFT or spike/recovery claim is made.

Harness: [`load-tests/`](load-tests/README.md). Raw results and the run manifest stay in ignored `load-tests/results/`; test accounts remain local for reuse.

## Deployment

Deployed on Render (free tier) with zero monthly cost:

| Service | URL |
|---------|-----|
| Frontend | `curalink-medical-assistant-frontend.onrender.com` |
| Express API | `curalink-medical-assistant.onrender.com` |
| FastAPI Orchestrator | `curalink-medical-assistant-python.onrender.com` |
| Database | Neon Postgres (managed, free tier) |

> **Note:** Free-tier services spin down after ~15 min of inactivity. First request after spin-down takes 30-60 seconds (cold start). Ping `/health` on all services before demo.

## API Endpoints

### Express (Node) — User-Facing

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/auth/signup` | Create account |
| `POST` | `/api/auth/login` | Login, returns JWT |
| `POST` | `/api/session` | Create new session with intake form data |
| `GET` | `/api/sessions` | List all sessions for user |
| `POST` | `/api/chat/stream` | Send message, streams SSE pipeline response |
| `DELETE` | `/api/account` | Delete user account + all sessions + messages |
| `POST` | `/api/jobs` | Submit async pipeline job (returns 202) |
| `GET` | `/api/jobs/:id` | Poll job status + result |
| `DELETE` | `/api/jobs/:id` | Cancel a running job |
| `POST` | `/api/webhooks` | Register a webhook (job.completed/job.failed) |
| `GET` | `/api/webhooks` | List registered webhooks |
| `DELETE` | `/api/webhooks/:id` | Delete a webhook |
| `GET` | `/health` | Health check (includes queue_depth) |

### FastAPI (Python) — Internal Orchestrator

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/pipeline/run` | Full pipeline, returns JSON |
| `POST` | `/pipeline/stream` | Full pipeline with SSE streaming |
| `GET` | `/health` | Health check (includes queue_depth) |
| `POST` | `/jobs` | Submit async pipeline job (202 + job_id) |
| `GET` | `/jobs/{id}` | Get job status + result |
| `DELETE` | `/jobs/{id}` | Cancel a job |
| `GET` | `/jobs/{id}/events` | Replay SSE events (Last-Event-ID support) |
| `GET` | `/debug/fetch` | Debug: retrieval + normalization only |
| `GET` | `/debug/rank` | Debug: retrieval + ranking |
| `GET` | `/llm-ping` | Test LLM connectivity |

## Key Design Decisions

- **Thin Express, fat FastAPI** — routing and DB in Node; retrieval, ranking, generation, and async jobs in Python
- **Live-source RAG on cache misses** — uncached queries retrieve current PubMed, OpenAlex, and ClinicalTrials.gov records; tenant-isolated caches accelerate repeats
- **Request-contained synchronous pipeline** — synchronous context travels with each request; asynchronous job state and checkpoints live in Redis while queue workers are process-local
- **RRF over linear combination** — BM25 and cosine scores live on different scales; RRF uses rank position only, sidesteps normalization
- **Cite-or-abstain intent** — prompts instruct the model to cite retrieved documents and abstain when they are insufficient; users must still verify the original sources
- **Tenant-isolated result cache** — `query:<userId>:SHA-256(user|disease|intent|location|message|history)` with 24h TTL and Redis-to-Postgres fallback
- **Public-beta boundary** — de-identified research assistance only; no diagnosis, prescribing, treatment selection, trial-eligibility determination, or emergency assessment

## 📜 License

MIT License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.