import asyncio
import os
import time
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pinecone import Pinecone

from llm_backend import get_llm_backend
from embeddings.embedder import Embedder
from sources.pubmed import fetch_pubmed
from sources.openalex import fetch_openalex
from sources.trials import fetch_trials
from sources.normalizer import (
    normalize_pubmed,
    normalize_openalex,
    normalize_trial,
)
from sources.merger import merge_and_dedupe, filter_complete
from sources.geocode import geocode
from schemas.document import Document
from pinecone_store import prepare_records, embed_records, upsert_records, NAMESPACE
from ranking.cross_encoder import MedCPTReranker
from ranking.ranker import run_ranking
from stages.query_expander import expand_query
from stages.context_builder import build_context
from stages.llm_reasoner import run_reasoner
from stages.response_assembler import assemble_response

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "curalink")
BIENCODER_MODEL = os.getenv("BIENCODER_MODEL", "pritamdeka/S-PubMedBert-MS-MARCO")

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY not set in .env")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

llm = get_llm_backend()

# Model holder. With HF Inference API these are lightweight API wrappers,
# not loaded model weights. Instantiated at startup for consistency.
models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[startup] Initializing bi-encoder (HF API): {BIENCODER_MODEL}")
    models["embedder"] = Embedder(BIENCODER_MODEL)
    print(f"[startup] Bi-encoder ready (dim={models['embedder'].dim})")

    print("[startup] Initializing cross-encoder (HF API): ncbi/MedCPT-Cross-Encoder")
    models["reranker"] = MedCPTReranker()
    print("[startup] Cross-encoder ready")

    print(f"[startup] LLM backend: {llm.__class__.__name__}")
    yield
    models.clear()


app = FastAPI(title="Curalink Orchestrator", lifespan=lifespan)


class EmbedRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to embed")


class EmbedBatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=500)
    return_vectors: bool = Field(
        False, description="If false, return only dims + timing, not the vectors"
    )
    batch_size: int = Field(32, ge=1, le=128)


@app.get("/health")
async def health():
    return {"ok": True, "service": "fastapi"}


@app.get("/pinecone-ping")
async def pinecone_ping():
    try:
        stats = index.describe_index_stats()
        return {"ok": True, "index": PINECONE_INDEX, "stats": stats.to_dict()}
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={"ok": False, "error": "pinecone unreachable", "detail": str(e)},
        )


@app.post("/embed")
async def embed(req: EmbedRequest):
    """
    Return a 768-dim embedding for the given text using PubMedBERT-MS-MARCO.
    Vector is L2-normalized so cosine similarity = dot product downstream.
    """
    embedder: Embedder | None = models.get("embedder")
    if embedder is None:
        raise HTTPException(status_code=503, detail="embedder not loaded")

    vec = embedder.embed_text(req.text)
    return {
        "ok": True,
        "model": BIENCODER_MODEL,
        "dim": len(vec),
        "sample": vec[:5],
    }


@app.post("/embed/batch")
async def embed_batch(req: EmbedBatchRequest):
    """
    Batch-embed a list of texts. Returns timing + per-doc dim.
    Set `return_vectors=true` to get the full vectors in the response
    (makes the response large).
    """
    embedder: Embedder | None = models.get("embedder")
    if embedder is None:
        raise HTTPException(status_code=503, detail="embedder not loaded")

    t0 = time.perf_counter()
    vecs = embedder.embed_batch(req.texts, batch_size=req.batch_size)
    dt = time.perf_counter() - t0

    response: dict = {
        "ok": True,
        "model": BIENCODER_MODEL,
        "count": len(vecs),
        "dim": embedder.dim,
        "batch_size": req.batch_size,
        "timing_ms": round(dt * 1000),
        "throughput_per_sec": round(len(vecs) / dt, 1) if dt > 0 else None,
    }
    if req.return_vectors:
        response["vectors"] = vecs
    return response


@app.get("/llm-ping")
async def llm_ping():
    try:
        response = await llm.generate(
            "say hi in one word",
            max_tokens=10,
            temperature=0.2,
        )
        return {
            "ok": True,
            "backend": llm.__class__.__name__,
            "response": response.strip(),
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={"ok": False, "error": "llm unreachable", "detail": str(e)},
        )


@app.get("/debug/fetch")
async def debug_fetch(
    disease: str = Query(..., description="Primary disease (e.g. 'parkinson')"),
    query: str = Query(..., description="Full free-text query for pubmed+openalex"),
    pubmed_limit: int = Query(25, ge=1, le=100),
    openalex_limit: int = Query(25, ge=1, le=100),
    trials_limit: int = Query(10, ge=1, le=50),
    upsert: bool = Query(
        False, description="If true, embed + upsert complete docs to Pinecone"
    ),
    return_docs: bool = Query(
        False, description="If true, include full docs in response (large payload)"
    ),
):
    """
    Runs Phase 2 + optionally Phase 3 pipeline:
        fetch (3 sources) -> normalize -> merge+dedupe -> filter
        -> [optional: embed + upsert to Pinecone]
    """
    t0 = time.perf_counter()

    pubmed_raw, openalex_raw, trials_raw = await asyncio.gather(
        fetch_pubmed(query, limit=pubmed_limit),
        fetch_openalex(query, limit=openalex_limit),
        fetch_trials(disease=disease, limit=trials_limit),
        return_exceptions=True,
    )

    warnings: list[str] = []

    def _unwrap(result, name: str) -> list:
        if isinstance(result, Exception):
            warnings.append(f"{name}_failed: {type(result).__name__}: {result}")
            return []
        return result

    pubmed_raw = _unwrap(pubmed_raw, "pubmed")
    openalex_raw = _unwrap(openalex_raw, "openalex")
    trials_raw = _unwrap(trials_raw, "trials")

    t_fetch = time.perf_counter() - t0

    pubmed_docs = [normalize_pubmed(r) for r in pubmed_raw]
    openalex_docs = [normalize_openalex(r) for r in openalex_raw]
    trial_docs = [normalize_trial(r, disease_context=disease) for r in trials_raw]

    total_raw = len(pubmed_docs) + len(openalex_docs) + len(trial_docs)

    deduped = merge_and_dedupe([pubmed_docs, openalex_docs, trial_docs])
    complete = filter_complete(deduped)
    multi_source_count = sum(1 for d in deduped if len(d.sources) > 1)

    t_normalize = time.perf_counter() - t0

    # Optional: embed + upsert to Pinecone
    upsert_stats: dict | None = None
    if upsert:
        embedder: Embedder | None = models.get("embedder")
        if embedder is None:
            warnings.append("embedder_not_loaded: skipping upsert")
        else:
            try:
                records = prepare_records(complete)
                t_embed_start = time.perf_counter()
                embed_records(records, embedder)
                t_embed = time.perf_counter() - t_embed_start
                t_upsert_start = time.perf_counter()
                upserted = upsert_records(index, records)
                t_upsert = time.perf_counter() - t_upsert_start
                stats = index.describe_index_stats()
                ns_stats = (stats.namespaces or {}).get(NAMESPACE)
                ns_count = ns_stats.vector_count if ns_stats else 0
                upsert_stats = {
                    "records_prepared": len(records),
                    "records_upserted": upserted,
                    "embed_ms": round(t_embed * 1000),
                    "upsert_ms": round(t_upsert * 1000),
                    "namespace": NAMESPACE,
                    "pinecone_namespace_vectors": ns_count,
                }
            except Exception as e:
                warnings.append(f"upsert_failed: {type(e).__name__}: {e}")

    t_total = time.perf_counter() - t0

    response = {
        "ok": True,
        "query": query,
        "disease": disease,
        "source_stats": {
            "pubmed": len(pubmed_docs),
            "openalex": len(openalex_docs),
            "trials": len(trial_docs),
        },
        "counts": {
            "raw": total_raw,
            "after_dedupe": len(deduped),
            "after_is_complete_filter": len(complete),
            "multi_source": multi_source_count,
        },
        "timings_ms": {
            "fetch_parallel": round(t_fetch * 1000),
            "normalize_merge": round((t_normalize - t_fetch) * 1000),
            "total": round(t_total * 1000),
        },
        "warnings": warnings,
    }

    if upsert_stats:
        response["upsert"] = upsert_stats
    if return_docs:
        response["documents"] = [d.to_dict() for d in complete]

    return response


# ---------------------------------------------------------------------------
# Step 3.5 — Async write-back endpoint
# ---------------------------------------------------------------------------

class WritebackRequest(BaseModel):
    """Accepts a list of Document dicts and triggers background embed+upsert."""
    documents: list[dict] = Field(..., min_length=1)


def _bg_writeback(docs_dicts: list[dict]) -> None:
    """
    Background task: convert dicts -> Documents, prepare Pinecone records,
    embed, and upsert. Runs after the HTTP response is already sent.
    """
    embedder: Embedder | None = models.get("embedder")
    if embedder is None:
        print("[writeback] ERROR: embedder not loaded, skipping")
        return

    docs = []
    for d in docs_dicts:
        try:
            docs.append(Document(**d))
        except Exception as e:
            print(f"[writeback] WARN: skipping invalid doc: {e}")

    if not docs:
        print("[writeback] no valid docs, nothing to upsert")
        return

    t0 = time.perf_counter()
    records = prepare_records(docs)
    if not records:
        print("[writeback] no records after prepare, nothing to upsert")
        return

    embed_records(records, embedder)
    upserted = upsert_records(index, records)
    dt = time.perf_counter() - t0
    print(f"[writeback] upserted {upserted} records in {dt:.1f}s (background)")


@app.post("/writeback")
async def writeback(req: WritebackRequest, background_tasks: BackgroundTasks):
    """
    Accepts documents and triggers background embed+upsert to Pinecone.
    Returns immediately — the upsert runs after the response is sent.
    """
    background_tasks.add_task(_bg_writeback, req.documents)
    return {
        "ok": True,
        "queued": len(req.documents),
        "message": "writeback queued in background",
    }


# ---------------------------------------------------------------------------
# Phase 4 integration — /debug/rank endpoint
# ---------------------------------------------------------------------------

@app.get("/debug/rank")
async def debug_rank(
    disease: str = Query(..., description="Primary disease"),
    query: str = Query(..., description="Full free-text query"),
    pubmed_limit: int = Query(25, ge=1, le=100),
    openalex_limit: int = Query(25, ge=1, le=100),
    trials_limit: int = Query(10, ge=1, le=50),
    top_k: int = Query(8, ge=1, le=20),
):
    """
    Runs Phase 2 + Phase 3 + Phase 4 pipeline:
        fetch -> normalize -> merge+dedupe -> filter -> rank (full funnel)
    Returns the final top_k ranked docs with timing breakdown.
    """
    embedder = models.get("embedder")
    reranker = models.get("reranker")
    if embedder is None or reranker is None:
        raise HTTPException(status_code=503, detail="models not loaded")

    t0 = time.perf_counter()

    # Phase 2: fetch
    pubmed_raw, openalex_raw, trials_raw = await asyncio.gather(
        fetch_pubmed(query, limit=pubmed_limit),
        fetch_openalex(query, limit=openalex_limit),
        fetch_trials(disease=disease, limit=trials_limit),
        return_exceptions=True,
    )

    warnings: list[str] = []

    def _unwrap(result, name: str) -> list:
        if isinstance(result, Exception):
            warnings.append(f"{name}_failed: {type(result).__name__}: {result}")
            return []
        return result

    pubmed_raw = _unwrap(pubmed_raw, "pubmed")
    openalex_raw = _unwrap(openalex_raw, "openalex")
    trials_raw = _unwrap(trials_raw, "trials")

    t_fetch = time.perf_counter() - t0

    # Phase 3: normalize + merge + filter
    pubmed_docs = [normalize_pubmed(r) for r in pubmed_raw]
    openalex_docs = [normalize_openalex(r) for r in openalex_raw]
    trial_docs = [normalize_trial(r, disease_context=disease) for r in trials_raw]

    deduped = merge_and_dedupe([pubmed_docs, openalex_docs, trial_docs])
    complete = filter_complete(deduped)

    t_normalize = time.perf_counter() - t0

    # Phase 4: rank
    ranking_result = run_ranking(
        query=query,
        docs=complete,
        embedder=embedder,
        reranker=reranker,
        top_k=top_k,
    )

    t_total = time.perf_counter() - t0

    # Build response
    top_docs_summary = []
    for i, doc in enumerate(ranking_result.top_docs):
        top_docs_summary.append({
            "rank": i + 1,
            "doc_id": doc.doc_id,
            "doc_type": doc.doc_type,
            "title": doc.title,
            "year": doc.year,
            "sources": doc.sources,
            "url": doc.url,
        })

    return {
        "ok": True,
        "query": query,
        "disease": disease,
        "retrieval_counts": {
            "pubmed": len(pubmed_docs),
            "openalex": len(openalex_docs),
            "trials": len(trial_docs),
            "after_dedupe": len(deduped),
            "after_filter": len(complete),
        },
        "ranking": {
            "input": ranking_result.counts.get("input"),
            "after_rrf_top20": ranking_result.counts.get("after_rrf_top20"),
            "final": ranking_result.counts.get("after_mmr"),
        },
        "timings_ms": {
            "fetch_parallel": round(t_fetch * 1000),
            "normalize_merge": round((t_normalize - t_fetch) * 1000),
            **ranking_result.timings_ms,
            "total": round(t_total * 1000),
        },
        "top_docs": top_docs_summary,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# /pipeline/run — the production endpoint Express calls (inter-service contract)
# Also aliased as /debug/pipeline for direct testing.
# ---------------------------------------------------------------------------

class PipelineRequest(BaseModel):
    static: dict = Field(..., description="Static form context: disease, intent, location, patientName")
    dynamic: dict = Field(default_factory=dict, description="Chat history and entities")
    current: dict = Field(..., description="Current user message: {userMessage}")


@app.post("/pipeline/run")
async def pipeline_run(req: PipelineRequest, background_tasks: BackgroundTasks):
    """
    Non-streaming version of the pipeline. Returns a single JSON response.
    Same logic as /pipeline/stream but without SSE.
    """
    import json as _json

    embedder = models.get("embedder")
    reranker = models.get("reranker")
    if embedder is None or reranker is None:
        raise HTTPException(status_code=503, detail="models not loaded")

    stage_timings: dict = {}
    warnings: list[str] = []
    user_message = req.current.get("userMessage", "")
    chat_history = req.dynamic.get("recentMessages", [])

    t_pipeline = time.perf_counter()

    # Stage 1 — Query expansion
    t0 = time.perf_counter()
    expander_result = await expand_query(
        user_message=user_message,
        static_context=req.static,
        chat_history=chat_history,
        llm=llm,
    )
    stage_timings["query_expansion"] = round((time.perf_counter() - t0) * 1000)

    if expander_result.skip_retrieval:
        return {"skip_retrieval": True, "message": "Non-medical query"}

    # Stage 2 — Retrieval
    t0 = time.perf_counter()
    disease = req.static.get("disease", "")
    location_str = req.static.get("location", "")
    best_query = expander_result.expanded_queries[0]

    geo = await geocode(location_str) if location_str else None

    pubmed_raw, openalex_raw, trials_raw = await asyncio.gather(
        fetch_pubmed(best_query, limit=80),
        fetch_openalex(best_query, limit=80),
        fetch_trials(disease=disease, location=geo, limit=50),
        return_exceptions=True,
    )

    def _unwrap(result, name):
        if isinstance(result, Exception):
            warnings.append(f"{name}_failed: {type(result).__name__}: {result}")
            return []
        return result

    pubmed_raw = _unwrap(pubmed_raw, "pubmed")
    openalex_raw = _unwrap(openalex_raw, "openalex")
    trials_raw = _unwrap(trials_raw, "trials")
    stage_timings["retrieval"] = round((time.perf_counter() - t0) * 1000)

    retrieval_counts = {
        "pubmed": len(pubmed_raw),
        "openalex": len(openalex_raw),
        "trials": len(trials_raw),
    }

    # Stage 3 — Normalization
    t0 = time.perf_counter()
    pubmed_docs = [normalize_pubmed(r) for r in pubmed_raw]
    openalex_docs = [normalize_openalex(r) for r in openalex_raw]
    trial_docs = [normalize_trial(r, disease_context=disease) for r in trials_raw]

    deduped = merge_and_dedupe([pubmed_docs, openalex_docs, trial_docs])
    complete = filter_complete(deduped)
    stage_timings["normalization"] = round((time.perf_counter() - t0) * 1000)
    retrieval_counts["after_dedupe"] = len(deduped)
    retrieval_counts["after_filter"] = len(complete)

    if not complete:
        return {"error": "No documents retrieved"}

    # Stage 4 — Ranking
    t0 = time.perf_counter()
    ranking_result = run_ranking(
        query=best_query, docs=complete,
        embedder=embedder, reranker=reranker, top_k=14,
    )
    stage_timings["ranking"] = round((time.perf_counter() - t0) * 1000)
    retrieval_counts["after_ranking"] = len(ranking_result.top_docs)

    # Stage 5 — Context build
    t0 = time.perf_counter()
    payload = build_context(
        top_docs=ranking_result.top_docs,
        user_message=user_message,
        static_context=req.static,
        chat_history=chat_history,
    )
    stage_timings["context_build"] = round((time.perf_counter() - t0) * 1000)

    # Stage 6 — LLM reasoning
    t0 = time.perf_counter()
    reasoner_output = await run_reasoner(payload, llm=llm, max_tokens=1200)
    stage_timings["llm"] = round((time.perf_counter() - t0) * 1000)

    # Stage 7 — Assembly
    t0 = time.perf_counter()
    assembled = assemble_response(
        llm_output=reasoner_output.llm_output,
        doc_anchors=payload.doc_anchors,
        stage_timings=stage_timings,
        retrieval_counts=retrieval_counts,
    )
    stage_timings["assembly"] = round((time.perf_counter() - t0) * 1000)
    stage_timings["total"] = round((time.perf_counter() - t_pipeline) * 1000)
    assembled.user_facing_json["pipelineMeta"]["stage_timings_ms"] = stage_timings
    assembled.user_facing_json["pipelineMeta"]["warnings"] = warnings + assembled.warnings

    # Background writeback
    complete_dicts = [d.to_dict() for d in complete]
    background_tasks.add_task(_bg_writeback, complete_dicts)

    return assembled.user_facing_json


@app.post("/pipeline/stream")
async def pipeline_stream(req: PipelineRequest, background_tasks: BackgroundTasks):
    """
    SSE streaming version of /pipeline/run.
    Events:
      event: status   -> pipeline stage progress
      event: token    -> streamed LLM tokens
      event: metadata -> final assembled response JSON
      event: done     -> close signal
    """
    import json as _json

    embedder = models.get("embedder")
    reranker = models.get("reranker")
    if embedder is None or reranker is None:
        raise HTTPException(status_code=503, detail="models not loaded")

    async def event_generator():
        stage_timings: dict = {}
        warnings: list[str] = []
        user_message = req.current.get("userMessage", "")
        chat_history = req.dynamic.get("recentMessages", [])

        t_pipeline = time.perf_counter()

        # Stage 1
        yield f"event: status\ndata: {{\"stage\":\"query_expansion\",\"message\":\"Expanding query...\"}}\n\n"
        t0 = time.perf_counter()
        expander_result = await expand_query(
            user_message=user_message,
            static_context=req.static,
            chat_history=chat_history,
            llm=llm,
        )
        stage_timings["query_expansion"] = round((time.perf_counter() - t0) * 1000)

        if expander_result.skip_retrieval:
            yield f"event: metadata\ndata: {{\"skip_retrieval\":true,\"message\":\"Non-medical query\"}}\n\n"
            yield "event: done\ndata: {}\n\n"
            return

        # Stage 2
        yield f"event: status\ndata: {{\"stage\":\"retrieval\",\"message\":\"Fetching from PubMed, OpenAlex, ClinicalTrials...\"}}\n\n"
        t0 = time.perf_counter()
        disease = req.static.get("disease", "")
        location_str = req.static.get("location", "")
        best_query = expander_result.expanded_queries[0]

        # Geocode location for trial geo-filter
        geo = await geocode(location_str) if location_str else None

        pubmed_raw, openalex_raw, trials_raw = await asyncio.gather(
            fetch_pubmed(best_query, limit=80),
            fetch_openalex(best_query, limit=80),
            fetch_trials(disease=disease, location=geo, limit=50),
            return_exceptions=True,
        )

        def _unwrap(result, name):
            if isinstance(result, Exception):
                warnings.append(f"{name}_failed: {type(result).__name__}: {result}")
                return []
            return result

        pubmed_raw = _unwrap(pubmed_raw, "pubmed")
        openalex_raw = _unwrap(openalex_raw, "openalex")
        trials_raw = _unwrap(trials_raw, "trials")
        stage_timings["retrieval"] = round((time.perf_counter() - t0) * 1000)

        retrieval_counts = {
            "pubmed": len(pubmed_raw),
            "openalex": len(openalex_raw),
            "trials": len(trials_raw),
        }

        # Send retrieval counts so frontend can show them during loading
        yield f"event: status\ndata: {{\"stage\":\"normalization\",\"message\":\"Normalizing {len(pubmed_raw)+len(openalex_raw)+len(trials_raw)} documents...\",\"retrieval_counts\":{{\"pubmed\":{len(pubmed_raw)},\"openalex\":{len(openalex_raw)},\"trials\":{len(trials_raw)}}}}}\n\n"
        t0 = time.perf_counter()
        pubmed_docs = [normalize_pubmed(r) for r in pubmed_raw]
        openalex_docs = [normalize_openalex(r) for r in openalex_raw]
        trial_docs = [normalize_trial(r, disease_context=disease) for r in trials_raw]

        deduped = merge_and_dedupe([pubmed_docs, openalex_docs, trial_docs])
        complete = filter_complete(deduped)
        stage_timings["normalization"] = round((time.perf_counter() - t0) * 1000)
        retrieval_counts["after_dedupe"] = len(deduped)
        retrieval_counts["after_filter"] = len(complete)

        if not complete:
            yield f"event: metadata\ndata: {{\"error\":\"No documents retrieved\"}}\n\n"
            yield "event: done\ndata: {}\n\n"
            return

        # Stage 4
        yield f"event: status\ndata: {{\"stage\":\"ranking\",\"message\":\"BM25 filtering {len(complete)} docs → top 20 → embedding → RRF → MedCPT → source-balanced selection → top 10\"}}\n\n"
        t0 = time.perf_counter()
        ranking_result = run_ranking(
            query=best_query, docs=complete,
            embedder=embedder, reranker=reranker, top_k=14,
        )
        stage_timings["ranking"] = round((time.perf_counter() - t0) * 1000)
        retrieval_counts["after_ranking"] = len(ranking_result.top_docs)

        # Stage 5
        yield f"event: status\ndata: {{\"stage\":\"context_build\",\"message\":\"Building context for LLM...\"}}\n\n"
        t0 = time.perf_counter()
        payload = build_context(
            top_docs=ranking_result.top_docs,
            user_message=user_message,
            static_context=req.static,
            chat_history=chat_history,
        )
        stage_timings["context_build"] = round((time.perf_counter() - t0) * 1000)

        # Stage 6 — stream LLM tokens
        yield f"event: status\ndata: {{\"stage\":\"llm\",\"message\":\"Generating response...\"}}\n\n"
        t0 = time.perf_counter()
        full_text = ""
        try:
            async for token in llm.generate_stream(
                payload.user_prompt,
                system_prompt=payload.system_prompt,
                max_tokens=1200,
                temperature=0.2,
                json_mode=True,
            ):
                full_text += token
                escaped = _json.dumps(token)
                yield f"event: token\ndata: {escaped}\n\n"
        except Exception as e:
            warnings.append(f"llm_stream_error: {e}")

        stage_timings["llm"] = round((time.perf_counter() - t0) * 1000)

        # Parse LLM output
        try:
            from stages.llm_reasoner import _parse_llm_response, _validate_schema, _fallback_output
            parsed = _parse_llm_response(full_text)
            issues = _validate_schema(parsed)
            parsed.setdefault("overview", "")
            parsed.setdefault("insights", [])
            parsed.setdefault("trials", [])
            parsed.setdefault("abstain_reason", None)
        except Exception as e:
            parsed = _fallback_output(str(e))
            warnings.append(f"llm_parse_error: {e}")

        # Stage 7
        t0 = time.perf_counter()
        assembled = assemble_response(
            llm_output=parsed,
            doc_anchors=payload.doc_anchors,
            stage_timings=stage_timings,
            retrieval_counts=retrieval_counts,
        )
        stage_timings["assembly"] = round((time.perf_counter() - t0) * 1000)
        stage_timings["total"] = round((time.perf_counter() - t_pipeline) * 1000)
        assembled.user_facing_json["pipelineMeta"]["stage_timings_ms"] = stage_timings
        assembled.user_facing_json["pipelineMeta"]["warnings"] = warnings + assembled.warnings

        # Send final metadata
        meta_json = _json.dumps(assembled.user_facing_json)
        yield f"event: metadata\ndata: {meta_json}\n\n"
        yield "event: done\ndata: {}\n\n"

        # Background writeback
        complete_dicts = [d.to_dict() for d in complete]
        background_tasks.add_task(_bg_writeback, complete_dicts)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
