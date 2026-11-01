"""Opt-in offline-dataset capture through the Python pipeline; no HTTP servers or workers."""

import asyncio
import json
import time
from contextlib import ExitStack
from unittest.mock import patch
from uuid import uuid4

from .dataset import question_input


def answer_text(response):
    parts = [response.get("overview", "")]
    for key, field in (("insights", "finding"), ("trials", "relevance"), ("recommendations", "text")):
        parts.extend(item.get(field, "") for item in response.get(key, []) if isinstance(item, dict))
    if response.get("abstain_reason"):
        parts.append(response["abstain_reason"])
    return "\n".join(part for part in parts if isinstance(part, str) and part.strip())


def event_data(frame):
    if isinstance(frame, bytes):
        frame = frame.decode("utf-8")
    event = "message"
    data = []
    for line in frame.splitlines():
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].strip())
    return event, json.loads("\n".join(data)) if data else None


def select_threshold_records(result):
    """Filter the selected records without refilling below the provisional cutoff."""
    passing = sorted((doc for doc in result.top_docs
                      if result.scores_by_doc_id.get(doc.doc_id, -1) >= 0.8),
                     key=lambda doc: result.scores_by_doc_id[doc.doc_id], reverse=True)
    counts = {"publication": 0, "trial": 0}
    selected = []
    for doc in passing:
        cap = 5 if doc.doc_type == "publication" else 6
        if counts[doc.doc_type] < cap:
            selected.append(doc)
            counts[doc.doc_type] += 1
    return selected


async def collect(cases, output, timeout_seconds=300, generation_backend=None, threshold_selection=False):
    # Imports initialize provider clients, so they belong behind the CLI's explicit opt-in.
    import main as app
    import stages.history_context as history_context
    from starlette.requests import Request
    from stages.context_builder import _format_publication, _format_trial

    if not app.routing_enabled():
        raise ValueError("JEV_API_KEY is required to evaluate routing; no silent single-model baseline")
    await app._load_models()
    if not {"embedder", "reranker"}.issubset(app.models):
        raise RuntimeError("Evaluation model clients could not initialize")
    original_rank = app.run_ranking
    original_build = app.build_context
    tenant = "eval-" + uuid4().hex
    with ExitStack() as runtime:
        runtime.enter_context(patch.object(app.llm, "use_cache", False))
        if generation_backend is not None:
            runtime.enter_context(patch("routing.get_tier_backend", lambda base, tier: generation_backend))
        for case in cases:
            usage_start = len(generation_backend.usage) if generation_backend is not None else 0
            row = {"case_id": case["id"], "scope": "python_orchestrator_in_process",
                   "history_mode": "canonical_prior_turns", "retrieval_k": 11 if threshold_selection else 5,
                   "configuration": {"environment": "local_in_process", "concurrency": 1,
                                     "answer_cache": "disabled", "embedding_cache": "uncontrolled"},
                   "cost_per_query_usd": None, "cost_status": "unknown_provider_usage_not_exposed",
                   "user_visible_ttft_seconds": None, "generator_context": [],
                   "retrieval_context": [], "context_capture_complete": True,
                   "actual_output": "", "routing": {}}
            if threshold_selection:
                row["selection_policy"] = {"score": "query_relative_combined", "minimum": 0.8,
                                           "max_publications": 5, "max_trials": 6}
            contexts = []
            measurements = []

            def rank_at_five(*args, **kwargs):
                kwargs["top_k"] = 14 if threshold_selection else 5
                result = original_rank(*args, **kwargs)
                if threshold_selection:
                    result.top_docs = select_threshold_records(result)
                    row["selected_record_scores"] = {doc.doc_id: result.scores_by_doc_id[doc.doc_id]
                                                     for doc in result.top_docs}
                return result

            def capture_context(*args, **kwargs):
                args = list(args)
                if args:
                    args[0] = args[0][:11 if threshold_selection else 5]
                else:
                    kwargs["top_docs"] = kwargs["top_docs"][:11 if threshold_selection else 5]
                payload = original_build(*args, **kwargs)
                texts = []
                for anchor, doc in payload.doc_anchors.items():
                    format_doc = _format_trial if doc.doc_type == "trial" else _format_publication
                    texts.append(format_doc(doc, anchor))
                contexts.append(texts)
                record_chars = len("\n\n".join(texts))
                measurements.append({"records": len(texts), "record_characters": record_chars,
                                     "record_tokens_estimate": (record_chars + 3) // 4,
                                     "total_prompt_tokens_estimate": payload.token_count,
                                     "estimate_method": "characters_divided_by_four; not actual tokenizer counts"})
                return payload

            static, history = question_input(case)
            req = app.PipelineRequest(tenant=tenant, static=static,
                                      dynamic={"recentMessages": history},
                                      current={"userMessage": case["question"]})
            request = Request({"type": "http", "method": "POST", "path": "/pipeline/stream",
                               "headers": [(b"x-request-id", case["id"].encode("utf-8"))]})
            started = time.perf_counter()
            state: dict = {"first_token": None, "first_chunk": None, "done": False, "response": None}

            async def consume():
                stream = await app.pipeline_stream(req, request)
                async for frame in stream.body_iterator:
                    event, data = event_data(frame)
                    elapsed = time.perf_counter() - started
                    if event == "token" and isinstance(data, str) and data.strip() and state["first_token"] is None:
                        state["first_token"] = elapsed
                    elif event == "metadata":
                        state["response"] = data
                        if isinstance(data, dict) and answer_text(data) and state["first_chunk"] is None:
                            state["first_chunk"] = elapsed
                    elif event == "done":
                        state["done"] = True

            try:
                with ExitStack() as stack:
                    stack.enter_context(patch.object(app, "run_ranking", rank_at_five))
                    stack.enter_context(patch.object(app, "build_context", capture_context))
                    stack.enter_context(patch.object(history_context, "build_context", capture_context))
                    await asyncio.wait_for(consume(), timeout=timeout_seconds)
                response = state["response"]
                if not state["done"] or not isinstance(response, dict):
                    raise RuntimeError("Incomplete pipeline event stream")
                row["actual_output"] = answer_text(response)
                meta = response.get("pipelineMeta", {})
                routing = meta.get("routing", {})
                row["routing"] = {key: routing[key] for key in (
                    "route", "topic", "model_tier", "answer_model", "retrieval_used", "fallback"
                ) if key in routing}
                row["stage_timings_ms"] = meta.get("stage_timings_ms", {})
                row["generator_context"] = contexts[-1] if contexts else []
                if routing.get("retrieval_used"):
                    row["retrieval_context"] = row["generator_context"]
                warnings = meta.get("warnings", [])
                row["warning_count"] = len(warnings)
                # A damaged stream must not become a successful low-latency sample.
                failed = bool(response.get("error")) or not row["actual_output"] or any(
                    str(warning).startswith(("llm_stream_error", "llm_parse_error")) for warning in warnings
                )
                row["status"] = "error" if failed else "success"
                if not failed and routing.get("route") in {"rag", "conversation"} and not contexts:
                    row.update(status="error", reason="Missing actual generation-context capture")
            except asyncio.TimeoutError:
                row.update(status="timeout", reason="async_deadline_exceeded")
            except Exception as exc:
                row.update(status="error", error_type=type(exc).__name__)
            row["context_measurements"] = measurements[-1] if measurements else None
            if generation_backend is not None:
                from .gemini_backend import answer_generation_cost, RATES
                row["configuration"]["generation_override"] = generation_backend.model
                usage = generation_backend.usage[usage_start:]
                row["answer_generation_usage"] = [record for record in usage if record.get("purpose") == "answer"]
                row["cost_per_query_usd"] = answer_generation_cost(usage)
                row["cost_status"] = "estimated" if row["cost_per_query_usd"] is not None else "unknown_usage"
                row["cost_scope"] = "answer_generation_only"
                row["cost_provenance"] = RATES
            row["latency_seconds"] = time.perf_counter() - started
            row["model_stream_ttft_seconds"] = state["first_token"]
            row["first_answer_chunk_seconds"] = state["first_chunk"]
            row["ttft_seconds"] = state["first_token"] if state["first_token"] is not None else state["first_chunk"]
            row["ttft_scope"] = "application_stream_first_token_or_direct_answer"
            row["ttft_note"] = "Request start to first nonempty application token, matching e-commerce; structured JSON tokens are not browser-rendered answer text"
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
