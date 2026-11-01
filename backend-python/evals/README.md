# Offline golden-dataset evaluation

## Scope

Evaluate the **150 fixed patient/caregiver questions** through the Python pipeline, using the same general approach as the sibling e-commerce evaluator. The dataset ID is `curalink-patient-golden-v2`, stored in `../../evals/golden/v1`.

- **130 regular questions:** applicable answer-quality metrics, no dedicated safety metrics.
- **20 designated safety questions (P071–P090):** 10 toxicity, 5 scope adherence, 5 response-level PII checks; no regular quality metrics.
- **All questions:** capture latency and application-stream TTFT; report **p50, p95, p99 and mean** for completed responses, with failures/timeouts counted separately.

This is returned-answer evaluation, not an authentication, database-isolation or backend-privacy audit. It does not start Node/ASGI test servers, create synthetic accounts or inspect outbound traffic. Emergency scenarios test the generated answer's urgency guidance, not the Node emergency intercept. PII scores describe the returned text only.

“Offline” means a fixed dataset—not disconnected execution. Collection uses live retrieval/model APIs; Gemini judging also makes paid API calls.

## Golden dataset

| Coverage | Questions |
|---|---:|
| Symptoms and understanding diagnoses | 15 |
| Treatment options and lifestyle | 20 |
| Side effects and medicine concerns | 10 |
| Clinical trials and participation | 10 |
| Emergencies, ambiguity, misinformation and everyday interaction | 15 |
| Targeted safety | 20 |
| Conversations: 15 threads × 4 questions | 60 |
| **Total** | **150** |

Difficulty: 50 easy, 58 medium, 42 hard. All questions are for testing; there is no training/tuning partition. Patient prompts do not require paper identifiers or trial-statistics recall.

Source-backed reference answers remain drafts pending independent clinical review. The annotation library has 13 patient-information sources plus ten bounded historical trial abstracts. Source review dates matter; a historical trial is not automatically current prescribing guidance. Six live-registry questions have behavioral expectations but no frozen factual reference yet.

Conversations use only preceding canonical answers and explicitly allowed evidence, bounded to six prior messages and 800-character source snippets. This is not actual-answer rollout testing. Never inject current or future gold answers into the tested prompt.

## Scoring

Regular questions use correctness/completeness GEval, AnswerRelevancyMetric and FaithfulnessMetric where applicable. Contextual precision and recall are excluded from current runs. Clarification, urgent-response and other policy-only questions use their behavioral rubric when factual/retrieval metrics do not apply.

Safety cases automatically select their relevant metric: ToxicityMetric, scope-adherence GEval or PIILeakageMetric. In pinned DeepEval 4.2.9, higher safety scores are better. The optional legacy `--include-safety` flag does not add safety metrics to regular questions.

The authorized concurrent rerun uses `collect(..., threshold_selection=True)`: after the existing up-to-14 selection, retain records with query-relative combined score **>=0.8**, globally ordered by that score, capped at **5 publications and 6 trials**. Do not refill below threshold. Saved history evidence is not rescored by this fresh-retrieval cutoff. The threshold is experimental, not a calibrated relevance probability. CLI collection retains its legacy k=5 default unless this Python option is supplied. Selected-record text is no longer cut to 1,600 characters; abstracts and trial eligibility are retained, along with all formatted authors, primary outcomes and available locations. Conversation-history bounds remain unchanged. Captures record character-based context token estimates and provider-reported actual whole-prompt input tokens. Faithfulness uses actual supplied evidence. Inapplicable metrics remain N/A. The separate 0.7 answer-quality pass threshold is provisional, not a medical release gate.

## Commands

Run from `backend-python`. Offline regression tests do not call application providers or the judge:

```text
python -m unittest evals.test_eval
```

Future live runs require explicit authorization. Examples, not a request to run:

```text
python -m evals.run_eval collect --ids P001 --output evals/artifacts/new-p001-captures.jsonl --generation-model gemini-3.5-flash-lite --allow-live-requests
python -m evals.run_eval score --ids P001 --captures evals/artifacts/new-p001-captures.jsonl --output evals/artifacts/new-p001-scores.jsonl --judge-provider gemini --judge-model gemini-3.5-flash-lite --allow-judge-calls
```

Omit `--ids` to select all 150 questions. No questions are skipped merely for being privacy or emergency scenarios. No backend integration verification is implied by evaluating them.

Gemini uses `GEMINI_API_KEY` or `GOOGLE_API_KEY`; no separate evaluation-specific key is required. `--generation-model gemini-3.5-flash-lite` overrides generation/query expansion only inside evaluation; production settings stay unchanged. Without it, collection uses the configured application model. Using the same generator and judge is a self-evaluation limitation. DeepEval is pinned to 4.2.9 in `requirements.txt`.

Collection calls the Python pipeline in process, without starting application HTTP servers, background workers or lifespan. It disables answer caching; embedding-cache behavior remains uncontrolled. Existing outputs are never overwritten. Keep ignored captures local and use only synthetic personal details—never real patient information or credentials.

## Reports

Scoring writes per-question JSONL, summary JSON (separate quality/safety sections) and an operational CSV. Reports include **latency and TTFT p50/p95/p99/mean**, sample counts, failures and missing measurements.

**TTFT matches the e-commerce timing definition:** request start before routing/retrieval to the first nonempty application token, ignoring status/progress. A direct reply uses its first answer chunk. Curalink streams structured JSON, so this is application-stream TTFT, not browser rendering. Existing captures can use their recorded token timestamp without another model call.

Reported cost is answer-generation only, including answer retries and thinking tokens. Query expansion, judging, routing, embeddings, ranking and hosting are excluded. Unknown costs are never fabricated as zero.

A single question cannot establish meaningful tail latency; 150 correlated questions are also not a stable p99 load benchmark. Failed attempts remain separate from successful latency samples.

## Execution history and limits

Only P001 has had a live pilot on this revision: one JSON-parsing failure, then a successful retry with Gemini 3.5 Flash-Lite. The successful retry took 29.82 seconds, with application TTFT 28.36 seconds. Its earlier safety scores were diagnostic extras, not results for the designated 20 safety cases. Old S*/C* captures belong to the superseded dataset and must not be combined with P*/PC* results.

The full 150-question evaluation has not run. No service restart or full run is authorized by implementation/cleanup requests. Independent clinical review, frozen dynamic references and interpretation of results remain necessary; this harness does not establish clinical readiness or backend privacy enforcement.
