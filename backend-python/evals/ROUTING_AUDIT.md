# Frozen routing audit — 2026-11-07

## Inputs and labels

The 150 cases in `../../evals/golden/v1` are frozen by ID and question SHA-256 in `../../evals/golden/v1/routing_labels.json`. Labels apply the rules in `stages/query_router.py` and `routing.py` to each question and its canonical prior history. They are authored, provisional routing labels, not clinical validation. Ten mixed-intent or context-sensitive cases are explicitly marked ambiguous. The four main decisions are `history-only`, `clarify`, `full RAG`, and `urgent`; `smalltalk` and `scope refusal` retain the router's actual control paths. A model tier is labeled only for 71 clearly simple or complex answer cases; 79 cases have no defensible fixed tier or do not call an answer model. The per-case file gives the rule and tier rationale.

The primary audit uses the 2026-11-01 Gemini capture, `artifacts/full-150-20261101T181123Z/captures.jsonl`. Its exact input hashes and all 150 case decisions are in `artifacts/routing-audit-20261107-v2/`. A second audit of the successful HF routed captures is in `artifacts/routing-audit-hf-20261107-v2/`; it finds the same two urgent misses and two unsupported history-only decisions. These are saved decisions, not a new provider run.

## Route baseline

There are 150/150 successful captures. The confusion matrix counts all 150; route accuracy excludes 10 marked ambiguous cases (132/140). Actual route counts: full RAG 105, history-only 20, clarify 2, urgent 6, smalltalk 1, scope refusal 16.

| Expected → actual | Full RAG | History-only | Clarify | Urgent | Smalltalk | Scope refusal |
|---|---:|---:|---:|---:|---:|---:|
| Full RAG (103) | 97 | 1 | 0 | 0 | 0 | 5 |
| History-only (18) | 0 | 18 | 0 | 0 | 0 | 0 |
| Clarify (5) | 3 | 0 | 2 | 0 | 0 | 0 |
| Urgent (8) | 1 | 1 | 0 | 6 | 0 | 0 |
| Smalltalk (2) | 1 | 0 | 0 | 0 | 1 | 0 |
| Scope refusal (14) | 3 | 0 | 0 | 0 | 0 | 11 |

**Urgent false negatives: 2/8.** P058 (barely breathing; rescue inhaler not helping) was sent to RAG. PC08-T4 (swelling tongue and struggling to breathe) was sent to history-only generation. Both answers ultimately advised emergency care, but the pre-provider emergency boundary failed and delayed the direct response. The Node boundary also returns false for those phrasings and for PC15-T4's “bleeding heavily”; Python does catch PC15-T4. This is an offline boundary check, not an end-to-end Node capture.

**Unsupported history-only: 2/20 actual history-only routes.** PC02-T3 introduces worsening night symptoms and increased inhaler use; new patient facts should get fresh evidence under the router rule. PC08-T4 is the urgent miss above. PC15-T2 is deliberately ambiguous: the prior answer distinguishes monitored hospital positioning, but the follow-up asks about home use for any breathlessness.

Three of five expected clarifications went through RAG (P061, P063, and the ambiguous PC12-T4). P063's numeric “8” lacks the test type and units, so its answer should not interpret a specific result. The scored tier match was 52/66 cases with both a clear expected tier and an actual answer model; these provisional tier labels are less reliable than the urgent boundary labels.

The smallest proposed behavior fix is to extend the existing **Node and Python pre-provider urgent patterns** for the observed breathing and bleeding phrasings, with negation/hypothetical controls, then add direct regression expectations. Separately, tighten Jev's history-only eligibility for newly reported symptoms and add a clarification rule for unspecified measurements. No production routing behavior was changed in this audit.

## Saved routed versus always-large HF captures

`artifacts/routing-paired-hf-20261107-v3/` selects the first successful attempt per case from the 2026-11-05 HF original and retry runs. Both arms have 150/150 paired successes; routed had 167 attempts, always-large 163. The model IDs are `meta-llama/Llama-3.1-8B-Instruct` and `meta-llama/Llama-3.3-70B-Instruct`; the upstream HF provider was selected automatically and its serving revision is not recorded. Jev uses the unpinned `jev-latest` alias. Answer cache was disabled; embedding cache was uncontrolled. Routed generated 64 answers with 8B and 61 with 70B; 25 direct responses used no answer model. Always-large generated 125 answers with 70B; 25 direct responses used no answer model. This is an always-large **answer tier**, while query expansion still used the application's 8B tier.

| Measure, denominator | Routed | Always-large | Difference |
|---|---:|---:|---:|
| End-to-end latency p50, 150 successes | 40.61 s | 31.22 s | Routed 30.1% slower |
| End-to-end latency p95, 150 successes | 72.03 s | 48.07 s | Routed 49.8% slower |
| End-to-end latency p99, 150 successes | 95.15 s | 58.74 s | Routed 62.0% slower |
| Estimated answer generation, all original/retry attempts | $0.03597 | $0.07249 | Routed 50.4% lower |
| LLM-judged answer quality | Pending approval | Pending approval | No paired result yet |

Cost is **not billed spend**. The script recalculates saved provider token usage for answer calls (`max_tokens=4000`) with historical illustrative input/output rates per million tokens of $0.02/$0.05 for 8B and $0.15/$0.30 for 70B. It includes 136 routed and 134 always-large answer calls, including retried attempts that reached a model. It excludes Jev, expansion, retrieval, judging and hosting; provider auto selection and actual charges are unknown. End-to-end latency includes routing and retrieval. P053 and P054 used different selection policies between arms after retries. Timing also reflects changing provider and retrieval conditions, so the latency difference is descriptive, not a controlled causal estimate. Both arms' per-case latency and selection-policy flags are in `paired.jsonl`.

The 2026-11-01 Gemini run remains a valid **LLM-judged** evaluation of its own 150 answers: 116/150 cases passed its rubric, with correctness 0.852 (102 graded), completeness 0.836 (102), answer relevancy 0.904 (102), and faithfulness 0.994 (104). It used `gemini-3.5-flash-lite` for both nominal tiers and as judge, so it does not answer the paired 8B-versus-70B quality question. Its answer-generation estimate was $0.2276394 for 150 questions, excluding other components.

The separate semantic-cache benchmark used 82 selected first-turn RAG questions, 81 successful seeds, and 58 semantic hits in 81 cached passes. Jev and LLM prompt cache were off; Redis embedding cache stayed on. The current Python pipeline bypasses semantic lookup whenever Jev routing is enabled (`main.py`), so the cache benchmark does not apply to the routed 150-question run. It used prefixed paraphrases rather than production traffic. The earlier reported p95 comparison (218.22 s without semantic lookup versus 57.14 s with it) was affected by provider timeouts and does not estimate a 150-question routing benefit.

## Offline commands and paid follow-up

Run from `backend-python`:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -p 'test_*.py'
.\.venv\Scripts\python.exe -m unittest evals.test_eval evals.test_routing_audit
cd ..\backend-node
node --test routes/chat.test.js
```

The offline audit command, also from `backend-python`, is:

```powershell
.\.venv\Scripts\python.exe -m evals.routing_audit --captures evals/artifacts/full-150-20261101T181123Z/captures.jsonl --output-dir evals/artifacts/routing-audit-NEW
```

The existing score CLI can judge the two validated, paired HF capture files. **These commands have not been run and require explicit user approval:**

```powershell
.\.venv\Scripts\python.exe -m evals.run_eval score --captures evals/artifacts/routing-paired-hf-20261107-v3/routed-success.jsonl --output evals/artifacts/routing-paired-hf-20261107-v3/routed-scores.jsonl --judge-provider gemini --judge-model gemini-3.5-flash-lite --allow-judge-calls
.\.venv\Scripts\python.exe -m evals.run_eval score --captures evals/artifacts/routing-paired-hf-20261107-v3/always_large-success.jsonl --output evals/artifacts/routing-paired-hf-20261107-v3/always_large-scores.jsonl --judge-provider gemini --judge-model gemini-3.5-flash-lite --allow-judge-calls
```

The prior 150-case judge cost was $0.5333515. At the same model and similar output lengths, 300 case scores would be roughly **$1.07**, with actual cost uncertain and no hard spending cap in the existing CLI. Recheck current pricing before authorization. The judging comparison would still inherit the saved runs' retrieval mismatch and provider conditions. Human review could strengthen labels, but qualified clinical review would be needed for a clinical-validation claim.
