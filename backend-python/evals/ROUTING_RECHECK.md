# Offline routing recheck after guard changes

The 2026-11-07 [routing audit](ROUTING_AUDIT.md) is the baseline. This recheck uses its frozen 150 questions, labels, and saved Gemini capture. It applies the current Python pre-provider guard and new-patient-fact rule to the saved inputs and old route decisions. **It does not call a provider, retrieve new documents, or generate new answers.** The projected results are code-level routes, not a new end-to-end evaluation or clinical validation.

| Route measure | Saved capture baseline | Code-level projection |
| --- | ---: | ---: |
| Route matches, excluding 10 ambiguous labels | 132/140 | 135/140 |
| Urgent false negatives | 2/8 | 0/8 |
| Unsupported history-only routes | 2/20 actual history-only | 0/18 projected history-only |

P058 and PC08-T4 now meet the Python and Node pre-provider emergency checks. PC02-T3 is routed away from history-only generation because it reports new or worsening symptoms. The Node test checks all 150 frozen questions: all eight clear urgent labels intercept, and no clear nonurgent label intercepts. Focused tests also cover negated, quoted, and hypothetical wording, including a real symptom reported alongside hypothetical text.

The five remaining clear route mismatches are P061, P063, P067, P075, and P084. The existing audit describes them. The projected route does not prove that new retrieval, answers, latency, or costs would be correct. The routing labels are provisional; a qualified clinical review would be needed before claiming clinical validation. The urgent boundary is intentionally a limited phrase guard and should not be represented as exhaustive emergency detection.

## Saved evidence

- Baseline capture audit: `backend-python/evals/artifacts/routing-audit-20261107-v2/`.
- Recheck: `backend-python/evals/artifacts/routing-code-recheck-20261107-v2/`. Its `saved_capture_baseline/report.json` repeats the original route counts, `code_projected_report.json` records before/after counts and remaining mismatches, and `code_projected_decisions.jsonl` records each route change. The report includes input and Python source hashes.

## Offline commands

From `backend-python`:

```powershell
.\.venv\Scripts\python.exe -m evals.routing_audit --captures evals/artifacts/full-150-20261101T181123Z/captures.jsonl --output-dir evals/artifacts/routing-code-recheck-NEW --code-recheck
.\.venv\Scripts\python.exe -m unittest discover -p 'test_*.py'
.\.venv\Scripts\python.exe -m unittest evals.test_eval evals.test_routing_audit
```

From `backend-node`:

```powershell
npm test
```

The severe breathing and tongue-swelling examples are consistent with [NHS asthma guidance](https://www.nhs.uk/conditions/asthma/) and [NHS angioedema guidance](https://www.nhs.uk/conditions/angioedema/). The tests check routing behavior, not medical advice or clinical correctness.
