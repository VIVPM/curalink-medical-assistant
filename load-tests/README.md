# Load tests

k6: login → create session → browse every 5–10s. Optional jobs submit once per
user and poll up to 180s. Default: 30s ramp, 180s hold, 60s request timeout.

## Prerequisites

- Local Express/FastAPI with **isolated test database and Redis**. Never use live resources.
- Installed k6; dedicated accounts, one per peak user, in ignored `accounts.json`:
  `[{"email":"test@example.invalid","password":"test-password"}]`.
- Rate limits remain enabled except the dedicated local capacity API's shared-IP
  login allowance (`AUTH_RATE_MAX=10000`). Default mode makes no model calls.
  `--jobs --allow-model-calls` uses real providers and may incur charges.

## Commands (repository root)

```sh
python load-tests/run.py --accounts load-tests/accounts.json --users 1
# Review results before progressing through 25, 50, 75, 100, 150 users.
python load-tests/run.py --accounts load-tests/accounts.json --users 25
python load-tests/run.py --accounts load-tests/accounts.json --users 25 --peak 50 --spike
python load-tests/report.py load-tests/results/<run-tag>/summary.json
```

Results: ignored `results/<run-tag>/`; API p50/p95/p99, errors and job counters.
Cleanup deletes only sessions matching that run's unique title. After interruption,
use `--cleanup-tag load-<32-hex-characters>`. Accounts and job records remain;
uncompleted jobs may continue—inspect/cancel before rerunning. Cleanup or threshold
failures exit nonzero.

## Stubbed capacity suite

`stub_server.py` starts the real FastAPI app and job manager at port 8021, replacing
only pipeline execution with a 10.438s async wait. No model clients are started.
Start a dedicated local Express instance on port 4021 pointing at 8021 with
`AUTH_RATE_MAX=10000`. `provision.mjs` creates 150 dedicated account fixtures in the
configured database; credentials stay ignored. Never run provisioning accidentally
against an unrelated database. Then run with the backend virtualenv Python:

```sh
# Start separately; uses 60 async job slots in one local process (not 3 processes).
python load-tests/stub_server.py --workers 60
python load-tests/capacity_suite.py
```

The optional 60-slot profile matches ecommerce's aggregate job concurrency only;
it does not change production defaults or prove 60 real-model requests are safe.
Omit `--workers` to use the configured/default worker count. Do not use `--resume`
across different worker profiles. Polling respects Retry-After on 429/503; these
responses still count as errors.

Runs 25/50/75/100/150 users sequentially, 30s ramp + 180s hold each. Each user submits
one job. Workers drain between levels; existing worker and queue settings are
preserved and recorded. Threshold failures are retained, not hidden. API summaries
and server-observed job counts/queue wait support the README's two result tables.

**Not yet measured:** SSE TTFT or phase-specific spike recovery. Client polling
completion is not TTFT; server queue wait is instrumented only in the test stub.
