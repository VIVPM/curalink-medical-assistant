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

`stub_cluster.py` runs **three independent FastAPI processes with 20 async slots
each**, replacing AI/retrieval with a 10.438s wait. Each process has a 50-job queue.
No model clients start. A local gateway at port 8021 distributes submissions and
routes job polling/cancellation back to the owning process. Stats aggregate all
three processes.

Start a dedicated Express instance on port 4021 with `FASTAPI_URL=http://127.0.0.1:8021`
and `AUTH_RATE_MAX=10000`. Use the 150 dedicated accounts from `provision.mjs`;
credentials remain ignored. Run with the backend virtualenv Python:

```sh
python load-tests/stub_cluster.py
python load-tests/capacity_suite.py
```

The runner verifies three process IDs and 20 slots per process. Production defaults
are unchanged. Do not resume across worker topologies. Polling respects Retry-After
on 429/503; those responses still count as errors.

Runs 25/50/75/100/150 users sequentially, 30s ramp + 180s hold each. Each user submits
one job. Workers drain between levels; existing worker and queue settings are
preserved and recorded. Threshold failures are retained, not hidden. API summaries
and server-observed job counts/queue wait support the README's two result tables.

**Not yet measured:** SSE TTFT or phase-specific spike recovery. Client polling
completion is not TTFT; server queue wait is instrumented only in the test stub.
