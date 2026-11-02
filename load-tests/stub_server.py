"""Local-only real FastAPI/job workers with simulated AI; never a production entrypoint."""
import asyncio
from contextlib import asynccontextmanager
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend-python'))
from dotenv import load_dotenv
load_dotenv(ROOT / 'backend-python' / '.env', override=False)
os.environ['OTEL_SDK_DISABLED'] = 'true'
os.environ['LANGFUSE_ENABLED'] = 'false'

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=int(os.getenv('JOB_WORKERS', '3')))
    args = parser.parse_args()
    if not 1 <= args.workers <= 60:
        parser.error('--workers must be between 1 and 60')
    os.environ['JOB_WORKERS'] = str(args.workers)

from job_manager import get_job_manager
import main

DELAY = 10.438
jm = get_job_manager()
original_enqueue = jm.enqueue


async def enqueue(job):
    job.request_data['_load_enqueued_at'] = time.monotonic()
    accepted = await original_enqueue(job)
    job.request_data['_load_accepted'] = accepted
    return accepted


async def simulated_pipeline(data):
    wait_ms = (time.monotonic() - data['_load_enqueued_at']) * 1000
    await asyncio.sleep(DELAY)
    return {'overview': 'Simulated AI response for capacity testing.', 'insights': [],
            'trials': [], 'recommendations': [], 'follow_up_questions': [],
            'load_test': {'simulated': True, 'delay_seconds': DELAY, 'queue_wait_ms': wait_ms}}


@asynccontextmanager
async def lifespan(app):
    jm.enqueue = enqueue
    jm.set_pipeline(simulated_pipeline)
    await jm.start()
    yield


main.app.router.lifespan_context = lifespan


@main.app.get('/load-test/config')
async def config():
    return {'simulated': True, 'delay_seconds': DELAY, 'workers': jm._max_workers,
            'queue_size': jm._queue.maxsize}


@main.app.get('/load-test/stats')
async def stats(tag: str):
    jobs = [j for j in jm._jobs.values() if j.request_data.get('static', {}).get('intent') == tag]
    from load_stats import summarize_jobs
    return summarize_jobs(jobs)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(main.app, host='127.0.0.1', port=8021, log_level='warning')
