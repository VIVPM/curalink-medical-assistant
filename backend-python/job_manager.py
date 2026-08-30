"""
Async job queue with state machine and backpressure.

Jobs: pending → running → streaming → completed | failed | cancelled.
Backed by Redis for persistence. In-process asyncio.Queue for backpressure.

Queue rejects new jobs (503) when full — the client retries with Retry-After.
Workers are sized by JOB_WORKERS env (default 3).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid

logger = logging.getLogger(__name__)

# States
PENDING = "pending"
RUNNING = "running"
STREAMING = "streaming"
COMPLETED = "completed"
FAILED = "failed"
CANCELLED = "cancelled"

_JOB_TTL = 3600  # 1h — completed jobs expire from Redis


class Job:
    __slots__ = ("id", "state", "request_data", "result", "error",
                 "created_at", "updated_at")

    def __init__(self, job_id: str, request_data: dict):
        self.id = job_id
        self.state = PENDING
        self.request_data = request_data
        self.result = None
        self.error = None
        self.created_at = time.time()
        self.updated_at = time.time()

    def to_dict(self) -> dict:
        d = {
            "id": self.id, "state": self.state,
            "created_at": self.created_at, "updated_at": self.updated_at,
        }
        if self.result is not None:
            d["result"] = self.result
        if self.error is not None:
            d["error"] = self.error
        return d


class JobManager:
    def __init__(self, max_queue: int = 50, max_workers: int = 3):
        self._queue: asyncio.Queue[Job] = asyncio.Queue(maxsize=max_queue)
        self._jobs: dict[str, Job] = {}
        self._max_workers = max_workers
        self._workers: list[asyncio.Task] = []
        self._pipeline_fn = None
        self._started = False

    def set_pipeline(self, fn):
        """Set the async pipeline function: fn(request_data) -> result dict."""
        self._pipeline_fn = fn

    async def start(self):
        if self._started:
            return
        self._started = True
        for i in range(self._max_workers):
            self._workers.append(asyncio.create_task(self._worker(i)))
        logger.info("[jobs] %d workers started, queue cap %d",
                    self._max_workers, self._queue.maxsize)

    def create_job(self, request_data: dict) -> Job:
        job = Job(str(uuid.uuid4()), request_data)
        self._jobs[job.id] = job
        self._persist(job)
        return job

    async def enqueue(self, job: Job) -> bool:
        """Returns False if queue is full (backpressure)."""
        if self._queue.full():
            return False
        await self._queue.put(job)
        return True

    def get_job(self, job_id: str) -> dict | None:
        job = self._jobs.get(job_id)
        if job:
            return job.to_dict()
        return self._load(job_id)

    def cancel_job(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job or job.state in (COMPLETED, FAILED, CANCELLED):
            return False
        job.state = CANCELLED
        job.updated_at = time.time()
        self._persist(job)
        return True

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    async def _worker(self, wid: int):
        while True:
            job = await self._queue.get()
            if job.state == CANCELLED:
                self._queue.task_done()
                continue
            try:
                job.state = RUNNING
                job.updated_at = time.time()
                self._persist(job)
                logger.info("[jobs] worker %d processing %s", wid, job.id)

                if self._pipeline_fn:
                    result = await self._pipeline_fn(job.request_data)
                    if job.state == CANCELLED:
                        continue
                    job.state = COMPLETED
                    job.result = result
                else:
                    job.state = FAILED
                    job.error = "no pipeline configured"
            except Exception as e:
                job.state = FAILED
                job.error = str(e)
                logger.error("[jobs] worker %d job %s failed: %s", wid, job.id, e)
            finally:
                job.updated_at = time.time()
                self._persist(job)
                self._queue.task_done()

    def _persist(self, job: Job):
        try:
            from redis_cache import _get_client
            client = _get_client()
            if client:
                client.set(f"job:{job.id}", json.dumps(job.to_dict()), ex=_JOB_TTL)
        except Exception:
            pass  # ponytail: Redis down = in-memory only, still works

    def _load(self, job_id: str) -> dict | None:
        try:
            from redis_cache import _get_client
            client = _get_client()
            if client:
                data = client.get(f"job:{job_id}")
                if data:
                    return json.loads(data)
        except Exception:
            pass
        return None


_manager: JobManager | None = None


def get_job_manager() -> JobManager:
    global _manager
    if _manager is None:
        _manager = JobManager(
            max_queue=int(os.getenv("JOB_QUEUE_SIZE", "50")),
            max_workers=int(os.getenv("JOB_WORKERS", "3")),
        )
    return _manager
