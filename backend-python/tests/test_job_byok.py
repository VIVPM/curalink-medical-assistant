"""Queued BYOK references resolve at execution time without persisting secrets."""

import asyncio
import contextlib
import importlib
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from fastapi import HTTPException

os.environ.setdefault("INTERNAL_API_KEY", "job-auth-test-key")
os.environ.setdefault("LLM_MODEL", "test/model")
os.environ.setdefault("HF_TOKEN", "test-token")
main = importlib.import_module("main")
from own_keys import job_credential_headers
from job_manager import JobManager


class JobByokTests(unittest.IsolatedAsyncioTestCase):
    async def test_free_jobs_never_lookup_credentials(self):
        with patch("own_keys.httpx.AsyncClient") as client:
            self.assertEqual(await job_credential_headers("alice", "free"), {})
            client.assert_not_called()

    async def test_lookup_uses_only_account_and_provider_reference(self):
        for provider in ("huggingface", "cloudflare"):
            with self.subTest(provider=provider):
                keys = {"provider": provider, "hfToken": "hf_private", "cfAccountId": "a" * 32, "cfToken": "cf_private"}
                client = MagicMock()
                client.post = AsyncMock(return_value=SimpleNamespace(status_code=200, json=lambda: keys))
                context = MagicMock()
                context.__aenter__ = AsyncMock(return_value=client)
                context.__aexit__ = AsyncMock(return_value=None)
                with patch.dict(os.environ, {"NODE_API_URL": "https://node.example"}), patch("own_keys.httpx.AsyncClient", return_value=context):
                    headers = await job_credential_headers("alice", provider)
                self.assertEqual(headers["x-provider-name"], provider)
                self.assertEqual(client.post.call_args.args[0], "https://node.example/internal/job-credentials")
                self.assertEqual(client.post.call_args.kwargs["json"], {"tenant": "alice", "provider": provider})
                self.assertNotIn("hf_private", json.dumps(client.post.call_args.kwargs))

    async def test_missing_removed_changed_or_malformed_credentials_fail_closed(self):
        for status, body in [(400, {}), (409, {}), (503, {}), (200, {}),
                             (200, {"provider": "cloudflare", "hfToken": "hf_private"})]:
            with self.subTest(status=status, body=body):
                client = MagicMock()
                client.post = AsyncMock(return_value=SimpleNamespace(status_code=status, json=lambda: body))
                context = MagicMock()
                context.__aenter__ = AsyncMock(return_value=client)
                context.__aexit__ = AsyncMock(return_value=None)
                with patch.dict(os.environ, {"NODE_API_URL": "https://node.example"}), patch("own_keys.httpx.AsyncClient", return_value=context):
                    with self.assertRaises(RuntimeError) as raised:
                        await job_credential_headers("alice", "huggingface")
                self.assertNotIn("hf_private", str(raised.exception))
        with patch.dict(os.environ, {"NODE_API_URL": ""}):
            with self.assertRaisesRegex(RuntimeError, "NODE_API_URL"):
                await job_credential_headers("alice", "huggingface")

    async def test_network_and_json_failures_do_not_expose_credentials(self):
        for error in (httpx.ReadTimeout("hf_private"), ValueError("hf_private")):
            client = MagicMock()
            if isinstance(error, httpx.HTTPError):
                client.post = AsyncMock(side_effect=error)
            else:
                response = MagicMock(status_code=200)
                response.json.side_effect = error
                client.post = AsyncMock(return_value=response)
            context = MagicMock()
            context.__aenter__ = AsyncMock(return_value=client)
            context.__aexit__ = AsyncMock(return_value=None)
            with patch.dict(os.environ, {"NODE_API_URL": "https://node.example"}), patch("own_keys.httpx.AsyncClient", return_value=context):
                with self.assertRaises(RuntimeError) as raised:
                    await job_credential_headers("alice", "huggingface")
            self.assertNotIn("hf_private", str(raised.exception))

    async def test_background_pipeline_routes_with_user_owned_clients(self):
        for provider in ("huggingface", "cloudflare"):
            with self.subTest(provider=provider):
                headers = {"x-provider-name": provider, "x-provider-hf-key": "hf_private",
                           "x-provider-cf-account": "a" * 32, "x-provider-cf-key": "cf_private"}
                route = AsyncMock(return_value=SimpleNamespace(decision=None, response={"overview": "test", "pipelineMeta": {}}))
                data = {"tenant": "alice", "credential_provider": provider, "static": {}, "current": {"userMessage": "Hello"}, "_job_id": "test-job"}
                with patch.object(main, "job_credential_headers", AsyncMock(return_value=headers)) as lookup, \
                     patch.object(main, "route_request", route), patch.object(main, "routing_enabled", return_value=False), \
                     patch.object(main, "models", {}), patch("checkpoint.clear"):
                    await main._run_pipeline_for_job(data)
                lookup.assert_awaited_once_with("alice", provider)
                selected = route.call_args.args[3]
                self.assertIsNot(selected, main.llm)
                self.assertFalse(selected.use_cache)
                if provider == "huggingface":
                    self.assertEqual(selected.token, "hf_private")
                else:
                    self.assertEqual(selected.api_token, "cf_private")
                self.assertNotIn("hf_private", json.dumps(data))

    async def test_credential_failure_marks_job_failed_without_running_server_inference(self):
        manager = JobManager(max_workers=1)
        manager.set_pipeline(main._run_pipeline_for_job)
        data = {"tenant": "alice", "credential_provider": "huggingface", "static": {}, "current": {"userMessage": "Hello"}}
        with patch.object(manager, "_persist"), \
             patch.object(main, "job_credential_headers", AsyncMock(side_effect=RuntimeError("Saved provider credentials unavailable hf_private"))), \
             patch.object(main, "route_request", AsyncMock()) as route:
            job = manager.create_job(data)
            await manager._queue.put(job)
            worker = asyncio.create_task(manager._worker(0))
            try:
                await asyncio.wait_for(manager._queue.join(), timeout=2)
                self.assertEqual(job.state, "failed")
                self.assertIn("Saved-provider job failed", job.error or "")
                self.assertNotIn("hf_private", json.dumps(job.to_dict()))
                route.assert_not_awaited()
            finally:
                worker.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await worker

    async def test_job_results_cancellation_and_events_are_tenant_scoped(self):
        manager = JobManager()
        with patch.object(manager, "_persist"), patch("job_manager.get_job_manager", return_value=manager):
            job = manager.create_job({"tenant": "alice", "credential_provider": "huggingface"})
            self.assertEqual(job.to_dict()["tenant"], "alice")
            alice = SimpleNamespace(headers={"x-tenant-id": "alice"})
            bob = SimpleNamespace(headers={"x-tenant-id": "bob"})
            self.assertEqual((await main.poll_job(job.id, alice))["id"], job.id)
            for call in (main.poll_job, main.cancel_job, main.replay_job_events):
                with self.assertRaises(HTTPException) as raised:
                    await call(job.id, bob)
                self.assertEqual(raised.exception.status_code, 404)
            self.assertEqual(job.state, "pending")
            self.assertEqual((await main.cancel_job(job.id, alice))["state"], "cancelled")


if __name__ == "__main__":
    unittest.main()
