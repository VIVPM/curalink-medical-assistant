"""Tests FastAPI internal service authentication boundaries."""

import importlib
import os
import unittest

import httpx

os.environ["INTERNAL_API_KEY"] = "internal-auth-test-key"
os.environ["LLM_MODEL"] = "test/model"
os.environ["HF_TOKEN"] = "test-token"
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""
os.environ["GRAFANA_OTLP_ENDPOINT"] = ""
os.environ["GRAFANA_OTLP_AUTH"] = ""

app = importlib.import_module("main").app


class InternalAuthTests(unittest.IsolatedAsyncioTestCase):
    async def test_private_endpoints_require_shared_key(self):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            self.assertEqual((await client.get("/")).status_code, 200)
            self.assertEqual((await client.get("/docs")).status_code, 401)
            self.assertEqual(
                (await client.get("/docs", headers={"X-Internal-API-Key": "wrong"})).status_code,
                401,
            )
            self.assertEqual(
                (await client.get(
                    "/docs",
                    headers={"X-Internal-API-Key": "internal-auth-test-key"},
                )).status_code,
                200,
            )


if __name__ == "__main__":
    unittest.main()
