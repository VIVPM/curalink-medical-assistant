"""Tests user-owned inference credentials and per-request client selection."""

import importlib
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

os.environ.setdefault("INTERNAL_API_KEY", "internal-auth-test-key")
os.environ.setdefault("LLM_MODEL", "test/model")
os.environ.setdefault("HF_TOKEN", "test-token")

main = importlib.import_module("main")
own_keys = importlib.import_module("own_keys")


class OwnKeyTests(unittest.IsolatedAsyncioTestCase):
    def test_request_without_keys_uses_server_clients(self):
        main.models.update(embedder="server-embedder", reranker="server-reranker")
        embedder, reranker, selected_llm, own = main.pipeline_clients(SimpleNamespace(headers={}))
        self.assertEqual((embedder, reranker, selected_llm, own), ("server-embedder", "server-reranker", main.llm, False))

    def test_request_with_hf_key_uses_user_clients_without_prompt_cache(self):
        _, _, selected_llm, own = main.pipeline_clients(SimpleNamespace(headers={
            "x-provider-name": "huggingface", "x-provider-hf-key": "hf_user",
        }))
        self.assertTrue(own)
        self.assertEqual(selected_llm.token, "hf_user")
        self.assertFalse(selected_llm.use_cache)

    def test_cloudflare_requires_account_and_token(self):
        with self.assertRaises(HTTPException):
            own_keys.credentials_from_headers({
                "x-provider-name": "cloudflare", "x-provider-hf-key": "hf_user",
            })

    def test_cloudflare_selected_even_when_server_uses_huggingface(self):
        account = "a" * 32
        _, _, selected_llm, own = main.pipeline_clients(SimpleNamespace(headers={
            "x-provider-name": "cloudflare", "x-provider-hf-key": "hf_user",
            "x-provider-cf-account": account, "x-provider-cf-key": "cf_user",
        }))
        self.assertTrue(own)
        self.assertIn(account, selected_llm._base)
        self.assertFalse(selected_llm.use_cache)

    def test_huggingface_selected_even_when_server_uses_cloudflare(self):
        with patch.dict(os.environ, {"LLM_MODEL": "CLOUDFLARE"}):
            _, _, selected_llm, own = main.pipeline_clients(SimpleNamespace(headers={
                "x-provider-name": "huggingface", "x-provider-hf-key": "hf_user",
            }))
        self.assertTrue(own)
        self.assertEqual(selected_llm.model, os.getenv("HF_FALLBACK_MODEL", "meta-llama/Llama-3.3-70B-Instruct"))

    async def test_invalid_hf_token_is_rejected(self):
        client = MagicMock()
        client.get = AsyncMock(return_value=SimpleNamespace(status_code=401))
        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=client)
        context.__aexit__ = AsyncMock(return_value=None)
        with patch.object(own_keys.httpx, "AsyncClient", return_value=context):
            with self.assertRaises(HTTPException) as raised:
                await own_keys.validate_keys(own_keys.OwnKeys(provider="huggingface", hfToken="bad"))
        self.assertEqual(raised.exception.status_code, 400)

    async def test_invalid_cloudflare_token_is_rejected(self):
        client = MagicMock()
        client.get = AsyncMock(side_effect=[
            SimpleNamespace(status_code=200),
            SimpleNamespace(status_code=403),
        ])
        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=client)
        context.__aexit__ = AsyncMock(return_value=None)
        with patch.object(own_keys.httpx, "AsyncClient", return_value=context):
            with self.assertRaises(HTTPException) as raised:
                await own_keys.validate_keys(own_keys.OwnKeys(
                    provider="cloudflare", hfToken="hf_user",
                    cfAccountId="a" * 32, cfToken="bad",
                ))
        self.assertEqual(raised.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
