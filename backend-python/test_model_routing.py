"""Model-tier selection keeps user-owned credentials and provider boundaries."""

import unittest
from unittest.mock import patch

from llm_backend import CloudflareBackend, HFBackend, get_tier_backend


class ModelRoutingTests(unittest.TestCase):
    def test_huggingface_tiers_use_original_token_without_enabling_cache(self):
        base = HFBackend("hf_user", "old/model", use_cache=False)
        for tier, expected in (("8b", "meta-llama/Llama-3.1-8B-Instruct"), ("70b", "meta-llama/Llama-3.3-70B-Instruct")):
            with self.subTest(tier=tier):
                selected = get_tier_backend(base, tier)
                assert isinstance(selected, HFBackend)
                self.assertIs(selected._sem, base._sem)
                self.assertEqual(selected.model, expected)
                self.assertEqual(selected.token, "hf_user")
                self.assertFalse(selected.use_cache)

    def test_cloudflare_tiers_keep_account_and_token(self):
        base = CloudflareBackend("user_account", "user_token", use_cache=False)
        for tier, expected in (("8b", "@cf/meta/llama-3.1-8b-instruct-fp8"), ("70b", "@cf/meta/llama-3.3-70b-instruct-fp8-fast")):
            selected = get_tier_backend(base, tier)
            assert isinstance(selected, CloudflareBackend)
            self.assertIs(selected._sem, base._sem)
            self.assertEqual(selected.model, expected)
            self.assertEqual(selected.account_id, "user_account")
            self.assertEqual(selected.api_token, "user_token")
            self.assertFalse(selected.use_cache)

    def test_explicit_model_configuration(self):
        base = HFBackend("hf_user", "old/model")
        with patch.dict("os.environ", {"HF_LLM_8B_MODEL": "configured/8b"}):
            selected = get_tier_backend(base, "8b")
            assert isinstance(selected, HFBackend)
            self.assertEqual(selected.model, "configured/8b")

    def test_existing_matching_client_is_reused(self):
        base = HFBackend("hf_user", "meta-llama/Llama-3.1-8B-Instruct")
        self.assertIs(get_tier_backend(base, "8b"), base)

    def test_unknown_tier_and_empty_model_are_rejected(self):
        base = HFBackend("hf_user", "old/model")
        with self.assertRaises(ValueError):
            get_tier_backend(base, "invalid")
        with patch.dict("os.environ", {"HF_LLM_70B_MODEL": " "}), self.assertRaises(ValueError):
            get_tier_backend(base, "70b")


if __name__ == "__main__":
    unittest.main()
