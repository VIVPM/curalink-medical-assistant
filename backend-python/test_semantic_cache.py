import unittest

from semantic_cache import _bucket


class SemanticCacheIsolationTests(unittest.TestCase):
    def test_bucket_isolated_by_tenant_and_location(self):
        base = _bucket("user-a", "Parkinson's", "DBS", "Toronto")
        self.assertTrue(base.startswith("semq:user-a:"))
        self.assertNotIn("parkinson", base)
        self.assertNotIn("toronto", base)
        self.assertNotEqual(base, _bucket("user-b", "Parkinson's", "DBS", "Toronto"))
        self.assertNotEqual(base, _bucket("user-a", "Parkinson's", "DBS", "Boston"))
        self.assertEqual(base, _bucket(" USER-A ", "PARKINSON'S", "dbs", " toronto "))


if __name__ == "__main__":
    unittest.main()
