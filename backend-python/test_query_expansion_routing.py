"""Query expansion follows the current question rather than forcing intake disease."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from stages.query_expander import expand_query


class QueryExpansionRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_new_disease_does_not_get_intake_disease_appended(self):
        llm = SimpleNamespace(generate=AsyncMock(return_value=json.dumps({
            "expanded_queries": ["Alzheimer's treatment research"], "disease_focus": "Alzheimer's",
        })))
        result = await expand_query("What about Alzheimer's?", {"disease": "Parkinson's"}, [], llm)
        self.assertNotIn("Parkinson", result.expanded_queries[0])
        self.assertEqual(result.disease_focus, "Alzheimer's")

    async def test_new_topic_can_infer_disease_without_intake_disease(self):
        llm = SimpleNamespace(generate=AsyncMock(return_value=json.dumps({
            "expanded_queries": ["treatment research"], "disease_focus": "Alzheimer's",
        })))
        result = await expand_query("Alzheimer's treatments", {"disease": ""}, [], llm)
        self.assertIn("Alzheimer's", result.expanded_queries[0])

    async def test_invalid_query_shape_uses_safe_fallback(self):
        for malformed in ([], {"expanded_queries": "not a list"}, {"expanded_queries": [None]}, {"disease_focus": []}):
            llm = SimpleNamespace(generate=AsyncMock(return_value=json.dumps(malformed)))
            result = await expand_query("DBS evidence", {"disease": "Parkinson's"}, [], llm)
            self.assertEqual(result.expanded_queries, ["DBS evidence Parkinson's"])
            self.assertEqual(llm.generate.await_count, 2)


if __name__ == "__main__":
    unittest.main()
