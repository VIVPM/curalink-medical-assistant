"""History-only answers retain source snippets and fall back when evidence is absent."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from llm_backend import HFBackend
from routing import route_request
from stages.history_context import MAX_HISTORY_CHARS, bounded_history, build_history_context
from stages.query_router import RoutingDecision
from token_budget import TokenBudget, TokenBudgetExceeded


def research_history():
    return [
        {"role": "user", "content": "What does the DBS study say?"},
        {"role": "assistant", "content": "The study reports improved symptoms.", "structuredResponse": {
            "overview": "The study reports improved symptoms.",
            "insights": [{"finding": "Symptoms improved in the study.", "source_details": [{
                "title": "DBS research", "url": "https://pubmed.ncbi.nlm.nih.gov/123/", "platform": "pubmed",
                "snippet": "Participants had improved motor scores after DBS.",
            }]}],
            "trials": [], "recommendations": [], "abstain_reason": None,
            "pipelineMeta": {"secret": "not-router-input"}, "contact": "not-router-input",
        }},
    ]


class HistoryContextTests(unittest.TestCase):
    def test_history_contains_findings_and_sources_but_no_internal_metadata(self):
        history = bounded_history(research_history())
        serialized = json.dumps(history)
        self.assertIn("improved motor scores", serialized)
        self.assertNotIn("not-router-input", serialized)
        self.assertLessEqual(len(serialized), MAX_HISTORY_CHARS + 20)

    def test_history_budget_and_recent_turn_limit(self):
        messages = [{"role": "user", "content": "x" * 10000} for _ in range(30)]
        history = bounded_history(messages)
        self.assertEqual(len(history), 6)
        self.assertTrue(all(len(turn["content"]) == 1600 for turn in history))

    def test_history_payload_reuses_saved_sources(self):
        payload = build_history_context(bounded_history(research_history()), "Simplify that", {"disease": "Parkinson's"})
        self.assertIsNotNone(payload)
        self.assertIn("HISTORY-ONLY RESPONSE", payload.system_prompt)
        self.assertEqual(payload.doc_anchors["doc1"].url, "https://pubmed.ncbi.nlm.nih.gov/123/")
        self.assertIn("improved motor scores", payload.user_prompt)

    def test_overview_only_history_is_not_treated_as_source_evidence(self):
        self.assertIsNone(build_history_context([{"role": "assistant", "content": "A previous overview"}], "Explain", {}))


class RoutedRequestTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.base = HFBackend("hf_user", "old/model", use_cache=False)
        self.form = {"disease": "Parkinson's", "intent": "DBS", "location": "Toronto"}

    async def test_disabled_routing_preserves_original_pipeline(self):
        with patch("routing.routing_enabled", return_value=False), patch("routing.classify_query") as classify:
            routed = await route_request("Question", self.form, [], self.base)
        self.assertIs(routed.answer_llm, self.base)
        self.assertIsNone(routed.decision)
        classify.assert_not_called()

    async def test_new_topic_clears_intake_disease_and_intent_but_retains_location(self):
        with patch("routing.routing_enabled", return_value=True), patch("routing.classify_query", AsyncMock(return_value=RoutingDecision(topic="new_topic", model_tier="8b"))):
            routed = await route_request("What about Alzheimer's?", self.form, research_history(), self.base)
        self.assertEqual(routed.static_context, {"disease": "", "intent": "", "location": "Toronto"})
        self.assertEqual(self.form["disease"], "Parkinson's")
        self.assertIn("8B", routed.answer_llm.model)

    async def test_conversation_uses_selected_model_and_preserves_citations(self):
        output = {"overview": "The saved study reported improved symptoms.", "insights": [{"finding": "Improved symptoms", "sources": ["doc1"]}], "trials": [], "recommendations": [], "abstain_reason": None}
        reasoner = AsyncMock(return_value=SimpleNamespace(llm_output=output))
        with patch("routing.routing_enabled", return_value=True), patch("routing.classify_query", AsyncMock(return_value=RoutingDecision(route="conversation", model_tier="70b"))), patch("routing.run_reasoner", reasoner):
            routed = await route_request("Compare those findings", self.form, research_history(), self.base)
        self.assertIn("70B", reasoner.call_args.args[1].model)
        self.assertEqual(routed.response["insights"][0]["source_details"][0]["url"], "https://pubmed.ncbi.nlm.nih.gov/123/")
        self.assertFalse(routed.response["pipelineMeta"]["routing"]["retrieval_used"])

    async def test_missing_history_sources_fall_back_to_rag(self):
        with patch("routing.routing_enabled", return_value=True), patch("routing.classify_query", AsyncMock(return_value=RoutingDecision(route="conversation"))):
            routed = await route_request("Explain", self.form, [{"role": "assistant", "content": "Overview only"}], self.base)
        self.assertIsNone(routed.response)
        self.assertEqual(routed.decision.route, "rag")
        self.assertEqual(routed.decision.fallback, "history_insufficient")

    async def test_history_abstention_falls_back_to_rag(self):
        with patch("routing.routing_enabled", return_value=True), patch("routing.classify_query", AsyncMock(return_value=RoutingDecision(route="conversation"))), patch("routing.run_reasoner", AsyncMock(return_value=SimpleNamespace(llm_output={"abstain_reason": "Missing evidence"}))):
            routed = await route_request("Explain", self.form, research_history(), self.base)
        self.assertIsNone(routed.response)
        self.assertEqual(routed.decision.route, "rag")

    async def test_history_generation_respects_background_token_budget(self):
        with patch("routing.routing_enabled", return_value=True), patch("routing.run_reasoner") as reasoner:
            with self.assertRaises(TokenBudgetExceeded):
                await route_request("Summarize", self.form, research_history(), self.base,
                                    decision=RoutingDecision(route="conversation"), budget=TokenBudget(max_tokens=1))
        reasoner.assert_not_called()

    async def test_saved_background_decision_avoids_reclassification(self):
        with patch("routing.routing_enabled", return_value=True), patch("routing.classify_query") as classify:
            routed = await route_request("Question", self.form, [], self.base, decision=RoutingDecision(model_tier="8b"))
        classify.assert_not_called()
        self.assertEqual(routed.decision.model_tier, "8b")


if __name__ == "__main__":
    unittest.main()
