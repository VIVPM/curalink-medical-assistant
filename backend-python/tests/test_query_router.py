"""Contract tests for Jev routing, without external inference calls."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from stages.query_router import classify_query
from routing import immediate_response


def jev_response(route="rag", tier="8b", topic="session"):
    choices = {
        "route": (route, ["rag", "conversation", "smalltalk", "out_of_scope"]),
        "model_tier": (tier, ["8b", "70b"]),
        "topic": (topic, ["session", "new_topic"]),
    }
    return {"answers": {
        name: {"type": "choice", "choice": choice, "confidence": 0.9,
               "probabilities": {option: float(option == choice) for option in options}}
        for name, (choice, options) in choices.items()
    }}


class QueryRouterTests(unittest.IsolatedAsyncioTestCase):
    async def classify(self, response, history=None):
        client = MagicMock()
        client.post = AsyncMock(return_value=MagicMock(json=lambda: response))
        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=client)
        context.__aexit__ = AsyncMock(return_value=None)
        with patch.dict("os.environ", {"JEV_API_KEY": "test-key"}), patch(
            "stages.query_router.httpx.AsyncClient", return_value=context
        ):
            result = await classify_query("Explain that", {"disease": "Parkinson's", "intent": "DBS", "location": "Toronto"}, history or [])
        return result, client

    async def test_first_medical_question_forces_rag_but_keeps_model_tier(self):
        decision, client = await self.classify(jev_response("conversation", "70b"))
        self.assertEqual((decision.route, decision.model_tier), ("rag", "70b"))
        state = client.post.call_args.kwargs["json"]["state"]
        self.assertEqual(state["intake"], {"disease": "Parkinson's", "intent": "DBS", "location": "Toronto"})
        self.assertNotIn("test-key", str(state))

    async def test_followup_can_use_conversation_and_either_model(self):
        history = [{"role": "user", "content": "Previous question"}, {"role": "assistant", "content": "Previous finding"}]
        for tier in ("8b", "70b"):
            with self.subTest(tier=tier):
                decision, _ = await self.classify(jev_response("conversation", tier), history)
                self.assertEqual((decision.route, decision.model_tier), ("conversation", tier))

    async def test_followup_without_assistant_context_requires_retrieval(self):
        decision, _ = await self.classify(jev_response("conversation"), [{"role": "user", "content": "Previous question"}])
        self.assertEqual(decision.route, "rag")

    async def test_greetings_are_exempt_from_first_turn_rag(self):
        decision, _ = await self.classify(jev_response("smalltalk"))
        self.assertEqual(decision.route, "smalltalk")

    async def test_new_topic_is_independent_of_model_tier(self):
        decision, _ = await self.classify(jev_response("rag", "8b", "new_topic"))
        self.assertEqual((decision.topic, decision.model_tier), ("new_topic", "8b"))

    async def test_invalid_decisions_fall_back_to_rag_and_70b(self):
        invalid = jev_response()
        invalid["answers"]["model_tier"]["confidence"] = 2
        for response in ({}, jev_response("invalid"), invalid):
            with self.subTest(response=response):
                decision, _ = await self.classify(response)
                self.assertEqual((decision.route, decision.model_tier), ("rag", "70b"))
                self.assertEqual(decision.fallback, "jev_unavailable_or_invalid")

    async def test_inconsistent_probabilities_are_rejected(self):
        response = jev_response()
        response["answers"]["route"]["probabilities"]["rag"] = 0
        decision, _ = await self.classify(response)
        self.assertEqual((decision.route, decision.model_tier), ("rag", "70b"))
        self.assertIsNotNone(decision.fallback)

    async def test_timeout_falls_back_without_exposing_error_text(self):
        with patch.dict("os.environ", {"JEV_API_KEY": "test-key"}), patch(
            "stages.query_router.httpx.AsyncClient", side_effect=httpx.TimeoutException("sensitive context")
        ):
            decision = await classify_query("Question", {}, [])
        self.assertEqual(decision.model_tier, "70b")
        self.assertNotIn("sensitive", decision.model_dump_json())

    def test_existing_preprovider_urgent_boundary(self):
        urgent = [
            "I have crushing chest pain and am struggling to breathe right now.",
            "My child has a swollen tongue and is having trouble breathing.",
            "I have chest pressure going into my jaw and I am sweating.",
            "Someone here is bleeding heavily after an injury right now.",
        ]
        for question in urgent:
            with self.subTest(question=question):
                self.assertEqual(immediate_response(question, [])[0], "emergency")
        for question in ("What are the signs of chest pain?", "I do not have chest pain."):
            with self.subTest(question=question):
                self.assertIsNone(immediate_response(question, []))

    def test_missing_prior_reference_clarifies_before_jev(self):
        self.assertEqual(immediate_response("Should I take the one you mentioned?", [])[0], "clarification")
        self.assertEqual(immediate_response("What did that trial you showed me say?", [])[0], "clarification")

    def test_observed_urgent_phrasings_are_intercepted(self):
        for question in (
            "I can barely breathe and my prescribed rescue inhaler is not helping. What should I do now?",
            "Their tongue is swelling and they are struggling to breathe right now.",
        ):
            with self.subTest(question=question):
                self.assertEqual(immediate_response(question, [])[0], "emergency")

    def test_urgent_words_in_negated_quoted_or_hypothetical_text_do_not_intercept(self):
        for question in (
            "I do not have trouble breathing or chest pain now.",
            "My tongue is not swelling and I am not struggling to breathe.",
            "I am not suicidal and I did not overdose.",
            "My child is not unconscious and is not having trouble breathing.",
            'The study quotes "I can barely breathe"; explain the quote.',
            "The paper says 'I cannot breathe'; explain it.",
            "Hypothetically, what if someone's tongue is swelling?",
            "If someone had trouble breathing, what would you do?",
            "What are the signs of severe bleeding?",
        ):
            with self.subTest(question=question):
                self.assertIsNone(immediate_response(question, []))

    def test_real_symptom_after_quoted_text_still_intercepts(self):
        question = 'The article says "I cannot breathe". My child is struggling to breathe now.'
        self.assertEqual(immediate_response(question, [])[0], "emergency")
        self.assertEqual(immediate_response("I can barely breathe and imagine it might be asthma.", [])[0], "emergency")


if __name__ == "__main__":
    unittest.main()
