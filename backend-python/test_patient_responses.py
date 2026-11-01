"""Offline checks for patient-first prompts and pre-retrieval response guards."""
import unittest
from unittest.mock import AsyncMock, patch
from llm_backend import HFBackend
from routing import immediate_response, route_request
from stages.context_builder import GROUNDING_RULES
from stages.query_expander import SYSTEM_PROMPT


class PatientPromptTests(unittest.TestCase):
    def test_prompt_does_not_force_paper_or_trial_quotas(self):
        self.assertIn('patient-facing', GROUNDING_RULES)
        self.assertIn('There is no minimum', GROUNDING_RULES)
        self.assertIn('planned outcome', GROUNDING_RULES)
        self.assertNotIn('aim for 6', GROUNDING_RULES)
        self.assertNotIn('exactly 2', GROUNDING_RULES)
        self.assertIn('query1 the most direct', SYSTEM_PROMPT)
        self.assertIn('named medicine plus the specific adverse effect', SYSTEM_PROMPT)

    def test_urgent_symptoms_and_missing_references(self):
        for message in [
            'My child has a swollen tongue and trouble breathing after peanuts.',
            'I have chest pressure and sweating now.',
            'My father with heart failure is suddenly extremely breathless.',
            'Someone is bleeding heavily now.',
            "I have no chest pain but I cannot breathe.",
        ]:
            with self.subTest(message=message):
                self.assertEqual(immediate_response(message, [])[0], 'emergency')
        self.assertEqual(immediate_response('What did that trial you showed me say?', [])[0], 'clarification')

    def test_nonurgent_questions_do_not_trigger_guard(self):
        for message in ['What are the signs of chest pain?', 'I have no chest pain now.',
                        'My doctor said I have prediabetes. What does it mean?',
                        'What are symptoms of anaphylaxis?']:
            with self.subTest(message=message):
                self.assertIsNone(immediate_response(message, []))
        self.assertIsNone(immediate_response('Explain that study', [{'role':'assistant','content':'Study details'}]))


class DirectResponseTests(unittest.IsolatedAsyncioTestCase):
    async def test_guard_runs_without_classifier_or_generation(self):
        base = HFBackend('synthetic', 'test/model', use_cache=False)
        for enabled in (True, False):
            with patch('routing.routing_enabled', return_value=enabled), patch('routing.classify_query', AsyncMock()) as classify, patch('routing.run_reasoner', AsyncMock()) as reasoner:
                response = await route_request('I have crushing chest pain now', {}, [], base)
                self.assertEqual(response.metadata()['route'], 'emergency')
                self.assertFalse(response.metadata()['retrieval_used'])
                self.assertIn('emergency number now', response.response['overview'])
                classify.assert_not_called()
                reasoner.assert_not_called()


if __name__ == '__main__':
    unittest.main()
