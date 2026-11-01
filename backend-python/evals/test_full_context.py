"""Offline checks for complete selected records and explicitly labelled token measurements."""
import unittest

from schemas.document import Document
from stages.context_builder import build_context
from .run_eval import context_token_maxima, validate_capture
from .capture import select_threshold_records
from types import SimpleNamespace


class FullRecordContextTests(unittest.TestCase):
    def test_threshold_selection_preserves_boundary_and_type_caps(self):
        docs = [Document(doc_id=f'p{i}', doc_type='publication', title='Publication') for i in range(8)]
        docs += [Document(doc_id=f't{i}', doc_type='trial', title='Trial') for i in range(6)]
        scores = {doc.doc_id: 0.9 for doc in docs}
        scores['p0'] = 0.799
        scores['t0'] = 0.8
        result = SimpleNamespace(top_docs=docs, scores_by_doc_id=scores)
        selected = select_threshold_records(result)
        self.assertEqual(sum(d.doc_type == 'publication' for d in selected), 5)
        self.assertEqual(sum(d.doc_type == 'trial' for d in selected), 6)
        self.assertNotIn('p0', [d.doc_id for d in selected])
        self.assertIn('t0', [d.doc_id for d in selected])
        result.scores_by_doc_id = {doc.doc_id: 0.79 for doc in docs}
        self.assertEqual(select_threshold_records(result), [])

    def test_explicit_threshold_capture_accepts_eleven_records(self):
        from .test_eval import capture_row
        row = capture_row()
        row.update(retrieval_k=11, selection_policy={'score': 'query_relative_combined', 'minimum': 0.8,
                   'max_publications': 5, 'max_trials': 6})
        row['generator_context'] = row['retrieval_context'] = ['record'] * 11
        validate_capture(row, {'fixture'})
        row['generator_context'] = ['record'] * 12
        with self.assertRaises(ValueError):
            validate_capture(row, {'fixture'})

    def test_publication_end_is_preserved_beyond_old_cutoff(self):
        abstract = "Background. " * 400 + "IMPORTANT_FINAL_FINDING"
        doc = Document(doc_id="publication", doc_type="publication", title="Example", abstract=abstract,
                       authors=[f"Author {i}" for i in range(7)], sources=["pubmed"])
        payload = build_context([doc], "What was found?", {})
        self.assertIn(abstract, payload.user_prompt)
        self.assertIn("Author 6", payload.user_prompt)
        self.assertEqual(payload.truncations, [])
        self.assertGreater(payload.token_count, 0)

    def test_trial_eligibility_outcomes_and_locations_are_preserved(self):
        eligibility = "Detailed eligibility. " * 300 + "FINAL_EXCLUSION"
        doc = Document(doc_id="trial", doc_type="trial", title="Example trial", nct_id="NCT-SYNTHETIC",
                       abstract="Summary. " * 300, eligibility_criteria=eligibility,
                       primary_outcomes=[f"Outcome {i}" for i in range(6)],
                       locations=[{"facility": f"Facility {i}", "country": "Example"} for i in range(6)])
        payload = build_context([doc], "Who can join?", {})
        for text in (doc.abstract, eligibility, "Outcome 5", "Facility 5"):
            self.assertIn(text, payload.user_prompt)
        self.assertEqual(payload.truncations, [])

    def test_records_estimates_and_actual_total_input_are_not_confused(self):
        rows = [{"case_id": "P001", "context_measurements": {
            "record_tokens_estimate": 100, "total_prompt_tokens_estimate": 200},
            "answer_generation_usage": [{"input_tokens": 250}]},
            {"case_id": "P002", "context_measurements": {
                "record_tokens_estimate": 300, "total_prompt_tokens_estimate": 400},
             "answer_generation_usage": [{"input_tokens": 450}]}]
        summary = context_token_maxima(rows)
        self.assertEqual(summary["max_selected_records_tokens_estimate"], {"case_id": "P002", "tokens": 300})
        self.assertEqual(summary["max_total_input_tokens_actual"], {"case_id": "P002", "tokens": 450})
        self.assertIsNone(context_token_maxima([])["max_total_input_tokens_actual"])


if __name__ == "__main__":
    unittest.main()
