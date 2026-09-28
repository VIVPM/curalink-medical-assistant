"""Tests citation and recommendation safety in assembled responses."""

import unittest

from schemas.document import Document
from stages.llm_reasoner import _repair_schema
from stages.response_assembler import assemble_response


def publication():
    return Document(
        doc_id="pubmed:1",
        doc_type="publication",
        title="Example study",
        abstract="The study reported an association between exercise and improved outcomes.",
        year=2025,
        url="https://pubmed.ncbi.nlm.nih.gov/1/",
        sources=["pubmed"],
    )


class ResponseSafetyTests(unittest.TestCase):
    def test_only_safe_source_linked_recommendations_are_returned(self):
        result = assemble_response(
            {
                "overview": "Overview",
                "insights": [],
                "trials": [],
                "recommendations": [
                    {"text": "Discuss the exercise findings with a qualified clinician.", "sources": ["doc1"]},
                    {"text": "Discuss an unsupported idea.", "sources": ["missing"]},
                    {"text": "Take 5 mg daily.", "sources": ["doc1"]},
                ],
                "follow_up_questions": [],
                "abstain_reason": None,
            },
            {"doc1": publication()},
        ).user_facing_json

        self.assertEqual(len(result["recommendations"]), 1)
        self.assertEqual(result["recommendations"][0]["text"], "Discuss the exercise findings with a qualified clinician.")
        self.assertTrue(result["recommendations"][0]["source_details"])
        self.assertIn("uncited_recommendation_removed", result["pipelineMeta"]["warnings"])
        self.assertIn("unsafe_recommendation_removed", result["pipelineMeta"]["warnings"])

    def test_legacy_recommendation_strings_are_dropped(self):
        repaired = _repair_schema({
            "overview": "Overview",
            "insights": [],
            "trials": [],
            "recommendations": ["Start treatment"],
            "abstain_reason": None,
        })
        self.assertEqual(repaired["recommendations"], [])


if __name__ == "__main__":
    unittest.main()
