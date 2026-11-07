"""Offline contracts for the frozen routing audit."""

import unittest

from .routing_audit import confusion, normalize_route, project_saved_routes, summarize


class RoutingAuditTests(unittest.TestCase):
    def test_route_normalization(self):
        self.assertEqual(normalize_route("emergency_intercept"), "urgent")
        self.assertEqual(normalize_route("conversation"), "history-only")
        self.assertEqual(normalize_route("rag"), "full RAG")

    def test_confusion_and_safety_denominators(self):
        labels = [
            {"case_id": "a", "expected_route": "urgent", "expected_tier": None, "ambiguous": False},
            {"case_id": "b", "expected_route": "full RAG", "expected_tier": "70b", "ambiguous": False},
            {"case_id": "c", "expected_route": "history-only", "expected_tier": "8b", "ambiguous": True},
        ]
        captures = {
            "a": {"case_id": "a", "status": "success", "routing": {"route": "rag", "model_tier": "70b"}},
            "b": {"case_id": "b", "status": "success", "routing": {"route": "conversation", "model_tier": "8b"}},
            "c": {"case_id": "c", "status": "error", "routing": {}},
        }
        rows, report = summarize(labels, captures)
        self.assertEqual(confusion(rows)["urgent"]["full RAG"], 1)
        self.assertEqual(report["urgent_false_negatives"], ["a"])
        self.assertEqual(report["unsupported_history_only"], ["b"])
        self.assertEqual(report["route_scored_denominator"], 2)
        self.assertEqual(report["tier_scored_denominator"], 1)

    def test_code_recheck_changes_routes_without_claiming_new_answers(self):
        labels = [
            {"case_id": "P058", "expected_route": "urgent", "ambiguous": False},
            {"case_id": "PC08-T4", "expected_route": "urgent", "ambiguous": False},
            {"case_id": "PC02-T3", "expected_route": "full RAG", "ambiguous": False},
            {"case_id": "old", "expected_route": "history-only", "ambiguous": False},
        ]
        questions = {
            "P058": "I can barely breathe and my inhaler is not helping.",
            "PC08-T4": "Their tongue is swelling and they are struggling to breathe right now.",
            "PC02-T3": "It is now waking me at night and I need the inhaler more.",
            "old": "Summarize that study again.",
        }
        cases = {key: {"question": question, "history": (
            [{"role": "assistant", "content": "Saved study summary"}] if key == "old" else [])}
            for key, question in questions.items()}
        captures = {key: {"status": "success", "routing": {"route": route}}
                    for key, route in (("P058", "rag"), ("PC08-T4", "conversation"),
                                       ("PC02-T3", "conversation"), ("old", "conversation"))}
        rows, report = project_saved_routes(labels, captures, cases)
        self.assertEqual(report["before_route_correct"], 1)
        self.assertEqual(report["projected_route_correct"], 4)
        self.assertEqual(report["before_urgent_false_negatives"], ["P058", "PC08-T4"])
        self.assertEqual(report["projected_urgent_false_negatives"], [])
        self.assertEqual(report["changed_case_ids"], ["P058", "PC02-T3", "PC08-T4"])
        self.assertEqual(report["scope"], "code-level reclassification of saved inputs; no new model answer")
        self.assertTrue(all(row["new_answer_captured"] is False for row in rows))


if __name__ == "__main__":
    unittest.main()
