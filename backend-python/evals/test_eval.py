"""Offline regression tests. No application imports, provider calls or DeepEval dependency."""

import copy
import csv
import io
import json
import tempfile
import unittest
from collections import Counter, defaultdict
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch, sentinel

from .gemini_backend import EvalGeminiBackend, answer_generation_cost, usage_record

from .capture import answer_text, event_data
from .dataset import load_dataset, question_input, rubric
from .run_eval import (application_ttft, build_metric, distribution, grade, judge_api_key, main, metric_specs,
                       operational_summary, validate_capture, write_operational_csv)


def capture_row():
    return {"case_id": "fixture", "status": "success", "retrieval_k": 5,
            "context_capture_complete": True, "generator_context": ["First record", "Second record"],
            "retrieval_context": ["First record", "Second record"],
            "actual_output": "Synthetic answer", "routing": {"route": "rag", "retrieval_used": True},
            "latency_seconds": 2.0, "user_visible_ttft_seconds": None,
            "cost_per_query_usd": None, "cost_status": "unknown"}


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.evidence, cls.cases = load_dataset()

    def test_testing_only_dataset_counts(self):
        self.assertEqual(len(self.cases), 150)
        self.assertEqual(len({case["id"] for case in self.cases}), 150)
        self.assertEqual(Counter(case["difficulty"] for case in self.cases),
                         {"easy": 50, "medium": 58, "hard": 42})
        self.assertTrue(all("split" not in case for case in self.cases))
        self.assertNotIn("development_turns", self.manifest["counts"])
        self.assertNotIn("held_out_turns", self.manifest["counts"])
        self.assertEqual(self.manifest["retrieval"]["k"], 5)
        self.assertEqual(sum(case.get("static_correctness_applicable") is False
                             for case in self.cases), 6)

    def test_patient_coverage_and_fresh_ids(self):
        standalone = [case for case in self.cases if not case.get("thread_id")]
        self.assertEqual(len(standalone), 90)
        self.assertEqual(Counter(case["category"] for case in standalone), {
            "symptoms_and_diagnosis": 15, "treatment_and_lifestyle": 20,
            "side_effects_and_medicines": 10, "clinical_trials": 10,
            "emergency_and_reliability": 15, "targeted_safety": 20,
        })
        self.assertEqual(self.manifest["dataset_id"], "curalink-patient-golden-v2")
        for case in self.cases:
            self.assertTrue(case["id"].startswith(("P0", "PC")))
            self.assertNotRegex(case["question"], r"(?i)\bPMID\b|pubmed\.ncbi")
            self.assertEqual(case["evaluation_scope"], "answer_only")
            self.assertNotIn("requires_node_capture", case)
            self.assertNotIn("privacy_fixture", case)

    def test_history_retains_patient_source_platforms(self):
        case = next(case for case in self.cases if case["id"] == "PC01-T2")
        source = case["history"][1]["structuredResponse"]["insights"][0]["source_details"][0]
        self.assertEqual(source["platform"], "NIDDK")
        self.assertIn("niddk.nih.gov", source["url"])

    def test_targeted_safety_questions_are_loaded(self):
        safety = [case for case in self.cases if case.get("safety_categories")]
        self.assertEqual(Counter(category for case in safety for category in case["safety_categories"]),
                         {"toxicity": 10, "scope_adherence": 5, "pii_leakage": 5})
        names = {"toxicity": "toxicity", "scope_adherence": "scope_adherence", "pii_leakage": "response_pii"}
        for case in safety:
            self.assertEqual(set(metric_specs(case, capture_row())), {names[category] for category in case["safety_categories"]})
            self.assertEqual(case["evaluation_scope"], "answer_only")
        ordinary = [case for case in self.cases if not case.get("safety_categories")]
        self.assertEqual(len(ordinary), 130)
        for case in ordinary:
            self.assertFalse(set(metric_specs(case, capture_row())) & set(names.values()))

    def test_canonical_history_has_only_preceding_turns(self):
        threads = defaultdict(list)
        for case in self.cases:
            if case.get("thread_id"):
                threads[case["thread_id"]].append(case)
        self.assertEqual(len(threads), 15)
        for turns in threads.values():
            self.assertEqual(len(turns), 4)
            for index, turn in enumerate(turns):
                self.assertEqual(len(turn["history"]), index * 2)
                self.assertEqual([message["content"] for message in turn["history"]
                                  if message["role"] == "user"],
                                 [prior["question"] for prior in turns[:index]])
                for message in turn["history"]:
                    for insight in message.get("structuredResponse", {}).get("insights", []):
                        for source in insight["source_details"]:
                            self.assertLessEqual(len(source["snippet"]), 800)

    def test_private_intake_is_not_forwarded(self):
        static, _ = question_input({"intake": {"disease": "test", "patientName": "SYNTHETIC_MARKER"}})
        self.assertEqual(static, {"disease": "test", "intent": "", "location": ""})
        self.assertNotIn("SYNTHETIC_MARKER", json.dumps(static))

    def test_rubric_separates_requirements_and_forbidden_claims(self):
        case = {"id": "fixture", "reference_answer": "Expected", "forbidden_claims": ["Invented cure"]}
        text = rubric(case, {}, {"global_boundaries": ["Do not invent results."]})
        self.assertIn("Safety and evidence requirements: Do not invent results.", text)
        self.assertIn("Forbidden claims: Invented cure", text)


class MetricTests(unittest.TestCase):
    def test_history_answers_do_not_receive_retrieval_metrics(self):
        row = capture_row()
        row["routing"] = {"retrieval_used": False}
        specs = metric_specs({}, row)
        self.assertIn("faithfulness", specs)
        self.assertNotIn("contextual_precision", specs)
        self.assertNotIn("contextual_recall", specs)

    def test_dynamic_and_policy_cases_have_no_static_retrieval_grade(self):
        for case in ({"static_correctness_applicable": False}, {"retrieval_metrics_applicable": False}):
            specs = metric_specs(case, capture_row())
            self.assertIn("behavior", specs)
            self.assertNotIn("correctness", specs)
            self.assertNotIn("contextual_precision", specs)
            self.assertNotIn("contextual_recall", specs)

    def test_retrieval_precision_and_recall_are_disabled(self):
        self.assertEqual(set(metric_specs({}, capture_row())),
                         {"correctness", "completeness", "answer_relevancy", "faithfulness"})

    def test_gemini_judge_uses_gemini_or_google_api_key(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "gemini-key", "GOOGLE_API_KEY": "google-key"}, clear=False):
            self.assertEqual(judge_api_key("gemini"), "gemini-key")
        with patch.dict("os.environ", {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": "google-key"}, clear=False):
            self.assertEqual(judge_api_key("gemini"), "google-key")

    def test_safety_factories_use_dedicated_metrics_and_explicit_judge(self):
        toxicity, pii, scope = Mock(), Mock(), Mock()
        metrics = SimpleNamespace(AnswerRelevancyMetric=Mock(), ContextualPrecisionMetric=Mock(),
                                  ContextualRecallMetric=Mock(), FaithfulnessMetric=Mock(), GEval=scope,
                                  ToxicityMetric=toxicity, PIILeakageMetric=pii)
        with patch.dict("sys.modules", {"deepeval": SimpleNamespace(), "deepeval.metrics": metrics,
                                       "deepeval.test_case": SimpleNamespace(SingleTurnParams=Mock())}):
            for name in ("toxicity", "response_pii", "scope_adherence"):
                build_metric(name, sentinel.judge, .7)
        for factory in (toxicity, pii):
            factory.assert_called_once_with(include_reason=True, model=sentinel.judge, threshold=.7, async_mode=False)
        self.assertEqual(scope.call_args.kwargs["name"], "scope_adherence")
        self.assertIs(scope.call_args.kwargs["model"], sentinel.judge)

    def test_privacy_case_scores_only_the_returned_answer(self):
        case = {"id": "fixture", "difficulty": "easy", "question": "Do not disclose private records", "safety_categories": ["pii_leakage"],
                "routing_scored": False, "reference_answer": "Do not leak private data"}
        metric = SimpleNamespace(score=1.0, reason="No PII in the response", evaluation_cost=0,
                                 measure=Mock(), is_successful=lambda: True)
        with patch.dict("sys.modules", {"deepeval.test_case": SimpleNamespace(LLMTestCase=Mock())}), \
                patch("evals.run_eval.build_metric", return_value=metric):
            result = grade(case, capture_row(), {}, {}, sentinel.judge, .7)
        self.assertEqual(result["case_status"], "passed")
        self.assertEqual(set(result["metrics"]), {"response_pii"})
        self.assertNotIn("boundary_checks", result)
        self.assertIn("does not verify backend privacy", result["safety_scope"])

    def test_failed_application_does_not_invoke_judge(self):
        with patch("evals.run_eval.build_metric") as factory:
            result = grade({"id": "fixture", "difficulty": "easy"},
                           {"status": "timeout"}, {}, {}, sentinel.judge, .7)
        factory.assert_not_called()
        self.assertEqual(result["metrics"], {})


class GeminiGenerationTests(unittest.IsolatedAsyncioTestCase):
    def test_generation_cost_excludes_expansion_and_preserves_missing_usage(self):
        records = [{"purpose": "query_expansion", "cost_usd": 100},
                   {"purpose": "answer", "cost_usd": .002}]
        self.assertEqual(answer_generation_cost(records), .002)
        self.assertEqual(answer_generation_cost(records[:1]), 0)
        self.assertIsNone(answer_generation_cost([{"purpose": "answer", "cost_usd": None}]))

    def test_cost_counts_thinking_output_and_preserves_unknown(self):
        usage = usage_record(SimpleNamespace(prompt_token_count=1000, candidates_token_count=100, thoughts_token_count=50))
        self.assertAlmostEqual(usage["cost_usd"], .000675)
        self.assertIsNone(usage_record(None)["cost_usd"])
        with self.assertRaises(ValueError):
            usage_record(SimpleNamespace(prompt_token_count=-1, candidates_token_count=0, thoughts_token_count=0))

    async def test_stream_usage_is_recorded_once_not_summed_per_chunk(self):
        async def chunks():
            for count, text in ((1, "a"), (2, "b")):
                yield SimpleNamespace(text=text, usage_metadata=SimpleNamespace(
                    prompt_token_count=10, candidates_token_count=count, thoughts_token_count=0))
        backend = EvalGeminiBackend.__new__(EvalGeminiBackend)
        backend.usage = []
        backend._config = Mock(return_value={})
        backend.client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(
            generate_content_stream=AsyncMock(return_value=chunks()))))
        self.assertEqual([part async for part in backend.generate_stream("test")], ["a", "b"])
        self.assertEqual(len(backend.usage), 1)
        self.assertEqual(backend.usage[0]["output_tokens"], 2)
        backend._config.assert_called_once_with(None, 800, .2, False, answer=True)


class CaptureAndReportingTests(unittest.TestCase):
    def test_actual_context_order_and_cap_are_enforced(self):
        validate_capture(capture_row(), {"fixture"})
        for key, value in (("retrieval_k", 14), ("context_capture_complete", False),
                           ("generator_context", ["record"] * 6),
                           ("retrieval_context", ["Second record", "First record"])):
            row = capture_row()
            row[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_capture(row, {"fixture"})

    def test_failures_cannot_smuggle_invalid_telemetry(self):
        for value in (float("nan"), float("inf"), -1, True, "2"):
            row = capture_row()
            row.update(status="error", latency_seconds=value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_capture(row, {"fixture"})

    def test_non_null_cost_requires_provenance(self):
        row = capture_row()
        row["cost_per_query_usd"] = 0
        with self.assertRaises(ValueError):
            validate_capture(row, {"fixture"})

    def test_percentile_interpolation_and_unknown_values(self):
        self.assertEqual(distribution([None])["n"], 0)
        self.assertIsNone(distribution([])["mean"])
        stats = distribution([1, 2, 3, None])
        self.assertEqual(stats["p50"], 2)
        self.assertAlmostEqual(stats["p95"], 2.9)
        self.assertAlmostEqual(stats["p99"], 2.98)
        self.assertEqual(stats["mean"], 2)

    def test_failed_elapsed_and_unknown_cost_are_not_hidden(self):
        failure = copy.deepcopy(capture_row())
        failure.update(status="timeout", latency_seconds=10)
        summary = operational_summary([capture_row(), failure, {"status": "skipped"}])
        self.assertEqual(summary["attempted"], 2)
        self.assertEqual(summary["success_rate"], .5)
        segment = summary["segments"][0]
        self.assertEqual(segment["successful_request_latency_seconds"]["mean"], 2)
        self.assertEqual(segment["failed_elapsed_seconds"]["mean"], 10)
        self.assertEqual(segment["unknown_cost_count"], 2)
        self.assertIsNone(segment["cost_per_query_usd"]["mean"])

    def test_ttft_matches_first_application_token_and_populates_csv(self):
        row = capture_row()
        row.update(model_stream_ttft_seconds=.4, first_answer_chunk_seconds=1.9)
        self.assertEqual(application_ttft(row), .4)
        stats = operational_summary([row])["segments"][0]["ttft_seconds"]
        for key in ("p50", "p95", "p99", "mean"):
            self.assertEqual(stats[key], .4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ttft.csv"
            write_operational_csv(path, operational_summary([row]))
            with path.open(encoding="utf-8", newline="") as output:
                report = next(csv.DictReader(output))
            self.assertEqual(report["TTFT (s) p95"], "0.4")
            self.assertIn("not browser", report["TTFT scope"])
        row["ttft_seconds"] = 0
        self.assertEqual(application_ttft(row), 0)
        self.assertEqual(application_ttft({"first_answer_chunk_seconds": .8}), .8)
        self.assertIsNone(application_ttft({}))

    def test_invalid_ttft_is_rejected(self):
        for value in (-1, float("nan"), True, "1"):
            row = capture_row()
            row["ttft_seconds"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_capture(row, {"fixture"})

    def test_table_has_requested_statistics_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.csv"
            write_operational_csv(path, operational_summary([capture_row()]))
            with path.open(encoding="utf-8", newline="") as output:
                rows = list(csv.DictReader(output))
            self.assertEqual(rows[0]["Latency (s) p50"], "2.0")
            self.assertEqual(rows[0]["TTFT (s) mean"], "unknown")
            self.assertEqual(rows[0]["Cost per query ($) p99"], "unknown")
            with self.assertRaises(FileExistsError):
                write_operational_csv(path, {"segments": []})

    def test_in_process_sse_events_and_answer_text(self):
        self.assertEqual(event_data(b'event: token\ndata: "hello"\n\n'), ("token", "hello"))
        self.assertEqual(event_data('event: status\ndata: {"stage":"ranking"}\n\n')[0], "status")
        self.assertEqual(answer_text({"overview": "Summary", "insights": [{"finding": "Evidence"}]}),
                         "Summary\nEvidence")

    def test_live_and_judge_opt_ins_are_required_before_loading_data(self):
        for mode in ("collect", "score"):
            with patch("evals.run_eval.load_dataset") as loader, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main([mode, "--output", "never-created.jsonl"])
                self.assertEqual(error.exception.code, 2)
                loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
