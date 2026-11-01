"""Exercise routing through all three pipeline entry points without network calls."""

import importlib
import json
import os
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

os.environ.setdefault("INTERNAL_API_KEY", "internal-auth-test-key")
os.environ.setdefault("LLM_MODEL", "test/model")
os.environ.setdefault("HF_TOKEN", "test-token")

from llm_backend import HFBackend
from schemas.document import Document
from stages.query_router import RoutingDecision
from test_history_routing import research_history

main = importlib.import_module("main")


class PipelineRoutingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.base = HFBackend("hf_user", "old/model", use_cache=False)
        self.doc = Document("pubmed:123", "publication", "DBS study", abstract="Participants had improved symptoms.", url="https://pubmed.ncbi.nlm.nih.gov/123/", sources=["pubmed"])
        self.output = {"overview": "The study reported improved symptoms.", "insights": [{"finding": "Improved symptoms", "sources": ["doc1"]}], "trials": [], "recommendations": [], "follow_up_questions": [], "abstain_reason": None}
        self.generated_models = []

    async def invoke(self, mode, history=None):
        req = main.PipelineRequest(tenant="user-a", static={"disease": "Parkinson's", "intent": "DBS", "location": ""}, dynamic={"recentMessages": history or []}, current={"userMessage": "Explain the evidence"})
        request = SimpleNamespace(headers={"x-request-id": "routing-test"})
        if mode == "run":
            return await main.pipeline_run(req, request)
        if mode == "job":
            return await main._run_pipeline_for_job({**req.model_dump(), "_job_id": "routing-test-job"})
        response = await main.pipeline_stream(req, request)
        self.events = [event async for event in response.body_iterator]
        metadata = [event for event in self.events if event.startswith("event: metadata")]
        self.assertEqual(len(metadata), 1)
        self.assertEqual(self.events[-1], "event: done\ndata: {}\n\n")
        return json.loads(metadata[0].split("data: ", 1)[1])

    def setup_pipeline(self, stack, decision):
        stack.enter_context(patch("routing.routing_enabled", return_value=True))
        stack.enter_context(patch("main.routing_enabled", return_value=True))
        classify = stack.enter_context(patch("routing.classify_query", AsyncMock(return_value=decision)))
        stack.enter_context(patch.object(main, "pipeline_clients", return_value=("embedder", "reranker", self.base, True)))
        stack.enter_context(patch.object(main, "llm", self.base))
        stack.enter_context(patch.dict(main.models, {"embedder": "embedder", "reranker": "reranker"}))
        stack.enter_context(patch("checkpoint.load", return_value=None))
        save = stack.enter_context(patch("checkpoint.save"))
        stack.enter_context(patch("checkpoint.clear"))
        expansion = stack.enter_context(patch.object(main, "expand_query", AsyncMock(return_value=SimpleNamespace(expanded_queries=["DBS Parkinson's"], disease_focus="Parkinson's", skip_retrieval=False))))
        fetch = stack.enter_context(patch.object(main, "fetch_pubmed", AsyncMock(return_value=[{}])))
        stack.enter_context(patch.object(main, "fetch_openalex", AsyncMock(return_value=[])))
        trials = stack.enter_context(patch.object(main, "fetch_trials", AsyncMock(return_value=[])))
        stack.enter_context(patch.object(main, "normalize_pubmed", return_value=self.doc))
        stack.enter_context(patch.object(main, "run_ranking", return_value=SimpleNamespace(top_docs=[self.doc])))
        reasoner = stack.enter_context(patch.object(main, "run_reasoner", AsyncMock(return_value=SimpleNamespace(llm_output=self.output))))
        history_reasoner = stack.enter_context(patch("routing.run_reasoner", AsyncMock(return_value=SimpleNamespace(llm_output=self.output))))
        output = json.dumps(self.output)
        generated_models = self.generated_models

        async def stream(backend, *args, **kwargs):
            generated_models.append(backend.model)
            yield output

        stack.enter_context(patch.object(HFBackend, "generate_stream", stream))
        return classify, expansion, fetch, trials, reasoner, history_reasoner, save

    async def test_rag_uses_8b_expansion_and_70b_answer_in_every_path(self):
        for mode in ("run", "stream", "job"):
            with self.subTest(mode=mode), ExitStack() as stack:
                _, expansion, fetch, _, reasoner, history_reasoner, save = self.setup_pipeline(stack, RoutingDecision(model_tier="70b"))
                response = await self.invoke(mode)
                self.assertIn("8B", expansion.call_args.kwargs["llm"].model)
                if mode == "stream":
                    self.assertIn("70B", self.generated_models[-1])
                else:
                    self.assertIn("70B", reasoner.call_args.kwargs["llm"].model)
                fetch.assert_awaited_once()
                history_reasoner.assert_not_called()
                self.assertEqual(response["pipelineMeta"]["routing"]["model_tier"], "70b")
                self.assertTrue(response["pipelineMeta"]["routing"]["retrieval_used"])
                if mode == "job":
                    self.assertTrue(any(call.args[1] == "routing" for call in save.call_args_list))

    async def test_history_only_skips_expansion_and_retrieval_in_every_path(self):
        for mode in ("run", "stream", "job"):
            with self.subTest(mode=mode), ExitStack() as stack:
                _, expansion, fetch, _, reasoner, history_reasoner, _ = self.setup_pipeline(stack, RoutingDecision(route="conversation", model_tier="8b"))
                response = await self.invoke(mode, research_history())
                expansion.assert_not_called()
                fetch.assert_not_called()
                reasoner.assert_not_called()
                history_reasoner.assert_awaited_once()
                self.assertIn("8B", history_reasoner.call_args.args[1].model)
                self.assertEqual(response["pipelineMeta"]["routing"]["route"], "conversation")
                self.assertFalse(response["pipelineMeta"]["routing"]["retrieval_used"])

    async def test_new_topic_controls_trial_search_and_context_in_every_path(self):
        for mode in ("run", "stream", "job"):
            with self.subTest(mode=mode), ExitStack() as stack:
                _, expansion, _, trials, _, _, _ = self.setup_pipeline(stack, RoutingDecision(topic="new_topic", model_tier="8b"))
                expansion.return_value = SimpleNamespace(expanded_queries=["Alzheimer's research"], disease_focus="Alzheimer's", skip_retrieval=False)
                await self.invoke(mode, research_history())
                self.assertEqual(expansion.call_args.kwargs["static_context"]["disease"], "")
                self.assertEqual(expansion.call_args.kwargs["static_context"]["intent"], "")
                self.assertEqual(trials.call_args.kwargs["disease"], "Alzheimer's")

    async def test_expander_cannot_override_first_question_rag_decision(self):
        for mode in ("run", "stream", "job"):
            with self.subTest(mode=mode), ExitStack() as stack:
                _, expansion, fetch, _, _, _, _ = self.setup_pipeline(stack, RoutingDecision(model_tier="8b"))
                expansion.return_value.skip_retrieval = True
                response = await self.invoke(mode)
                fetch.assert_awaited_once()
                self.assertEqual(response["pipelineMeta"]["routing"]["route"], "rag")

    async def test_abstained_history_attempt_retrieves_in_every_path(self):
        for mode in ("run", "stream", "job"):
            with self.subTest(mode=mode), ExitStack() as stack:
                _, expansion, fetch, _, _, history_reasoner, _ = self.setup_pipeline(stack, RoutingDecision(route="conversation"))
                history_reasoner.return_value = SimpleNamespace(llm_output={"abstain_reason": "Missing evidence"})
                response = await self.invoke(mode, research_history())
                expansion.assert_awaited_once()
                fetch.assert_awaited_once()
                self.assertEqual(response["pipelineMeta"]["routing"]["route"], "rag")
                self.assertEqual(response["pipelineMeta"]["routing"]["fallback"], "history_insufficient")


if __name__ == "__main__":
    unittest.main()
