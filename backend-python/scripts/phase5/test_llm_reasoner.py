"""
Standalone test for Phase 5 Step 5.3 — LLM Reasoner (Stage 6).
Run from the backend-python directory:
    python scripts/phase5/test_llm_reasoner.py

Builds a real PromptPayload with mock docs, sends to Groq, and verifies
the response is valid structured JSON with overview, insights, trials,
and source citations using [docN] anchors.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio
from dotenv import load_dotenv

from schemas.document import Document
from llm_backend import get_llm_backend
from stages.context_builder import build_context
from stages.llm_reasoner import run_reasoner

load_dotenv()


def make_mock_docs() -> list[Document]:
    """6 publications + 2 trials about Parkinson's + Vitamin D."""
    docs = [
        Document(
            doc_id="pubmed:11111", doc_type="publication",
            title="Vitamin D supplementation in Parkinson's disease: a randomized controlled trial",
            abstract="This RCT enrolled 120 Parkinson's patients and found that daily vitamin D "
                     "supplementation (2000 IU) significantly improved bone mineral density and "
                     "reduced fall risk over 12 months compared to placebo (p<0.01). Motor symptoms "
                     "measured by UPDRS showed modest improvement in the treatment group.",
            authors=["Smith J", "Patel K"], year=2023, sources=["pubmed"],
            url="https://pubmed.ncbi.nlm.nih.gov/11111",
            disease_tags=["parkinson"],
        ),
        Document(
            doc_id="pubmed:22222", doc_type="publication",
            title="Vitamin D deficiency prevalence in neurodegenerative diseases",
            abstract="A systematic review of 45 studies found that 60-80% of Parkinson's disease "
                     "patients have vitamin D deficiency compared to 20-40% in age-matched controls. "
                     "Low vitamin D levels correlated with worse motor outcomes and faster disease progression.",
            authors=["Chen L", "Wang M"], year=2024, sources=["pubmed"],
            url="https://pubmed.ncbi.nlm.nih.gov/22222",
            disease_tags=["parkinson"],
        ),
        Document(
            doc_id="openalex:W33333", doc_type="publication",
            title="Neuroprotective effects of cholecalciferol in animal models of Parkinson's",
            abstract="In a mouse model of PD, vitamin D3 (cholecalciferol) administration showed "
                     "neuroprotective effects on dopaminergic neurons. Treatment reduced neuroinflammation "
                     "markers and oxidative stress. These preclinical findings suggest a potential "
                     "disease-modifying role that warrants clinical investigation.",
            authors=["Garcia R"], year=2022, sources=["openalex"],
            url="https://openalex.org/W33333",
            disease_tags=["parkinson"],
        ),
        Document(
            doc_id="pubmed:44444", doc_type="publication",
            title="Safety profile of vitamin D supplementation in elderly patients on levodopa",
            abstract="This observational study of 200 PD patients taking levodopa found no significant "
                     "drug interactions with vitamin D supplementation up to 4000 IU daily. Calcium "
                     "levels remained within normal range. However, patients on calcium channel blockers "
                     "should monitor serum calcium more closely.",
            authors=["Brown A", "Lee S"], year=2023, sources=["pubmed"],
            url="https://pubmed.ncbi.nlm.nih.gov/44444",
            disease_tags=["parkinson"],
        ),
        Document(
            doc_id="pubmed:55555", doc_type="publication",
            title="Deep brain stimulation outcomes and nutritional factors in Parkinson's",
            abstract="A retrospective analysis of 85 DBS patients showed that those with adequate "
                     "vitamin D levels (>30 ng/mL) had better post-surgical motor outcomes at 6 months "
                     "compared to deficient patients. Nutritional optimization before DBS surgery may "
                     "improve outcomes.",
            authors=["Kim J", "Park H"], year=2024, sources=["pubmed"],
            url="https://pubmed.ncbi.nlm.nih.gov/55555",
            disease_tags=["parkinson"],
        ),
        Document(
            doc_id="openalex:W66666", doc_type="publication",
            title="Exercise and vitamin D: combined effects on Parkinson's gait and balance",
            abstract="A 6-month trial combining structured exercise with vitamin D supplementation "
                     "in PD patients showed synergistic improvements in gait speed and balance "
                     "compared to either intervention alone.",
            authors=["Taylor E"], year=2023, sources=["openalex"],
            url="https://openalex.org/W66666",
            disease_tags=["parkinson"],
        ),
        Document(
            doc_id="nct:NCT09999991", doc_type="trial",
            title="Vitamin D3 Supplementation in Early Parkinson's Disease",
            abstract="This phase 3 trial evaluates whether high-dose vitamin D3 can slow "
                     "disease progression in early-stage PD patients over 24 months.",
            nct_id="NCT09999991", status="RECRUITING",
            eligibility_criteria="Inclusion: Age 45-75, PD diagnosis within 3 years, "
                                 "Hoehn and Yahr stage 1-2. Exclusion: Hypercalcemia, "
                                 "kidney disease, current vitamin D >1000 IU/day.",
            primary_outcomes=["Change in MDS-UPDRS total score at 24 months"],
            locations=[{"facility": "Toronto Western Hospital", "country": "Canada"}],
            sources=["clinicaltrials"], disease_tags=["parkinson"],
        ),
        Document(
            doc_id="nct:NCT09999992", doc_type="trial",
            title="Nutritional Optimization Before DBS Surgery",
            abstract="Evaluating whether pre-surgical vitamin D and nutritional optimization "
                     "improves DBS outcomes in Parkinson's patients.",
            nct_id="NCT09999992", status="COMPLETED",
            eligibility_criteria="Inclusion: Scheduled for DBS, PD diagnosis. "
                                 "Exclusion: Severe dementia.",
            primary_outcomes=["Post-DBS motor improvement at 6 months"],
            locations=[{"facility": "Johns Hopkins", "country": "USA"}],
            sources=["clinicaltrials"], disease_tags=["parkinson"],
        ),
    ]
    return docs


async def run_test():
    print("=" * 60)
    print("Phase 5 Step 5.3: LLM Reasoner Test")
    print("=" * 60)

    llm = get_llm_backend()
    docs = make_mock_docs()

    static_context = {
        "disease": "Parkinson's disease",
        "intent": "Deep Brain Stimulation",
        "location": "Toronto, Canada",
        "patientName": "John Smith",
    }
    chat_history = [
        {"role": "user", "content": "Tell me about DBS treatment options"},
        {"role": "assistant", "content": "Deep brain stimulation has shown significant benefits..."},
    ]

    # Build context (Stage 5)
    print("\n[1/3] Building prompt payload")
    payload = build_context(
        top_docs=docs,
        user_message="Can I take Vitamin D?",
        static_context=static_context,
        chat_history=chat_history,
    )
    print(f"  Token estimate: {payload.token_count}")
    print(f"  Doc anchors: {list(payload.doc_anchors.keys())}")

    # Run reasoner (Stage 6)
    print("\n[2/3] Sending to LLM (Groq)")
    result = await run_reasoner(payload, llm)
    print(f"  Timing: {result.timing_ms}ms")
    print(f"  Retried: {result.retried}")
    if result.parse_error:
        print(f"  Parse error: {result.parse_error}")

    # Validate output
    print(f"\n[3/3] Validating output")
    output = result.llm_output

    # 1. Required keys
    required = {"overview", "insights", "trials", "abstain_reason"}
    missing = required - set(output.keys())
    if not missing:
        print("  PASS: all required keys present")
    else:
        print(f"  FAIL: missing keys: {missing}")

    # 2. Overview
    overview = output.get("overview", "")
    if overview and len(overview) > 20:
        print(f"  PASS: overview present ({len(overview)} chars)")
    else:
        print(f"  WARN: overview too short or missing")

    # 3. Insights with citations
    insights = output.get("insights", [])
    print(f"  Insights: {len(insights)}")
    cited_anchors = set()
    for i, ins in enumerate(insights):
        finding = ins.get("finding", "")[:80]
        sources = ins.get("sources", [])
        cited_anchors.update(sources)
        print(f"    {i+1}. {finding}...")
        print(f"       Sources: {sources}")

    if insights:
        print("  PASS: insights populated")
    else:
        print("  WARN: no insights returned")

    # 4. Trials
    trials = output.get("trials", [])
    print(f"  Trials: {len(trials)}")
    for t in trials:
        nct = t.get("nct_id", "?")
        title = (t.get("title", "") or "")[:60]
        sources = t.get("sources", [])
        cited_anchors.update(sources)
        print(f"    - {nct}: {title} (sources: {sources})")

    # 5. Citations reference valid anchors
    valid_anchors = set(payload.doc_anchors.keys())
    invalid = cited_anchors - valid_anchors
    if not invalid:
        print(f"  PASS: all cited sources ({cited_anchors}) are valid anchors")
    else:
        print(f"  WARN: invalid citations: {invalid} (not in {valid_anchors})")

    # 6. Abstain
    abstain = output.get("abstain_reason")
    if abstain is None:
        print("  PASS: abstain_reason is null (LLM answered the question)")
    else:
        print(f"  INFO: abstain_reason set: {abstain}")

    print(f"\n  Raw overview: {overview[:200]}")

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    asyncio.run(run_test())
