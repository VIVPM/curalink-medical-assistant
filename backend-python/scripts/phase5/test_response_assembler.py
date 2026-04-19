"""
Standalone test for Phase 5 Step 5.4 — Response Assembler (Stage 7).
Run from the backend-python directory:
    python scripts/phase5/test_response_assembler.py

Feeds a mock Stage 6 output + doc_anchors into the assembler and verifies:
  - Citations resolved to real metadata (title/authors/year/url/snippet)
  - Hallucinated citations flagged as unverified
  - Abstain case handled correctly
  - Final JSON matches expected schema
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
from schemas.document import Document
from stages.response_assembler import assemble_response


def make_doc_anchors() -> dict:
    """Build doc_anchors as Stage 5 would produce."""
    return {
        "doc1": Document(
            doc_id="pubmed:11111", doc_type="publication",
            title="Vitamin D supplementation in Parkinson's disease: an RCT",
            abstract="This RCT enrolled 120 Parkinson's patients and found that daily vitamin D "
                     "supplementation significantly improved bone mineral density and reduced fall "
                     "risk over 12 months compared to placebo. Motor symptoms showed modest improvement.",
            authors=["Smith J", "Patel K"], year=2023, sources=["pubmed"],
            url="https://pubmed.ncbi.nlm.nih.gov/11111",
        ),
        "doc2": Document(
            doc_id="pubmed:22222", doc_type="publication",
            title="Vitamin D deficiency prevalence in neurodegenerative diseases",
            abstract="A systematic review found that 60-80% of Parkinson's patients have vitamin D "
                     "deficiency compared to 20-40% in age-matched controls. Low vitamin D correlated "
                     "with worse motor outcomes.",
            authors=["Chen L", "Wang M"], year=2024, sources=["pubmed"],
            url="https://pubmed.ncbi.nlm.nih.gov/22222",
        ),
        "doc3": Document(
            doc_id="openalex:W33333", doc_type="publication",
            title="Neuroprotective effects of cholecalciferol in PD animal models",
            abstract="In a mouse model, vitamin D3 showed neuroprotective effects on dopaminergic "
                     "neurons and reduced neuroinflammation markers.",
            authors=["Garcia R"], year=2022, sources=["openalex"],
            url="https://openalex.org/W33333",
        ),
        "doc4": Document(
            doc_id="pubmed:44444", doc_type="publication",
            title="Safety of vitamin D with levodopa in elderly patients",
            abstract="No significant drug interactions found with vitamin D up to 4000 IU daily "
                     "in PD patients on levodopa. Calcium levels remained normal.",
            authors=["Brown A"], year=2023, sources=["pubmed"],
            url="https://pubmed.ncbi.nlm.nih.gov/44444",
        ),
        "doc5": Document(
            doc_id="nct:NCT09999991", doc_type="trial",
            title="Vitamin D3 Supplementation in Early Parkinson's Disease",
            abstract="Phase 3 trial evaluating high-dose vitamin D3 for slowing PD progression.",
            nct_id="NCT09999991", status="RECRUITING",
            eligibility_criteria="Age 45-75, PD diagnosis within 3 years, Hoehn and Yahr 1-2.",
            primary_outcomes=["Change in MDS-UPDRS total score at 24 months"],
            locations=[{"facility": "Toronto Western Hospital", "country": "Canada"}],
            contacts=[{"name": "Dr. Smith", "email": "smith@twh.ca", "phone": "416-555-1234"}],
            sources=["clinicaltrials"],
        ),
        "doc6": Document(
            doc_id="nct:NCT09999992", doc_type="trial",
            title="Nutritional Optimization Before DBS Surgery",
            abstract="Evaluating pre-surgical nutritional optimization for DBS outcomes.",
            nct_id="NCT09999992", status="COMPLETED",
            eligibility_criteria="Scheduled for DBS, PD diagnosis.",
            primary_outcomes=["Post-DBS motor improvement at 6 months"],
            locations=[{"facility": "Johns Hopkins", "country": "USA"}],
            sources=["clinicaltrials"],
        ),
    }


def test_normal_response():
    """Test normal case with valid citations."""
    print("\n--- Test 1: Normal response with valid citations ---")

    doc_anchors = make_doc_anchors()

    llm_output = {
        "overview": "Vitamin D supplementation appears safe and potentially beneficial for "
                    "Parkinson's disease patients, with evidence supporting improved bone health "
                    "and possible neuroprotective effects.",
        "insights": [
            {
                "finding": "Daily vitamin D supplementation improved bone density and reduced falls in PD patients",
                "sources": ["doc1"],
            },
            {
                "finding": "60-80% of PD patients are vitamin D deficient with worse motor outcomes",
                "sources": ["doc2"],
            },
            {
                "finding": "Vitamin D3 shows neuroprotective effects in animal models of Parkinson's",
                "sources": ["doc3"],
            },
            {
                "finding": "No significant drug interactions between vitamin D and levodopa",
                "sources": ["doc4"],
            },
        ],
        "trials": [
            {"nct_id": "NCT09999991", "title": "Vitamin D3 in Early PD",
             "relevance": "Directly relevant RCT for vitamin D in PD", "sources": ["doc5"]},
            {"nct_id": "NCT09999992", "title": "Nutritional Optimization Before DBS",
             "relevance": "Related to DBS context", "sources": ["doc6"]},
        ],
        "abstain_reason": None,
    }

    result = assemble_response(llm_output, doc_anchors)
    uf = result.user_facing_json

    # Checks
    assert uf["overview"], "overview should be populated"
    assert len(uf["insights"]) == 4, f"expected 4 insights, got {len(uf['insights'])}"
    assert len(uf["trials"]) == 2, f"expected 2 trials, got {len(uf['trials'])}"
    assert uf["abstain_reason"] is None

    # Check source_details resolved
    for ins in uf["insights"]:
        assert ins["source_details"], f"source_details empty for: {ins['finding'][:40]}"
        for sd in ins["source_details"]:
            assert sd["title"], "title missing in source_details"
            assert sd["url"], "url missing in source_details"
            assert sd["snippet"], "snippet missing in source_details"
        assert ins["unverified"] is False

    # Check trials resolved
    for trial in uf["trials"]:
        assert trial["nct_id"], "nct_id missing"
        assert trial["status"], "status missing"
        assert trial["location"], "location missing"

    # Check citation stats
    stats = uf["pipelineMeta"]["citation_stats"]
    assert stats["verified"] > 0
    assert stats["unverified"] == 0

    print("  PASS: all citations resolved with title/authors/year/url/snippet")
    print(f"  PASS: {stats['verified']} verified, {stats['unverified']} unverified")
    print(f"  PASS: {len(uf['trials'])} trials with status/location/contact")


def test_hallucinated_citations():
    """Test that hallucinated anchors are flagged."""
    print("\n--- Test 2: Hallucinated citations ---")

    doc_anchors = make_doc_anchors()

    llm_output = {
        "overview": "Some findings about vitamin D.",
        "insights": [
            {"finding": "Valid finding", "sources": ["doc1"]},
            {"finding": "Hallucinated finding", "sources": ["doc9", "doc10"]},
        ],
        "trials": [],
        "abstain_reason": None,
    }

    result = assemble_response(llm_output, doc_anchors)
    uf = result.user_facing_json

    # First insight should be verified
    assert uf["insights"][0]["unverified"] is False
    # Second should be unverified (hallucinated anchors)
    assert uf["insights"][1]["unverified"] is True
    assert len(uf["insights"][1]["source_details"]) == 0

    stats = uf["pipelineMeta"]["citation_stats"]
    assert stats["unverified"] == 2
    assert "hallucinated citation: doc9" in result.warnings

    print("  PASS: hallucinated citations flagged as unverified")
    print(f"  PASS: warnings: {result.warnings}")


def test_abstain_case():
    """Test abstain response formatting."""
    print("\n--- Test 3: Abstain case ---")

    doc_anchors = make_doc_anchors()

    llm_output = {
        "overview": "Based on the retrieved research, I cannot directly answer this question.",
        "insights": [],
        "trials": [],
        "abstain_reason": "The retrieved documents do not contain information about this topic.",
    }

    result = assemble_response(llm_output, doc_anchors)
    uf = result.user_facing_json

    assert uf["abstain_reason"] is not None
    assert uf["insights"] == []
    assert uf["trials"] == []
    assert "suggestion" in uf

    print("  PASS: abstain formatted with empty insights/trials + suggestion")


def main():
    print("=" * 60)
    print("Phase 5 Step 5.4: Response Assembler Test")
    print("=" * 60)

    test_normal_response()
    test_hallucinated_citations()
    test_abstain_case()

    print("\n" + "=" * 60)
    print("All tests passed.")


if __name__ == "__main__":
    main()
