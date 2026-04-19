"""
Standalone test for Phase 5 Step 5.2 — Context Builder (Stage 5).
Run from the backend-python directory:
    python scripts/phase5/test_context_builder.py

Feeds 8 mock docs + user query into the context builder and verifies:
  - Token count under 7400
  - Every doc has a [docN] anchor
  - Grounding rules present in system prompt
  - doc_anchors dict populated correctly
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from schemas.document import Document
from stages.context_builder import build_context, GROUNDING_RULES


def make_mock_docs() -> list[Document]:
    """Create 8 mock docs (6 publications + 2 trials) for testing."""
    docs = []
    for i in range(6):
        docs.append(Document(
            doc_id=f"pubmed:{10000 + i}",
            doc_type="publication",
            title=f"Study {i+1}: Effects of treatment X on Parkinson's disease",
            abstract=f"This study investigated the effects of treatment approach {i+1} "
                     f"on patients with Parkinson's disease. Results showed significant "
                     f"improvement in motor function scores after 12 weeks of treatment. "
                     f"A total of {50 + i*10} patients were enrolled in this randomized "
                     f"controlled trial conducted across multiple centers." * 2,
            authors=[f"Author{j} A" for j in range(3)],
            year=2020 + i,
            journal=f"Journal of Neurology Vol.{i+1}",
            doi=f"10.1000/test.{10000+i}",
            url=f"https://pubmed.ncbi.nlm.nih.gov/{10000+i}",
            sources=["pubmed"],
            disease_tags=["parkinson"],
        ))

    # 2 trials
    for i in range(2):
        docs.append(Document(
            doc_id=f"nct:NCT0000000{i}",
            doc_type="trial",
            title=f"Trial {i+1}: Vitamin D Supplementation in Parkinson's",
            abstract=f"This trial evaluates the efficacy of vitamin D supplementation "
                     f"in patients with Parkinson's disease over a 6-month period.",
            nct_id=f"NCT0000000{i}",
            status="RECRUITING" if i == 0 else "COMPLETED",
            eligibility_criteria="Inclusion: Age 40-80, diagnosed with PD, "
                                 "Hoehn and Yahr stage 1-3. "
                                 "Exclusion: Severe cognitive impairment, "
                                 "current vitamin D supplementation.",
            primary_outcomes=["Change in UPDRS motor score from baseline"],
            locations=[{"facility": "Toronto General Hospital", "country": "Canada"}],
            sources=["clinicaltrials"],
            disease_tags=["parkinson"],
        ))

    return docs


def main():
    print("=" * 60)
    print("Phase 5 Step 5.2: Context Builder Test")
    print("=" * 60)

    docs = make_mock_docs()
    user_message = "Can I take Vitamin D?"
    static_context = {
        "disease": "Parkinson's disease",
        "intent": "Deep Brain Stimulation",
        "location": "Toronto, Canada",
        "patientName": "John Smith",
    }
    chat_history = [
        {"role": "user", "content": "Latest treatment options for DBS"},
        {"role": "assistant", "content": "Based on recent research, deep brain stimulation has shown significant improvements in motor symptoms for Parkinson's disease patients..."},
    ]

    print(f"\n[1/5] Building context with {len(docs)} docs")
    payload = build_context(
        top_docs=docs,
        user_message=user_message,
        static_context=static_context,
        chat_history=chat_history,
    )

    # 1. Token count check
    print(f"\n[2/5] Token count: {payload.token_count}")
    if payload.token_count < 7400:
        print("  PASS: under 7400 token budget")
    else:
        print(f"  WARN: {payload.token_count} tokens exceeds 7400 budget")

    # 2. Doc anchors check
    print(f"\n[3/5] Doc anchors: {list(payload.doc_anchors.keys())}")
    if len(payload.doc_anchors) == len(docs):
        print(f"  PASS: {len(payload.doc_anchors)} anchors for {len(docs)} docs")
    else:
        print(f"  FAIL: {len(payload.doc_anchors)} anchors for {len(docs)} docs")

    # Verify each anchor appears in user prompt
    all_anchors_present = True
    for anchor in payload.doc_anchors:
        if f"[{anchor}]" not in payload.user_prompt:
            print(f"  FAIL: [{anchor}] not found in user prompt")
            all_anchors_present = False
    if all_anchors_present:
        print("  PASS: all [docN] anchors present in user prompt")

    # 3. Grounding rules in system prompt
    print(f"\n[4/5] Checking system prompt")
    if "ONLY using the documents" in payload.system_prompt:
        print("  PASS: grounding rules present")
    else:
        print("  FAIL: grounding rules missing")

    if '"overview"' in payload.system_prompt and '"insights"' in payload.system_prompt:
        print("  PASS: JSON schema present in system prompt")
    else:
        print("  FAIL: JSON schema missing from system prompt")

    if "abstain_reason" in payload.system_prompt:
        print("  PASS: abstain instruction present")
    else:
        print("  FAIL: abstain instruction missing")

    # 4. Content checks
    print(f"\n[5/5] Content verification")
    if "Parkinson" in payload.user_prompt:
        print("  PASS: disease context in prompt")
    else:
        print("  FAIL: disease context missing")

    if "PREVIOUS CONVERSATION" in payload.user_prompt:
        print("  PASS: chat history included")
    else:
        print("  FAIL: chat history missing")

    if "Can I take Vitamin D?" in payload.user_prompt:
        print("  PASS: user message included")
    else:
        print("  FAIL: user message missing")

    if "NCT" in payload.user_prompt:
        print("  PASS: trial NCT IDs present")
    else:
        print("  FAIL: trial NCT IDs missing")

    if "RECRUITING" in payload.user_prompt:
        print("  PASS: trial status present")
    else:
        print("  FAIL: trial status missing")

    # Truncations
    if payload.truncations:
        print(f"\n  Truncations: {payload.truncations}")
    else:
        print("\n  No truncations needed")

    # Print a snippet of the prompt
    print(f"\n--- System prompt (first 200 chars) ---")
    print(payload.system_prompt[:200])
    print(f"\n--- User prompt (first 500 chars) ---")
    print(payload.user_prompt[:500])

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
