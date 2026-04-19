"""
Standalone test for Phase 5 Step 5.1 — Query Expander (Stage 1).
Run from the backend-python directory:
    python scripts/phase5/test_query_expander.py

Runs the query expander on 5 different user messages with various static
contexts. Verifies disease_focus is injected, intent is valid, and
expanded_queries are populated.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio
from dotenv import load_dotenv

from llm_backend import get_llm_backend
from stages.query_expander import expand_query

load_dotenv()


TEST_CASES = [
    {
        "name": "Basic treatment query",
        "user_message": "Latest treatment options",
        "static_context": {
            "disease": "Parkinson's disease",
            "intent": "Deep Brain Stimulation",
            "location": "Toronto, Canada",
            "patientName": "John Smith",
        },
        "chat_history": None,
    },
    {
        "name": "Follow-up with context (Vitamin D test)",
        "user_message": "Can I take Vitamin D?",
        "static_context": {
            "disease": "Parkinson's disease",
            "intent": "Deep Brain Stimulation",
            "location": "Toronto, Canada",
            "patientName": "John Smith",
        },
        "chat_history": [
            {"role": "user", "content": "Latest treatment options for DBS"},
            {"role": "assistant", "content": "Based on recent research, deep brain stimulation..."},
        ],
    },
    {
        "name": "Clinical trials search",
        "user_message": "Clinical trials for diabetes near me",
        "static_context": {
            "disease": "Type 2 Diabetes",
            "intent": "",
            "location": "New York, USA",
            "patientName": "Jane Doe",
        },
        "chat_history": None,
    },
    {
        "name": "Greeting (should skip retrieval)",
        "user_message": "Hi, thanks for helping!",
        "static_context": {
            "disease": "Lung Cancer",
            "intent": "",
            "location": "",
            "patientName": "Test",
        },
        "chat_history": None,
    },
    {
        "name": "Abbreviation expansion",
        "user_message": "What about MI risk with this treatment?",
        "static_context": {
            "disease": "Alzheimer's disease",
            "intent": "immunotherapy",
            "location": "",
            "patientName": "Bob",
        },
        "chat_history": [
            {"role": "user", "content": "Tell me about immunotherapy options"},
            {"role": "assistant", "content": "Current immunotherapy approaches for Alzheimer's include..."},
        ],
    },
]


async def run_tests():
    llm = get_llm_backend()

    print("=" * 60)
    print("Phase 5 Step 5.1: Query Expander Test")
    print("=" * 60)

    all_pass = True

    for i, tc in enumerate(TEST_CASES):
        print(f"\n--- Test {i+1}/{len(TEST_CASES)}: {tc['name']} ---")
        print(f"  User message: \"{tc['user_message']}\"")
        print(f"  Disease: {tc['static_context']['disease']}")

        result = await expand_query(
            user_message=tc["user_message"],
            static_context=tc["static_context"],
            chat_history=tc.get("chat_history"),
            llm=llm,
        )

        print(f"  Timing: {result.timing_ms}ms")
        print(f"  Intent: {result.intent}")
        print(f"  Disease focus: {result.disease_focus}")
        print(f"  Skip retrieval: {result.skip_retrieval}")
        print(f"  Entities: {result.entities}")
        print(f"  Keywords: {result.keywords}")
        print(f"  Expanded queries ({len(result.expanded_queries)}):")
        for q in result.expanded_queries:
            print(f"    - {q}")

        # Checks
        disease = tc["static_context"]["disease"].lower()
        checks_passed = True

        # 1. expanded_queries not empty
        if not result.expanded_queries:
            print("  FAIL: expanded_queries is empty")
            checks_passed = False

        # 2. disease present in every expanded query
        if not result.skip_retrieval:
            for q in result.expanded_queries:
                if disease not in q.lower() and disease.split()[0].lower() not in q.lower():
                    print(f"  WARN: disease not found in query: \"{q}\"")

        # 3. intent is valid
        valid_intents = {
            "treatment_overview", "drug_interaction_safety",
            "clinical_trials_search", "side_effects", "prognosis",
            "diagnosis_criteria", "general_info",
        }
        if result.intent not in valid_intents:
            print(f"  FAIL: invalid intent: {result.intent}")
            checks_passed = False

        # 4. disease_focus populated
        if not result.disease_focus:
            print("  FAIL: disease_focus is empty")
            checks_passed = False

        # 5. at most 4 queries
        if len(result.expanded_queries) > 4:
            print(f"  FAIL: too many queries ({len(result.expanded_queries)} > 4)")
            checks_passed = False

        # 6. greeting test: skip_retrieval should be true
        if tc["name"] == "Greeting (should skip retrieval)":
            if result.skip_retrieval:
                print("  PASS: correctly set skip_retrieval=true for greeting")
            else:
                print("  WARN: expected skip_retrieval=true for greeting")

        if checks_passed:
            print("  PASS")
        else:
            all_pass = False

    print("\n" + "=" * 60)
    if all_pass:
        print("All tests passed.")
    else:
        print("Some tests had failures — check output above.")


if __name__ == "__main__":
    asyncio.run(run_tests())
