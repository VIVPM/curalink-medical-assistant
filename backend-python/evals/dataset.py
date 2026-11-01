"""Golden inputs and reference rubrics, without importing the application or judge."""

import json
from pathlib import Path


GOLDEN = Path(__file__).resolve().parents[2] / "evals" / "golden" / "v1"


def load_dataset(root=GOLDEN):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    evidence = json.loads((root / "evidence.json").read_text(encoding="utf-8"))["sources"]
    standalone = json.loads((root / "standalone.json").read_text(encoding="utf-8"))
    threads = json.loads((root / "conversations.json").read_text(encoding="utf-8"))
    cases = []
    for group in standalone["groups"]:
        for case in group["cases"]:
            cases.append({**standalone["defaults"],
                          "intake": group["intake"], **case})
    cases.extend(standalone["behavioral_cases"])
    cases.extend(standalone.get("safety_cases", []))
    for thread in threads["conversations"]:
        history = []
        supplied = {}
        for turn in thread["turns"]:
            case = {**threads["defaults"],
                    "intake": thread["intake"], **turn, "thread_id": thread["id"],
                    "history": list(history)}
            cases.append(case)
            content = reference_answer(case, evidence)
            cited = set()
            for ref in turn.get("history_evidence_after", []):
                source = ref.split(".", 1)[0]
                cited.add(source)
                supplied.setdefault(source, {})[ref] = resolve_fact(ref, evidence)
            insights = []
            for source in sorted(cited):
                record = evidence[source]
                snippet = " ".join(supplied[source].values())
                if len(snippet) > 800:
                    raise ValueError(f"{turn['id']}: canonical source exceeds 800 characters")
                insights.append({"finding": snippet, "source_details": [{
                    "title": record["title"], "url": record["url"], "year": record["year"],
                    "platform": record.get("platform", "PubMed"), "snippet": snippet, "authors": [],
                }]})
            history.extend([{"role": "user", "content": turn["question"]}, {
                "role": "assistant", "content": content,
                "structuredResponse": {"insights": insights, "trials": [], "abstain_reason": None},
            }])
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate golden case IDs")
    for case in cases:
        reference_answer(case, evidence)
    return manifest, evidence, cases


def resolve_fact(ref, evidence):
    source, fact = ref.split(".", 1)
    return evidence[source]["facts"][fact]


def reference_answer(case, evidence):
    parts = [resolve_fact(ref, evidence) for ref in case.get("reference_facts", [])]
    if case.get("reference_answer"):
        parts.append(case["reference_answer"])
    if not parts:
        raise ValueError(f"{case['id']}: missing reference answer")
    return "\n".join(parts)


def rubric(case, evidence, manifest):
    parts = [reference_answer(case, evidence)]
    boundaries = list(manifest["global_boundaries"])
    for source in sorted({ref.split(".", 1)[0] for ref in case.get("reference_facts", [])}):
        boundaries.extend(evidence[source].get("boundaries", []))
    parts.append("Safety and evidence requirements: " + "; ".join(boundaries))
    if case.get("forbidden_claims"):
        parts.append("Forbidden claims: " + "; ".join(case["forbidden_claims"]))
    if case.get("checks"):
        parts.append("Additional requirements: " + "; ".join(case["checks"]))
    return "\n".join(parts)


def question_input(case):
    intake = case.get("intake_override", case.get("intake", {}))
    static = {key: intake.get(key, "") for key in ("disease", "intent", "location")}
    return static, case.get("history", [])


def judge_input(case):
    static, history = question_input(case)
    return json.dumps({"intake_context_not_evidence": static,
                       "previous_conversation": history[-6:],
                       "current_question": case["question"]}, ensure_ascii=False)
