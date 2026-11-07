"""Offline, read-only audit of frozen route labels against saved pipeline captures."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from .dataset import GOLDEN, load_dataset


LABELS = GOLDEN / "routing_labels.json"
ALIASES = {"rag": "full RAG", "conversation": "history-only",
           "emergency": "urgent", "emergency_intercept": "urgent",
           "clarification": "clarify", "out_of_scope": "scope refusal",
           "smalltalk": "smalltalk"}


def normalize_route(route):
    return ALIASES.get(route, route or "missing")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def confusion(rows):
    matrix = defaultdict(Counter)
    for row in rows:
        if row["status"] == "success":
            matrix[row["expected_route"]][row["actual_route"]] += 1
    return {expected: dict(actual) for expected, actual in sorted(matrix.items())}


def summarize(labels, captures):
    rows = []
    for label in labels:
        case_id = label["case_id"]
        capture = captures.get(case_id, {})
        actual = capture.get("routing") or {}
        status = capture.get("status", "missing")
        row = {**label, "status": status,
               "actual_route": normalize_route(actual.get("route")),
               "actual_tier": actual.get("model_tier") if actual.get("route") in {"rag", "conversation"} else None,
               "answer_model": actual.get("answer_model"),
               "retrieval_used": actual.get("retrieval_used"),
               "fallback": actual.get("fallback"),
               "route_match": status == "success" and normalize_route(actual.get("route")) == label["expected_route"],
               "tier_match": None if not label.get("expected_tier") or status != "success" else actual.get("model_tier") == label["expected_tier"]}
        rows.append(row)
    scored = [r for r in rows if r["status"] == "success" and not r.get("ambiguous")]
    tier_scored = [r for r in scored if r["expected_tier"] and r["actual_tier"]]
    urgent = [r for r in scored if r["expected_route"] == "urgent"]
    report = {
        "labels": len(labels), "captures": len(captures),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "expected_route_counts": dict(Counter(r["expected_route"] for r in rows)),
        "actual_route_counts": dict(Counter(r["actual_route"] for r in rows)),
        "ambiguous_cases": [{"case_id": r["case_id"], "reason": r.get("ambiguity_reason")} for r in rows if r.get("ambiguous")],
        "confusion_all_successes": confusion(rows),
        "route_scored_denominator": len(scored),
        "route_correct": sum(r["route_match"] for r in scored),
        "tier_scored_denominator": len(tier_scored),
        "tier_correct": sum(r["tier_match"] for r in tier_scored),
        "urgent_denominator": len(urgent),
        "urgent_false_negatives": [r["case_id"] for r in urgent if r["actual_route"] != "urgent"],
        "unsupported_history_only": [r["case_id"] for r in scored if r["actual_route"] == "history-only" and r["expected_route"] in {"full RAG", "urgent", "clarify"}],
        "route_mismatches": [r["case_id"] for r in scored if not r["route_match"]],
        "tier_mismatches": [r["case_id"] for r in tier_scored if not r["tier_match"]],
    }
    return rows, report


def project_saved_routes(labels, captures, cases):
    """Reclassify saved inputs with current guards; never infer a new answer."""
    from routing import _reports_new_patient_fact, immediate_response

    rows = []
    for label in labels:
        case_id = label["case_id"]
        case = cases[case_id]
        capture = captures[case_id]
        status = capture.get("status", "missing")
        before = normalize_route((capture.get("routing") or {}).get("route"))
        projected = before
        reason = None
        if status == "success":
            direct = immediate_response(case["question"], case.get("history", []))
            if direct is not None:
                projected = normalize_route(direct[0])
                reason = "current_pre_provider_guard" if projected != before else None
            elif before == "history-only" and _reports_new_patient_fact(case["question"]):
                projected = "full RAG"
                reason = "new_patient_fact" if projected != before else None
        rows.append({
            "case_id": case_id, "status": status, "expected_route": label["expected_route"],
            "ambiguous": bool(label.get("ambiguous")), "before_route": before,
            "code_projected_route": projected, "change_reason": reason,
            "new_answer_captured": False,
        })

    scored = [r for r in rows if r["status"] == "success" and not r["ambiguous"]]
    urgent = [r for r in scored if r["expected_route"] == "urgent"]
    report = {
        "scope": "code-level reclassification of saved inputs; no new model answer",
        "total_cases": len(rows), "route_scored_denominator": len(scored),
        "before_route_correct": sum(r["before_route"] == r["expected_route"] for r in scored),
        "projected_route_correct": sum(r["code_projected_route"] == r["expected_route"] for r in scored),
        "urgent_denominator": len(urgent),
        "before_urgent_false_negatives": [r["case_id"] for r in urgent if r["before_route"] != "urgent"],
        "projected_urgent_false_negatives": [r["case_id"] for r in urgent if r["code_projected_route"] != "urgent"],
        "before_unsupported_history_only": [r["case_id"] for r in scored if r["before_route"] == "history-only" and r["expected_route"] in {"full RAG", "urgent", "clarify"}],
        "projected_unsupported_history_only": [r["case_id"] for r in scored if r["code_projected_route"] == "history-only" and r["expected_route"] in {"full RAG", "urgent", "clarify"}],
        "before_route_counts": dict(Counter(r["before_route"] for r in rows if r["status"] == "success")),
        "projected_route_counts": dict(Counter(r["code_projected_route"] for r in rows if r["status"] == "success")),
        "changed_case_ids": sorted(r["case_id"] for r in rows if r["change_reason"]),
        "remaining_route_mismatches": [r["case_id"] for r in scored if r["code_projected_route"] != r["expected_route"]],
    }
    return rows, report


def load_captures(paths):
    captures = {}
    duplicates = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                case_id = row["case_id"]
                if case_id in captures:
                    duplicates.append(case_id)
                captures[case_id] = row
    if duplicates:
        raise ValueError(f"Duplicate capture IDs: {duplicates}")
    return captures


def audit(labels_path, capture_paths, output_dir):
    frozen = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    for relative, digest in frozen["input_sha256"].items():
        if sha256(GOLDEN / relative) != digest:
            raise ValueError(f"Frozen routing input changed: {relative}")
    _, _, cases = load_dataset()
    by_id = {case["id"]: case for case in cases}
    labels = frozen["cases"]
    if len(labels) != 150 or {x["case_id"] for x in labels} != set(by_id):
        raise ValueError("Frozen labels do not cover the exact golden dataset")
    for label in labels:
        case = by_id[label["case_id"]]
        if label["question_sha256"] != hashlib.sha256(case["question"].encode()).hexdigest():
            raise ValueError(f"Question changed: {label['case_id']}")
    captures = load_captures(capture_paths)
    if set(captures) != set(by_id):
        raise ValueError("Capture IDs do not cover the exact golden dataset")
    rows, report = summarize(labels, captures)
    report["provenance"] = {"dataset_id": frozen["dataset_id"],
                            "label_file_sha256": sha256(labels_path),
                            "golden_input_sha256": frozen["input_sha256"],
                            "rule_source_sha256_at_labeling": frozen["rule_source_sha256"],
                            "capture_files_sha256": {str(path): sha256(path) for path in capture_paths},
                            "capture_configuration_counts": dict(Counter(json.dumps(row.get("configuration", {}), sort_keys=True) for row in captures.values())),
                            "scope": "saved Python in-process decisions; Node interception not exercised",
                            "quality": "not rejudged; existing LLM-judged results remain valid separately"}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "decisions.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def code_recheck(labels_path, capture_paths, output_dir):
    """Save the original capture audit and a clearly marked code-only projection."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    audit(labels_path, capture_paths, output_dir / "saved_capture_baseline")
    labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))["cases"]
    cases = {case["id"]: case for case in load_dataset()[2]}
    captures = load_captures(capture_paths)
    rows, report = project_saved_routes(labels, captures, cases)
    report["provenance"] = {
        "labels_sha256": sha256(labels_path),
        "capture_files_sha256": {str(path): sha256(path) for path in capture_paths},
        "python_routing_source_sha256": sha256(Path(__file__).resolve().parents[1] / "routing.py"),
        "note": "The projected routes do not include fresh Jev decisions, retrieval, generated answers or new latency measurements.",
    }
    with (output_dir / "code_projected_decisions.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "code_projected_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--captures", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--code-recheck", action="store_true",
                        help="Project current guard changes over saved inputs without generating new answers")
    args = parser.parse_args()
    run = code_recheck if args.code_recheck else audit
    print(json.dumps(run(args.labels, args.captures, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
