"""Pair successful saved HF captures; never invokes a provider or judge."""

import argparse
import json
import statistics
from pathlib import Path

from .dataset import load_dataset

ASSUMED_RATES_PER_MILLION = {
    "meta-llama/Llama-3.1-8B-Instruct": (0.02, 0.05),
    "meta-llama/Llama-3.3-70B-Instruct": (0.15, 0.30),
}


def generation_cost_estimate(patterns):
    """Historical assumed rates; 4000-token calls are answer generation."""
    folders = {path.parent for pattern in patterns for path in Path().glob(pattern)}
    cost = 0.0
    calls = 0
    for folder in folders:
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        for usage in summary["usage"]:
            if usage.get("max_tokens") != 4000:
                continue
            input_rate, output_rate = ASSUMED_RATES_PER_MILLION[usage["model"]]
            if usage.get("input_tokens") is None or usage.get("output_tokens") is None:
                raise ValueError(f"Missing provider usage in {folder}")
            cost += (usage["input_tokens"] * input_rate + usage["output_tokens"] * output_rate) / 1_000_000
            calls += 1
    return {"estimated_usd": cost, "answer_calls_with_usage": calls,
            "rate_assumption_usd_per_million_input_output": ASSUMED_RATES_PER_MILLION,
            "scope": "answer generation across all original and retry attempts, including failed attempts that reached a model; excludes expansion, Jev, retrieval, judging and hosting",
            "billing_status": "not billed; historical illustrative rate assumption, provider auto selection not confirmed"}


def percentile(values, percent):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * percent / 100
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def load_arm(patterns):
    attempts = {}
    for pattern in patterns:
        for path in sorted(Path().glob(pattern)):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line:
                    row = json.loads(line)
                    attempts.setdefault(row["case_id"], []).append(row)
    selected = {case_id: next((row for row in rows if row["status"] == "success"), None)
                for case_id, rows in attempts.items()}
    return selected, attempts


def compare(routed_patterns, large_patterns, output_dir):
    _, _, cases = load_dataset()
    ids = [case["id"] for case in cases]
    routed, routed_attempts = load_arm(routed_patterns)
    large, large_attempts = load_arm(large_patterns)
    if set(routed) != set(ids) or set(large) != set(ids):
        raise ValueError("Both arms must attempt all 150 golden cases")
    paired = [case_id for case_id in ids if routed[case_id] and large[case_id]]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    result = {"dataset_id": "curalink-patient-golden-v2", "paired_successes": len(paired),
              "routed_attempts": sum(map(len, routed_attempts.values())),
              "large_attempts": sum(map(len, large_attempts.values())),
              "failed_cases_after_retries": [case_id for case_id in ids if case_id not in paired],
              "answer_quality": "unjudged in these HF captures",
              "generation_cost": {"routed": generation_cost_estimate(routed_patterns),
                                  "always_large": generation_cost_estimate(large_patterns)},
              "retrieval_consistency": "retries may use changed selection policy; compare selection_policy per case",
              "arms": {}}
    for name, arm in (("routed", routed), ("always_large", large)):
        with (output_dir / f"{name}-success.jsonl").open("w", encoding="utf-8") as stream:
            for case_id in paired:
                stream.write(json.dumps(arm[case_id], ensure_ascii=False) + "\n")
        latencies = [arm[case_id]["latency_seconds"] for case_id in paired]
        result["arms"][name] = {
            "n": len(latencies), "latency_p50_seconds": percentile(latencies, 50),
            "latency_p95_seconds": percentile(latencies, 95),
            "latency_p99_seconds": percentile(latencies, 99),
            "latency_mean_seconds": statistics.mean(latencies),
            "model_counts": {model: sum(arm[cid].get("routing", {}).get("answer_model") == model for cid in paired)
                             for model in sorted({arm[cid].get("routing", {}).get("answer_model") for cid in paired}, key=str)},
            "configuration_counts": {config: sum(json.dumps(arm[cid].get("configuration", {}), sort_keys=True) == config for cid in paired)
                                     for config in sorted({json.dumps(arm[cid].get("configuration", {}), sort_keys=True) for cid in paired})},
        }
    changed = [case_id for case_id in paired if routed[case_id].get("selection_policy") != large[case_id].get("selection_policy")]
    result["selection_policy_mismatch_count"] = len(changed)
    result["selection_policy_mismatch_ids"] = changed
    with (output_dir / "paired.jsonl").open("w", encoding="utf-8") as stream:
        for case_id in paired:
            a, b = routed[case_id], large[case_id]
            stream.write(json.dumps({"case_id": case_id,
                "routed_route": a.get("routing", {}).get("route"),
                "routed_model": a.get("routing", {}).get("answer_model"),
                "large_model": b.get("routing", {}).get("answer_model"),
                "routed_latency_seconds": a["latency_seconds"],
                "large_latency_seconds": b["latency_seconds"],
                "latency_delta_seconds": a["latency_seconds"] - b["latency_seconds"],
                "same_selection_policy": a.get("selection_policy") == b.get("selection_policy"),
                "routed_attempts": len(routed_attempts[case_id]),
                "large_attempts": len(large_attempts[case_id]),
            }) + "\n")
    (output_dir / "comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routed", nargs="+", required=True, help="Glob patterns in chronological run order")
    parser.add_argument("--large", nargs="+", required=True, help="Glob patterns in chronological run order")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.routed, args.large, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
