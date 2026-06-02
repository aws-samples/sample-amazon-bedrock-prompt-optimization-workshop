"""
Evaluation helper functions for comparing agent versions.
"""

from __future__ import annotations

import json

import pandas as pd

from utils.runtime_helpers import invoke_agent_with_timing


def load_test_scenarios(filepath: str = "data/test_scenarios.json") -> list:
    """Load test scenarios from JSON file."""
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def run_evaluation_suite(data_client, agent_arn: str, scenarios: list, version_name: str = "unknown") -> dict:
    """Run full evaluation suite against an agent."""
    print(f"\nEvaluating {version_name}...")
    results = []
    total_latency = 0
    successful = 0
    failed = 0

    for scenario in scenarios:
        print(f"  [{scenario['id']}] {scenario['query'][:50]}...")
        try:
            response, latency_ms = invoke_agent_with_timing(data_client, agent_arn, scenario["query"])
            results.append(
                {
                    "scenario_id": scenario["id"],
                    "query": scenario["query"],
                    "response": response,
                    "latency_ms": latency_ms,
                    "success": True,
                }
            )
            total_latency += latency_ms
            successful += 1
        except Exception as e:
            results.append(
                {
                    "scenario_id": scenario["id"],
                    "query": scenario["query"],
                    "error": str(e),
                    "success": False,
                }
            )
            failed += 1

    return {
        "version": version_name,
        "agent_arn": agent_arn,
        "results": results,
        "summary": {
            "total_scenarios": len(scenarios),
            "successful": successful,
            "failed": failed,
            "avg_latency_ms": total_latency / successful if successful > 0 else 0,
        },
    }


def compare_versions(all_results: dict) -> pd.DataFrame:
    """Generate comparison DataFrame across all versions."""
    rows = []
    for version_name, eval_result in all_results.items():
        summary = eval_result["summary"]
        success_rate = (summary["successful"] / summary["total_scenarios"]) * 100
        rows.append(
            {
                "Version": version_name,
                "Success Rate": f"{success_rate:.1f}%",
                "Avg Latency (ms)": f"{summary['avg_latency_ms']:.0f}",
                "Successful": summary["successful"],
                "Failed": summary["failed"],
            }
        )
    return pd.DataFrame(rows)


def print_evaluation_summary(eval_results: dict):
    """Print formatted evaluation summary."""
    print("\n" + "=" * 60)
    print(f"EVALUATION: {eval_results['version']}")
    print("=" * 60)

    summary = eval_results["summary"]
    success_rate = (summary["successful"] / summary["total_scenarios"]) * 100

    print(f"Scenarios: {summary['total_scenarios']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Avg Latency: {summary['avg_latency_ms']:.0f}ms")

    print("\nResults:")
    for result in eval_results["results"]:
        status = "PASS" if result["success"] else "FAIL"
        latency = result.get("latency_ms", 0)
        print(f"  [{status}] {result['scenario_id']}: {latency:.0f}ms")

    print("=" * 60)
