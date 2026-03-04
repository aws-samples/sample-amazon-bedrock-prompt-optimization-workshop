"""
Quality gate checker for CI/CD pipelines.

Reads evaluation results from a JSON file (produced by evaluate_prompt.py)
and exits with non-zero status if quality thresholds are not met.

This script is intentionally lightweight so it can run as a separate
pipeline step without Langfuse or Bedrock dependencies.

Usage:
    python scripts/check_quality_gate.py [results_file]

Arguments:
    results_file  Path to evaluation_results.json (default: ./evaluation_results.json)

Environment variables (optional overrides):
    EVAL_SCORE_THRESHOLD     - Minimum average score (default: 0.7)
    EVAL_PASS_RATE_THRESHOLD - Minimum pass rate (default: 0.8)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def check_quality_gate(results_path: str | Path) -> bool:
    """
    Check whether evaluation results meet quality thresholds.

    Args:
        results_path: Path to the evaluation results JSON file.

    Returns:
        True if the quality gate passes, False otherwise.
    """
    results_path = Path(results_path)
    if not results_path.exists():
        print(f"[ERROR] Results file not found: {results_path}")
        return False

    with open(results_path, encoding="utf-8") as fh:
        results = json.load(fh)

    score_threshold = float(
        results.get("score_threshold", 0)
        or float(__import__("os").environ.get("EVAL_SCORE_THRESHOLD", "0.7"))
    )
    pass_rate_threshold = float(
        results.get("pass_rate_threshold", 0)
        or float(__import__("os").environ.get("EVAL_PASS_RATE_THRESHOLD", "0.8"))
    )

    avg_score = results.get("avg_score", 0.0)
    pass_rate = results.get("pass_rate", 0.0)
    total = results.get("total", 0)
    successful = results.get("successful", 0)
    failed = results.get("failed", 0)

    print("=" * 60)
    print("  QUALITY GATE CHECK")
    print("=" * 60)
    print(f"  Results file:     {results_path}")
    print(f"  Total items:      {total}")
    print(f"  Successful:       {successful}")
    print(f"  Failed:           {failed}")
    print("-" * 60)
    print(f"  Average score:    {avg_score:.3f}  (threshold: {score_threshold})")
    print(f"  Pass rate:        {pass_rate:.1%}  (threshold: {pass_rate_threshold:.1%})")
    print("=" * 60)

    score_ok = avg_score >= score_threshold
    rate_ok = pass_rate >= pass_rate_threshold

    if not score_ok:
        print(f"  [FAIL] Average score {avg_score:.3f} < {score_threshold}")
    if not rate_ok:
        print(f"  [FAIL] Pass rate {pass_rate:.1%} < {pass_rate_threshold:.1%}")

    gate_passed = score_ok and rate_ok

    if gate_passed:
        print("\n  RESULT: PASSED")
    else:
        print("\n  RESULT: FAILED")

    # Print per-item details if available
    details = results.get("details", [])
    if details:
        print(f"\n{'=' * 60}")
        print("  PER-ITEM DETAILS")
        print(f"{'=' * 60}")
        for item in details:
            status = "PASS" if item.get("success", False) else "FAIL"
            item_id = item.get("item_id", "unknown")
            combined = item.get("combined_score", 0.0)
            error = item.get("error", "")
            suffix = f" ({error})" if error else ""
            print(f"  [{status}] {item_id}: {combined:.3f}{suffix}")

    return gate_passed


if __name__ == "__main__":
    default_path = "evaluation_results.json"
    path_arg = sys.argv[1] if len(sys.argv) > 1 else default_path

    passed = check_quality_gate(path_arg)
    sys.exit(0 if passed else 1)
