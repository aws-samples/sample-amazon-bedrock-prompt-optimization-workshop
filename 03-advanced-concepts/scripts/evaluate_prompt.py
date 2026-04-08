"""
Evaluation script for CI/CD pipeline integration.

Fetches a dataset from Langfuse, runs each item through an agent
(simulated via Bedrock Converse API), scores results, and pushes
scores back to Langfuse. Exits with non-zero status if the quality
gate fails.

Usage:
    python scripts/evaluate_prompt.py

Environment variables required:
    LANGFUSE_PUBLIC_KEY  - Langfuse public key
    LANGFUSE_SECRET_KEY  - Langfuse secret key
    LANGFUSE_BASE_URL    - Langfuse host URL (default: https://cloud.langfuse.com)
    AWS_DEFAULT_REGION   - AWS region for Bedrock (default: us-east-1)

Optional CI/CD environment variables (auto-detected):
    CODEBUILD_RESOLVED_SOURCE_VERSION - Git commit SHA (CodeBuild)
    CODEBUILD_BUILD_ID                - Build identifier (CodeBuild)
    CODEBUILD_BUILD_NUMBER            - Build number (CodeBuild)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import boto3

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_NAME = os.environ.get("EVAL_DATASET_NAME", "customer-support-eval")
SCORE_THRESHOLD = float(os.environ.get("EVAL_SCORE_THRESHOLD", "0.7"))
PASS_RATE_THRESHOLD = float(os.environ.get("EVAL_PASS_RATE_THRESHOLD", "0.8"))
MODEL_ID = os.environ.get(
    "EVAL_MODEL_ID", "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
)
JUDGE_MODEL_ID = os.environ.get(
    "EVAL_JUDGE_MODEL_ID", "global.anthropic.claude-haiku-4-5-20251001-v1:0"
)
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

# System prompt used by the agent under test
SYSTEM_PROMPT = """You are a helpful customer support assistant for TechMart Electronics.
Help customers with: product information, return policies, technical support, and order inquiries.
Be concise and professional. If you cannot answer, offer to escalate."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_langfuse():
    """Return a Langfuse client or None if credentials are missing."""
    try:
        from langfuse import Langfuse

        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
        host = os.environ.get(
            "LANGFUSE_BASE_URL", "https://cloud.langfuse.com"
        )

        if not public_key or not secret_key:
            print(
                "[WARN] LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY not set. "
                "Scores will not be pushed to Langfuse."
            )
            return None

        return Langfuse(public_key=public_key, secret_key=secret_key, host=host)
    except ImportError:
        print("[WARN] langfuse package not installed. Scores will not be pushed.")
        return None


def call_agent(query: str, bedrock_client) -> str:
    """Call the agent under test via Bedrock Converse API."""
    response = bedrock_client.converse(
        modelId=MODEL_ID,
        system=[{"text": SYSTEM_PROMPT}],
        messages=[{"role": "user", "content": [{"text": query}]}],
        inferenceConfig={"maxTokens": 500, "temperature": 0.0},
    )
    return response["output"]["message"]["content"][0]["text"]


def calculate_keyword_score(response: str, expected: dict) -> float:
    """Calculate accuracy score based on expected output keywords."""
    score = 0.0
    checks = 0

    if "should_contain" in expected:
        for keyword in expected["should_contain"]:
            checks += 1
            if keyword.lower() in response.lower():
                score += 1

    if "should_use_tool" in expected:
        checks += 1
        # In a real pipeline this would check tool invocation logs;
        # here we check if the tool name appears in the response text.
        if expected["should_use_tool"].lower() in response.lower():
            score += 1

    return score / checks if checks > 0 else 1.0


def evaluate_with_llm_judge(
    query: str, response: str, bedrock_client
) -> dict[str, float | str]:
    """Use an LLM judge to score helpfulness on a 0-10 scale."""
    eval_prompt = f"""Rate the helpfulness of this customer support response on a scale of 0 to 10.

Customer Query: {query}

Agent Response: {response}

Scoring criteria:
- 0-3: Unhelpful, incorrect, or confusing
- 4-6: Partially helpful, missing key information
- 7-9: Helpful, accurate, addresses the query well
- 10: Exceptional, exceeds expectations

Return ONLY a JSON object: {{"score": <number>, "reason": "<one sentence>"}}"""

    try:
        judge_response = bedrock_client.converse(
            modelId=JUDGE_MODEL_ID,
            messages=[{"role": "user", "content": [{"text": eval_prompt}]}],
            inferenceConfig={"maxTokens": 150, "temperature": 0.0},
        )
        raw = judge_response["output"]["message"]["content"][0]["text"].strip()
        # Parse JSON from the response
        parsed = json.loads(raw)
        return {
            "score": float(parsed.get("score", 0)) / 10.0,
            "reason": parsed.get("reason", ""),
        }
    except (json.JSONDecodeError, KeyError, ValueError):
        return {"score": 0.0, "reason": "Failed to parse judge response"}


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------

def run_evaluation() -> dict:
    """Run the full evaluation pipeline."""
    bedrock = boto3.client("bedrock-runtime", region_name=REGION)
    langfuse = _init_langfuse()

    # Fetch dataset from Langfuse (single source of truth)
    if langfuse is None:
        print("[ERROR] Langfuse credentials required. Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY.")
        return {
            "total": 0, "successful": 0, "failed": 0,
            "avg_score": 0.0, "pass_rate": 0.0,
        }

    langfuse_dataset = langfuse.get_dataset(name=DATASET_NAME)
    print(f"Loaded dataset '{DATASET_NAME}' from Langfuse "
          f"({len(langfuse_dataset.items)} items)")

    eval_items: list[tuple[dict, dict, dict]] = []
    for item in langfuse_dataset.items:
        eval_items.append((item.input, item.expected_output or {}, item.metadata or {}))

    # Commit / build metadata for trace tagging
    commit_sha = os.environ.get("CODEBUILD_RESOLVED_SOURCE_VERSION", "local")
    build_id = os.environ.get("CODEBUILD_BUILD_ID", "local")
    build_number = os.environ.get("CODEBUILD_BUILD_NUMBER", "local")
    results: list[dict] = []

    for idx, (input_data, expected, metadata) in enumerate(eval_items):
        query = input_data.get("query", "")
        item_id = metadata.get("id", f"item-{idx}")
        print(f"  [{idx + 1}/{len(eval_items)}] {query[:60]}...", end=" ")

        try:
            # Run agent
            agent_response = call_agent(query, bedrock)

            # Keyword score
            keyword_score = calculate_keyword_score(agent_response, expected)

            # LLM judge score
            judge_result = evaluate_with_llm_judge(query, agent_response, bedrock)
            judge_score = judge_result["score"]

            # Combined score (average of keyword and judge)
            combined_score = (keyword_score + judge_score) / 2.0

            # Push traces and scores to Langfuse (v4 context manager API)
            if langfuse is not None:
                with langfuse.start_as_current_observation(
                    as_type="span",
                    name="ci-eval",
                    metadata={
                        "commit": commit_sha,
                        "build_id": build_id,
                        "item_id": item_id,
                    },
                ) as trace:
                    with langfuse.start_as_current_observation(
                        as_type="generation",
                        name="agent-response",
                        input=query,
                    ) as generation:
                        generation.update(output=agent_response)

                    trace_id = trace.trace_id
                    langfuse.create_score(
                        trace_id=trace_id,
                        name="keyword_accuracy",
                        value=keyword_score,
                        comment=f"Keyword match: {keyword_score:.2f}",
                    )
                    langfuse.create_score(
                        trace_id=trace_id,
                        name="helpfulness_llm",
                        value=judge_score,
                        comment=judge_result.get("reason", ""),
                    )
                    langfuse.create_score(
                        trace_id=trace_id,
                        name="combined",
                        value=combined_score,
                    )

                    trace.update(output=agent_response)

            results.append({
                "item_id": item_id,
                "keyword_score": keyword_score,
                "judge_score": judge_score,
                "combined_score": combined_score,
                "success": True,
            })
            print(f"score={combined_score:.2f}")

        except Exception as exc:
            results.append({
                "item_id": item_id,
                "keyword_score": 0.0,
                "judge_score": 0.0,
                "combined_score": 0.0,
                "success": False,
                "error": str(exc),
            })
            print(f"FAILED: {exc}")

    if langfuse is not None:
        langfuse.flush()
        # Allow a moment for flush to complete
        time.sleep(1)

    # Calculate summary
    successful = [r for r in results if r["success"]]
    avg_score = (
        sum(r["combined_score"] for r in successful) / len(successful)
        if successful
        else 0.0
    )
    pass_rate = len(successful) / len(results) if results else 0.0

    return {
        "total": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "avg_score": avg_score,
        "pass_rate": pass_rate,
        "details": results,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Prompt Evaluation Pipeline")
    print("=" * 60)
    print(f"  Dataset:          {DATASET_NAME}")
    print(f"  Model:            {MODEL_ID}")
    print(f"  Judge Model:      {JUDGE_MODEL_ID}")
    print(f"  Score Threshold:  {SCORE_THRESHOLD}")
    print(f"  Pass Rate Target: {PASS_RATE_THRESHOLD}")
    print("=" * 60)

    evaluation_results = run_evaluation()

    # Write results to JSON for CI artifact collection
    output_path = Path("evaluation_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2)
    print(f"\nResults written to {output_path}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("  EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total items:    {evaluation_results['total']}")
    print(f"  Successful:     {evaluation_results['successful']}")
    print(f"  Failed:         {evaluation_results['failed']}")
    print(f"  Average score:  {evaluation_results['avg_score']:.3f}")
    print(f"  Pass rate:      {evaluation_results['pass_rate']:.1%}")
    print(f"{'=' * 60}")

    # Quality gate check
    gate_passed = (
        evaluation_results["avg_score"] >= SCORE_THRESHOLD
        and evaluation_results["pass_rate"] >= PASS_RATE_THRESHOLD
    )

    if gate_passed:
        print(
            f"\n  QUALITY GATE PASSED: "
            f"avg_score={evaluation_results['avg_score']:.3f} >= {SCORE_THRESHOLD}, "
            f"pass_rate={evaluation_results['pass_rate']:.1%} >= {PASS_RATE_THRESHOLD:.1%}"
        )
        sys.exit(0)
    else:
        print(
            f"\n  QUALITY GATE FAILED: "
            f"avg_score={evaluation_results['avg_score']:.3f} (need >= {SCORE_THRESHOLD}), "
            f"pass_rate={evaluation_results['pass_rate']:.1%} (need >= {PASS_RATE_THRESHOLD:.1%})"
        )
        sys.exit(1)
