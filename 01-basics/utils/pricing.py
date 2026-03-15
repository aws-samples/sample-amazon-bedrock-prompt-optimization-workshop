"""
Pricing and cost calculation utilities for AWS Bedrock.
"""

# Pricing per 1M tokens (as of January 2026)
# Using global CRIS profile pricing (~10% savings vs in-region)
# Always verify current pricing at: https://aws.amazon.com/bedrock/pricing/
from __future__ import annotations

PRICING = {
    # Global CRIS profile - recommended for most use cases
    "global.anthropic.claude-sonnet-4-6": {
        "name": "Claude Sonnet 4.6 (Global)",
        "input": 3.00,
        "output": 15.00,
        # 5-minute cache pricing (standard TTL)
        "cache_write_5m": 3.75,  # 1.25x input - first time writing to cache
        "cache_read_5m": 0.30,  # 0.1x input - 90% savings on cache hits!
    },
    "global.anthropic.claude-3-5-haiku-20241022-v1:0": {
        "name": "Claude Haiku 4.5 (Global)",
        "input": 1.00,
        "output": 5.00,
        "cache_write_5m": 1.25,
        "cache_read_5m": 0.10,
    },
}

# Output burndown rate for TPM quota calculation
OUTPUT_BURNDOWN_RATE = 5  # For Claude Sonnet 4.6 and newer


def calculate_cost(input_tokens, output_tokens, num_requests, model_id, cache_write_tokens=0, cache_read_tokens=0):
    """
    Calculate the cost for Bedrock inference requests.

    Args:
        input_tokens: Number of input tokens per request (uncached)
        output_tokens: Number of output tokens per request
        num_requests: Total number of requests
        model_id: The model ID for pricing lookup
        cache_write_tokens: Tokens written to cache (1.25x cost for 5m cache)
        cache_read_tokens: Tokens read from cache (0.1x cost for 5m cache)

    Returns:
        dict with detailed cost breakdown
    """
    if model_id not in PRICING:
        raise ValueError(f"Unknown model: {model_id}. Available: {list(PRICING.keys())}")

    prices = PRICING[model_id]

    # Calculate costs (prices are per 1M tokens)
    total_input = input_tokens * num_requests
    total_output = output_tokens * num_requests

    input_cost = (total_input / 1_000_000) * prices["input"]
    output_cost = (total_output / 1_000_000) * prices["output"]
    cache_write_cost = (cache_write_tokens / 1_000_000) * prices["cache_write_5m"]
    cache_read_cost = (cache_read_tokens / 1_000_000) * prices["cache_read_5m"]

    total_cost = input_cost + output_cost + cache_write_cost + cache_read_cost

    return {
        "model": prices["name"],
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "cache_write_cost": cache_write_cost,
        "cache_read_cost": cache_read_cost,
        "total_cost": total_cost,
    }


def calculate_actual_cost(input_tokens, output_tokens, model_id):
    """
    Calculate actual cost in dollars for a single request.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model_id: The model ID for pricing lookup

    Returns:
        float: Total cost in dollars
    """
    if model_id not in PRICING:
        raise ValueError(f"Unknown model: {model_id}. Available: {list(PRICING.keys())}")

    prices = PRICING[model_id]
    input_cost = (input_tokens / 1_000_000) * prices["input"]
    output_cost = (output_tokens / 1_000_000) * prices["output"]
    return input_cost + output_cost


def compare_optimization(original_tokens, optimized_tokens, output_tokens, num_requests, model_id):
    """
    Compare costs between unoptimized and optimized prompts.

    Args:
        original_tokens: Input tokens for unoptimized prompt
        optimized_tokens: Input tokens for optimized prompt
        output_tokens: Output tokens per request
        num_requests: Total number of requests
        model_id: The model ID for pricing lookup

    Returns:
        dict with original cost, optimized cost, and savings
    """
    original = calculate_cost(original_tokens, output_tokens, num_requests, model_id)
    optimized = calculate_cost(optimized_tokens, output_tokens, num_requests, model_id)

    savings = original["total_cost"] - optimized["total_cost"]
    savings_pct = (savings / original["total_cost"]) * 100 if original["total_cost"] > 0 else 0

    return {"original": original, "optimized": optimized, "savings": savings, "savings_pct": savings_pct}


def calculate_tpm_reservation(input_tokens, max_tokens, burndown_rate=OUTPUT_BURNDOWN_RATE):
    """
    Calculate TPM quota RESERVED at request start (based on max_tokens).

    Args:
        input_tokens: Number of input tokens
        max_tokens: The max_tokens setting in the API call
        burndown_rate: Output token burndown rate (default 5x for Claude Sonnet 4.6)

    Returns:
        int: Total TPM quota reserved
    """
    return input_tokens + (max_tokens * burndown_rate)


def calculate_tpm_actual(input_tokens, output_tokens, cache_write_tokens=0, burndown_rate=OUTPUT_BURNDOWN_RATE):
    """
    Calculate ACTUAL TPM consumption after request completes.

    TPM Formula: InputTokenCount + CacheWriteInputTokens + (OutputTokenCount x burndown_rate)

    Args:
        input_tokens: Number of input tokens
        output_tokens: Actual output tokens generated
        cache_write_tokens: Tokens written to cache
        burndown_rate: Output token burndown rate (default 5x for Claude Sonnet 4.6)

    Returns:
        int: Actual TPM consumed
    """
    return input_tokens + cache_write_tokens + (output_tokens * burndown_rate)


def print_pricing_table():
    """Print a formatted pricing table for all models."""
    print("Model Pricing (per 1M tokens) - as of January 2026:")
    print("=" * 90)
    print(f"{'Model':<30} {'Input':>10} {'Output':>10} {'Cache Write':>14} {'Cache Read':>14}")
    print(f"{'':<30} {'':>10} {'':>10} {'(5m, 1.25x)':>14} {'(5m, 0.1x)':>14}")
    print("-" * 90)
    for _model_id, prices in PRICING.items():
        print(
            f"{prices['name']:<30} ${prices['input']:>8.2f} ${prices['output']:>8.2f} "
            f"${prices['cache_write_5m']:>12.2f} ${prices['cache_read_5m']:>12.2f}"
        )
    print("=" * 90)
    print("\nNote: Cache pricing shown is for 5-minute TTL cache.")
