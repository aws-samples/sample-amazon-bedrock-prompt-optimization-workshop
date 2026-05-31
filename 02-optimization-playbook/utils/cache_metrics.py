"""
Cache metrics utilities for Amazon Bedrock prompt caching.

This module provides helper functions to:
- Extract cache metrics from Bedrock responses
- Display cache metrics in a readable format
- Calculate cost savings from caching
"""

from __future__ import annotations


def extract_cache_metrics(response):
    """
    Extract cache metrics from Bedrock response.

    Args:
        response: Response from bedrock_runtime.converse()

    Returns:
        dict with keys:
            - input_tokens: Regular input tokens
            - output_tokens: Output tokens generated
            - cache_write: Cache write input tokens (first occurrence)
            - cache_read: Cache read input tokens (cached content reused)
    """
    usage = response.get("usage", {})

    metrics = {
        "input_tokens": usage.get("inputTokens", 0),
        "output_tokens": usage.get("outputTokens", 0),
        "cache_write": usage.get("cacheWriteInputTokens", 0),
        "cache_read": usage.get("cacheReadInputTokens", 0),
    }

    return metrics


def print_cache_metrics(metrics, request_num=None):
    """
    Pretty print cache metrics with color-coded status.

    Args:
        metrics: Dict from extract_cache_metrics()
        request_num: Optional request number to display
    """
    header = f"Request {request_num}" if request_num else "Cache Metrics"

    print(f"\n{'=' * 60}")
    print(f"{header}")
    print(f"{'=' * 60}")
    print(f"Input tokens:       {metrics['input_tokens']:,}")
    print(f"Output tokens:      {metrics['output_tokens']:,}")
    print(f"Cache write tokens: {metrics['cache_write']:,}")
    print(f"Cache read tokens:  {metrics['cache_read']:,}")

    print(f"{'=' * 60}\n")  # Extra newline for spacing


def calculate_cache_savings(all_metrics, input_price_per_million=3.0):
    """
    Calculate total cost savings from caching across multiple requests.

    Cost structure (as of Dec 2025):
    - Regular input tokens: 1.0x base price
    - Cache write tokens: 1.25x base price (investment)
    - Cache read tokens: 0.1x base price (90% savings)

    For current pricing, see: https://aws.amazon.com/bedrock/pricing/

    Args:
        all_metrics: List of metric dicts from extract_cache_metrics()
        input_price_per_million: Price per million input tokens (default: $3.00 for Claude Sonnet)

    Returns:
        dict with keys:
            - total_requests: Number of requests analyzed
            - cost_with_cache: Total cost with caching enabled
            - cost_no_cache: Total cost if caching was not used
            - savings: Dollar amount saved
            - savings_pct: Percentage savings
            - cache_hit_rate: Percentage of cacheable tokens that were cache hits
            - total_cache_write: Total cache write tokens
            - total_cache_read: Total cache read tokens
    """
    total_input = sum(m["input_tokens"] for m in all_metrics)
    total_cache_write = sum(m["cache_write"] for m in all_metrics)
    total_cache_read = sum(m["cache_read"] for m in all_metrics)

    # Cost with caching
    # Note: inputTokens from Bedrock = only NEW tokens (not cached)
    # So we don't need to subtract cache tokens - they're already separate!
    cost_with_cache = (
        (total_input / 1_000_000) * input_price_per_million
        + (total_cache_write / 1_000_000) * input_price_per_million * 1.25
        + (total_cache_read / 1_000_000) * input_price_per_million * 0.1
    )

    # Cost without caching (all tokens at regular price)
    # Total tokens = new input + cache write + cache read
    total_tokens_without_cache = total_input + total_cache_write + total_cache_read
    cost_no_cache = (total_tokens_without_cache / 1_000_000) * input_price_per_million

    # Calculate savings
    savings = cost_no_cache - cost_with_cache
    savings_pct = (savings / cost_no_cache * 100) if cost_no_cache > 0 else 0

    # Calculate cache hit rate
    # Hit rate = cache reads / (cache reads + cache writes)
    total_cacheable = total_cache_read + total_cache_write
    cache_hit_rate = (total_cache_read / total_cacheable * 100) if total_cacheable > 0 else 0

    return {
        "total_requests": len(all_metrics),
        "cost_with_cache": cost_with_cache,
        "cost_no_cache": cost_no_cache,
        "savings": savings,
        "savings_pct": savings_pct,
        "cache_hit_rate": cache_hit_rate,
        "total_cache_write": total_cache_write,
        "total_cache_read": total_cache_read,
    }
