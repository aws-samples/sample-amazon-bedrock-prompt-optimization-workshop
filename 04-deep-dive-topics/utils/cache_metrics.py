"""
Cache metrics utilities for advanced prompt caching module.
"""

from __future__ import annotations


def extract_cache_metrics(response):
    """Extract cache metrics from Bedrock Converse API response."""
    usage = response.get("usage", {})
    return {
        "input_tokens": usage.get("inputTokens", 0),
        "output_tokens": usage.get("outputTokens", 0),
        "cache_write": usage.get("cacheWriteInputTokens", 0),
        "cache_read": usage.get("cacheReadInputTokens", 0),
    }


def extract_invoke_metrics(response):
    """Extract cache metrics from Bedrock InvokeModel API response."""
    usage = response.get("usage", {})
    cache_creation = usage.get("cache_creation", {})

    return {
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "cache_write": usage.get("cache_creation_input_tokens", 0),
        "cache_read": usage.get("cache_read_input_tokens", 0),
        "ttl_5m_write": cache_creation.get("ephemeral_5m_input_tokens", 0),
        "ttl_1h_write": cache_creation.get("ephemeral_1h_input_tokens", 0),
    }


def print_cache_metrics(metrics, label=None):
    """Pretty print cache metrics."""
    header = f"Request {label}" if label else "Cache Metrics"
    print(f"{'=' * 50}")
    print(header)
    print(f"{'=' * 50}")
    print(f"Input tokens:  {metrics['input_tokens']:,}")
    print(f"Output tokens: {metrics['output_tokens']:,}")
    print(f"Cache write:   {metrics['cache_write']:,}")
    print(f"Cache read:    {metrics['cache_read']:,}")


def calculate_savings(metrics_list, input_price=3.0):
    """
    Calculate cost savings from caching across multiple requests.

    Cost structure (5-minute TTL):
    - Regular input: 1x base price
    - Cache write: 1.25x base price
    - Cache read: 0.1x base price

    For current pricing: https://aws.amazon.com/bedrock/pricing/
    """
    total_input = sum(m["input_tokens"] for m in metrics_list)
    total_write = sum(m["cache_write"] for m in metrics_list)
    total_read = sum(m["cache_read"] for m in metrics_list)

    cost_cached = (
        (total_input / 1_000_000) * input_price
        + (total_write / 1_000_000) * input_price * 1.25
        + (total_read / 1_000_000) * input_price * 0.1
    )

    total_tokens = total_input + total_write + total_read
    cost_no_cache = (total_tokens / 1_000_000) * input_price

    savings_pct = ((cost_no_cache - cost_cached) / cost_no_cache * 100) if cost_no_cache > 0 else 0
    hit_rate = (total_read / (total_read + total_write) * 100) if (total_read + total_write) > 0 else 0

    return {
        "hit_rate": hit_rate,
        "savings_pct": savings_pct,
        "cache_write": total_write,
        "cache_read": total_read,
        "cost_cached": cost_cached,
        "cost_no_cache": cost_no_cache,
    }


def analyze_caching_roi(metrics_list, input_price=3.0):
    """
    Analyze caching ROI with detailed breakdown.
    """
    total_input = sum(m["input_tokens"] for m in metrics_list)
    total_write = sum(m["cache_write"] for m in metrics_list)
    total_read = sum(m["cache_read"] for m in metrics_list)

    cost_with_cache = (
        (total_input / 1_000_000) * input_price
        + (total_write / 1_000_000) * input_price * 1.25
        + (total_read / 1_000_000) * input_price * 0.1
    )

    total_tokens = total_input + total_write + total_read
    cost_without_cache = (total_tokens / 1_000_000) * input_price

    savings = cost_without_cache - cost_with_cache
    hit_rate = (total_read / (total_read + total_write) * 100) if (total_read + total_write) > 0 else 0
    roi_pct = (savings / cost_without_cache * 100) if cost_without_cache > 0 else 0

    return {
        "total_requests": len(metrics_list),
        "total_tokens": total_tokens,
        "cache_write": total_write,
        "cache_read": total_read,
        "hit_rate": hit_rate,
        "cost_with_cache": cost_with_cache,
        "cost_without_cache": cost_without_cache,
        "savings": savings,
        "roi_pct": roi_pct,
    }
