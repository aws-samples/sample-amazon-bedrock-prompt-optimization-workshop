"""
Metrics extraction and cost calculation utilities.
"""

# Bedrock pricing per 1M tokens (as of 2024)
from __future__ import annotations

PRICING = {
    "sonnet": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
    "haiku": {
        "input": 0.25,
        "output": 1.25,
        "cache_write": 0.30,
        "cache_read": 0.03,
    },
}


def extract_metrics(response) -> dict:
    """Extract cost and latency metrics from agent response."""
    usage = response.metrics.accumulated_usage
    return {
        "input_tokens": usage.get("inputTokens", 0),
        "output_tokens": usage.get("outputTokens", 0),
        "cache_write_tokens": usage.get("cacheWriteInputTokens", 0),
        "cache_read_tokens": usage.get("cacheReadInputTokens", 0),
        "latency_ms": response.metrics.latency_ms,
        "total_tokens": usage.get("inputTokens", 0) + usage.get("outputTokens", 0),
    }


def calculate_cost(metrics: dict, model: str = "sonnet") -> float:
    """Calculate cost based on Bedrock pricing."""
    pricing = PRICING.get(model.lower(), PRICING["sonnet"])

    input_cost = (metrics.get("input_tokens", 0) / 1_000_000) * pricing["input"]
    output_cost = (metrics.get("output_tokens", 0) / 1_000_000) * pricing["output"]
    cache_write_cost = (metrics.get("cache_write_tokens", 0) / 1_000_000) * pricing["cache_write"]
    cache_read_cost = (metrics.get("cache_read_tokens", 0) / 1_000_000) * pricing["cache_read"]

    return input_cost + output_cost + cache_write_cost + cache_read_cost


def format_cost(cost: float) -> str:
    """Format cost as currency string."""
    if cost < 0.01:
        return f"${cost:.6f}"
    return f"${cost:.4f}"


def print_comparison(baseline: dict, optimized: dict, baseline_model: str = "sonnet", optimized_model: str = "sonnet"):
    """Print a comparison table between baseline and optimized metrics."""
    baseline_cost = calculate_cost(baseline, baseline_model)
    optimized_cost = calculate_cost(optimized, optimized_model)

    print("\n" + "=" * 60)
    print("METRICS COMPARISON")
    print("=" * 60)

    print(f"\n{'Metric':<25} {'Baseline':>15} {'Optimized':>15} {'Change':>10}")
    print("-" * 65)

    metrics_to_show = [
        ("input_tokens", "Input Tokens"),
        ("output_tokens", "Output Tokens"),
        ("cache_write_tokens", "Cache Write Tokens"),
        ("cache_read_tokens", "Cache Read Tokens"),
        ("total_tokens", "Total Tokens"),
    ]

    for metric_name, display_name in metrics_to_show:
        base_val = baseline.get(metric_name, 0)
        opt_val = optimized.get(metric_name, 0)
        change_str = f"{((opt_val - base_val) / base_val) * 100:+.1f}%" if base_val > 0 else "N/A"
        print(f"{display_name:<25} {base_val:>15,} {opt_val:>15,} {change_str:>10}")

    # Latency
    base_latency = baseline.get("latency_ms", 0)
    opt_latency = optimized.get("latency_ms", 0)
    latency_change_str = f"{((opt_latency - base_latency) / base_latency) * 100:+.1f}%" if base_latency > 0 else "N/A"
    print(f"{'Latency (ms)':<25} {base_latency:>15,.0f} {opt_latency:>15,.0f} {latency_change_str:>10}")

    # Cost
    cost_change_str = (
        f"{((optimized_cost - baseline_cost) / baseline_cost) * 100:+.1f}%" if baseline_cost > 0 else "N/A"
    )
    print(
        f"{'Estimated Cost':<25} {format_cost(baseline_cost):>15} {format_cost(optimized_cost):>15} {cost_change_str:>10}"
    )

    print("-" * 65)

    if baseline_cost > 0 and optimized_cost < baseline_cost:
        savings = baseline_cost - optimized_cost
        savings_pct = (savings / baseline_cost) * 100
        print(f"\nCost Savings: {format_cost(savings)} ({savings_pct:.1f}% reduction)")

    if base_latency > 0 and opt_latency < base_latency:
        latency_improvement = base_latency - opt_latency
        latency_pct = (latency_improvement / base_latency) * 100
        print(f"Latency Improvement: {latency_improvement:.0f}ms ({latency_pct:.1f}% faster)")

    print("=" * 60 + "\n")
