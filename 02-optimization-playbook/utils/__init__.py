"""
Utility functions for prompt caching workshops.
"""

from __future__ import annotations

from .cache_metrics import calculate_cache_savings, extract_cache_metrics, print_cache_metrics
from .pricing import (
    OUTPUT_BURNDOWN_RATE,
    PRICING,
    calculate_actual_cost,
    calculate_cost,
    calculate_tpm_actual,
    calculate_tpm_reservation,
    compare_optimization,
    print_pricing_table,
)

__all__ = [
    "OUTPUT_BURNDOWN_RATE",
    "PRICING",
    "calculate_actual_cost",
    "calculate_cache_savings",
    "calculate_cost",
    "calculate_tpm_actual",
    "calculate_tpm_reservation",
    "compare_optimization",
    "extract_cache_metrics",
    "print_cache_metrics",
    "print_pricing_table",
]
