"""
Langfuse configuration via OpenTelemetry.
"""

from __future__ import annotations

import base64
import os

from strands.telemetry import StrandsTelemetry


def setup_langfuse():
    """
    Configure Langfuse via OpenTelemetry for agent observability.

    Requires environment variables:
    - LANGFUSE_PUBLIC_KEY
    - LANGFUSE_SECRET_KEY
    - LANGFUSE_BASE_URL (default: https://cloud.langfuse.com)
    """
    langfuse_public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    langfuse_base_url = os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

    if not langfuse_public_key or not langfuse_secret_key:
        print("Warning: LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY not set.")
        return None

    # Create Basic Auth header
    auth = base64.b64encode(f"{langfuse_public_key}:{langfuse_secret_key}".encode()).decode()

    # Disable ADOT (conflicts with Langfuse)
    os.environ["DISABLE_ADOT_OBSERVABILITY"] = "true"

    # Configure OTLP exporter for Langfuse
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{langfuse_base_url}/api/public/otel"
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {auth}"

    # Remove conflicting OTEL vars
    for var in [
        "OTEL_EXPORTER_OTLP_LOGS_HEADERS",
        "AGENT_OBSERVABILITY_ENABLED",
        "OTEL_PYTHON_DISTRO",
        "OTEL_RESOURCE_ATTRIBUTES",
        "OTEL_PYTHON_CONFIGURATOR",
    ]:
        os.environ.pop(var, None)

    telemetry = StrandsTelemetry()
    telemetry.setup_otlp_exporter()
    return telemetry
