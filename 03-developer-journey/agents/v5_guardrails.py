"""
V5 Guardrails Agent - Bedrock Guardrails for content filtering.
- Block off-topic queries before LLM processes them
- Save tokens on irrelevant requests
"""

from __future__ import annotations

import os

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import Agent
from strands.models import BedrockModel
from strands.telemetry import StrandsTelemetry
from strands.types.content import SystemContentBlock

from utils.agent_config import (
    MODEL_HAIKU,
    MODEL_SONNET,
    SYSTEM_PROMPT_TEXT,
    classify_query_complexity,
    setup_langfuse_telemetry,
)
from utils.tools import get_product_info, get_return_policy, get_technical_support, web_search

setup_langfuse_telemetry()

app = BedrockAgentCoreApp()

GUARDRAIL_ID = os.environ.get("GUARDRAIL_ID")

SYSTEM_PROMPT = [
    SystemContentBlock(text=SYSTEM_PROMPT_TEXT),
    SystemContentBlock(cachePoint={"type": "default"}),
]


@app.entrypoint
def invoke(payload):
    user_input = payload.get("prompt", "")

    telemetry = StrandsTelemetry()
    telemetry.setup_otlp_exporter()

    complexity = classify_query_complexity(user_input)
    model_id = MODEL_HAIKU if complexity == "simple" else MODEL_SONNET

    model_kwargs = {
        "model_id": model_id,
        "temperature": 0.1,
        "max_tokens": 1024,
        "stop_sequences": ["###", "END_RESPONSE"],
        "cache_tools": "default",
        "region_name": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    }

    if GUARDRAIL_ID:
        model_kwargs["guardrail_id"] = GUARDRAIL_ID
        model_kwargs["guardrail_version"] = "DRAFT"
        model_kwargs["guardrail_trace"] = "enabled"

    agent = Agent(
        model=BedrockModel(**model_kwargs),
        tools=[get_return_policy, get_product_info, web_search, get_technical_support],
        system_prompt=SYSTEM_PROMPT,
        name="customer-support-v5-guardrails",
        trace_attributes={
            "version": "v5-guardrails",
            "query_complexity": complexity,
            "guardrails_enabled": bool(GUARDRAIL_ID),
            "langfuse.tags": ["guardrails", complexity],
        },
    )

    response = agent(user_input)
    response_text = response.message["content"][0]["text"]

    telemetry.tracer_provider.force_flush()

    return {"response": response_text, "guardrails_enabled": bool(GUARDRAIL_ID)}


if __name__ == "__main__":
    app.run()
