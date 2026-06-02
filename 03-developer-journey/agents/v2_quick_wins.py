"""
V2 Quick Wins Agent - Low-effort optimizations.
- Concise system prompt
- max_tokens limit
- stop_sequences
- Low temperature for accuracy
"""

from __future__ import annotations

import os

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import Agent
from strands.models import BedrockModel
from strands.telemetry import StrandsTelemetry

from utils.agent_config import MODEL_SONNET, SYSTEM_PROMPT_TEXT, setup_langfuse_telemetry
from utils.tools import get_product_info, get_return_policy, get_technical_support, web_search

setup_langfuse_telemetry()

app = BedrockAgentCoreApp()

SYSTEM_PROMPT = SYSTEM_PROMPT_TEXT


@app.entrypoint
def invoke(payload):
    user_input = payload.get("prompt", "")

    telemetry = StrandsTelemetry()
    telemetry.setup_otlp_exporter()

    model = BedrockModel(
        model_id=MODEL_SONNET,
        temperature=0.1,
        max_tokens=1024,
        stop_sequences=["###", "END_RESPONSE"],
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )

    agent = Agent(
        model=model,
        tools=[get_return_policy, get_product_info, web_search, get_technical_support],
        system_prompt=SYSTEM_PROMPT,
        name="customer-support-v2-quick-wins",
        trace_attributes={
            "version": "v2-quick-wins",
            "langfuse.tags": ["quick-wins", "concise-prompt", "max-tokens"],
        },
    )

    response = agent(user_input)
    response_text = response.message["content"][0]["text"]

    telemetry.tracer_provider.force_flush()

    return response_text


if __name__ == "__main__":
    app.run()
