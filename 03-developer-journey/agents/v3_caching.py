"""
V3 Caching Agent - Same as v2 + prompt caching.
- All v2 optimizations (structured prompt, max_tokens, stop_sequences, low temperature)
- System prompt caching with SystemContentBlock + cachePoint (provider-agnostic)
- Tool definition caching with cache_tools="default" on BedrockModel
- Note: System prompt must be 1,024+ tokens for caching to activate
"""

from __future__ import annotations

import os

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import Agent
from strands.models import BedrockModel
from strands.telemetry import StrandsTelemetry
from strands.types.content import SystemContentBlock

from utils.agent_config import MODEL_SONNET, SYSTEM_PROMPT_TEXT, setup_langfuse_telemetry
from utils.tools import get_product_info, get_return_policy, get_technical_support, web_search

setup_langfuse_telemetry()

app = BedrockAgentCoreApp()

# Provider-agnostic caching using SystemContentBlock
SYSTEM_PROMPT = [
    SystemContentBlock(text=SYSTEM_PROMPT_TEXT),
    SystemContentBlock(cachePoint={"type": "default"}),
]


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
        cache_tools="default",
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )

    agent = Agent(
        model=model,
        tools=[get_return_policy, get_product_info, web_search, get_technical_support],
        system_prompt=SYSTEM_PROMPT,
        name="customer-support-v3-caching",
        trace_attributes={
            "version": "v3-caching",
            "langfuse.tags": ["caching", "system-prompt-cache", "tool-cache"],
        },
    )

    response = agent(user_input)
    response_text = response.message["content"][0]["text"]

    telemetry.tracer_provider.force_flush()

    return response_text


if __name__ == "__main__":
    app.run()
