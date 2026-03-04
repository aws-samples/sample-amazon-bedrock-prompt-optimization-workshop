"""
V6 Gateway Agent - AgentCore Gateway with semantic tool search.
"""

from __future__ import annotations

import base64
import os
import uuid

import httpx
import requests
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from mcp.client.streamable_http import streamable_http_client
from mcp.types import Tool as MCPTool
from strands import Agent
from strands.models import BedrockModel
from strands.telemetry import StrandsTelemetry
from strands.tools.mcp import MCPClient
from strands.tools.mcp.mcp_client import MCPAgentTool
from strands.types.content import SystemContentBlock

from utils.agent_config import (
    MODEL_HAIKU,
    MODEL_SONNET,
    SYSTEM_PROMPT_TEXT,
    classify_query_complexity,
    setup_langfuse_telemetry,
)

setup_langfuse_telemetry()

app = BedrockAgentCoreApp()

GATEWAY_URL = os.environ.get("GATEWAY_URL")
GUARDRAIL_ID = os.environ.get("GUARDRAIL_ID")

# Cognito credentials (injected as env vars at deploy time)
COGNITO_CLIENT_ID = os.environ.get("COGNITO_CLIENT_ID")
COGNITO_CLIENT_SECRET = os.environ.get("COGNITO_CLIENT_SECRET")
COGNITO_TOKEN_URL = os.environ.get("COGNITO_TOKEN_URL")
COGNITO_SCOPE = os.environ.get("COGNITO_SCOPE")

SYSTEM_PROMPT = [
    SystemContentBlock(text=SYSTEM_PROMPT_TEXT),
    SystemContentBlock(cachePoint={"type": "default"}),
]


def get_cognito_token():
    """Get OAuth2 token from Cognito using env vars."""
    if not all([COGNITO_CLIENT_ID, COGNITO_CLIENT_SECRET, COGNITO_TOKEN_URL, COGNITO_SCOPE]):
        print("Missing Cognito credentials in env vars")
        print(f"  COGNITO_CLIENT_ID: {'set' if COGNITO_CLIENT_ID else 'missing'}")
        print(f"  COGNITO_CLIENT_SECRET: {'set' if COGNITO_CLIENT_SECRET else 'missing'}")
        print(f"  COGNITO_TOKEN_URL: {'set' if COGNITO_TOKEN_URL else 'missing'}")
        print(f"  COGNITO_SCOPE: {'set' if COGNITO_SCOPE else 'missing'}")
        return None

    auth = base64.b64encode(f"{COGNITO_CLIENT_ID}:{COGNITO_CLIENT_SECRET}".encode()).decode()
    response = requests.post(
        COGNITO_TOKEN_URL,
        headers={
            "Authorization": f"Basic {auth}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={"grant_type": "client_credentials", "scope": COGNITO_SCOPE},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def search_results_to_tools(results, client, top_n=1):
    """Convert search results to MCPAgentTool objects.

    Returns only top_n tools (default 1) to demonstrate semantic ranking.
    In production with many tools, you might use a higher limit or relevance threshold.
    """
    tools = []
    for tool in results[:top_n]:
        mcp_tool = MCPTool(
            name=tool["name"],
            description=tool["description"],
            inputSchema=tool["inputSchema"],
        )
        tools.append(MCPAgentTool(mcp_tool, client))
    return tools


@app.entrypoint
def invoke(payload, context=None):
    user_input = payload.get("prompt", "")

    telemetry = StrandsTelemetry()
    telemetry.setup_otlp_exporter()

    complexity = classify_query_complexity(user_input)
    model_id = MODEL_HAIKU if complexity == "simple" else MODEL_SONNET

    # Get Cognito token for Gateway auth
    bearer_token = get_cognito_token()
    if not bearer_token:
        return {"error": "Failed to get Cognito token - check env vars"}

    mcp_client = MCPClient(
        lambda: streamable_http_client(
            url=GATEWAY_URL,
            http_client=httpx.AsyncClient(
                headers={"Authorization": f"Bearer {bearer_token}"},
                timeout=30.0,
            ),
        )
    )

    with mcp_client:
        # Semantic search for relevant tools
        search_result = mcp_client.call_tool_sync(
            tool_use_id=str(uuid.uuid4()),
            name="x_amz_bedrock_agentcore_search",
            arguments={"query": user_input},
        )
        found_tools = search_result["structuredContent"]["tools"]
        tools = search_results_to_tools(found_tools, mcp_client)

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
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
            name="customer-support-v6-gateway",
            trace_attributes={
                "version": "v6-gateway",
                "tools_loaded": len(tools),
                "query_complexity": complexity,
                "langfuse.tags": ["gateway", "semantic-search"],
            },
        )

        response = agent(user_input)
        response_text = response.message["content"][0]["text"]

        telemetry.tracer_provider.force_flush()

        return {"response": response_text, "tools_loaded": len(tools)}


if __name__ == "__main__":
    app.run()
