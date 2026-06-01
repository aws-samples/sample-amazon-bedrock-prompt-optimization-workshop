"""
RAGAS quality-evaluation helpers for Lab 07.

Runs three RAGAS metrics against agents invoked LOCALLY (in-notebook), then
pushes each score onto the agent's Langfuse trace so they show up as numeric
Scores alongside the operational metrics.

Metrics and their scope:
- tool_call_accuracy  -- tool SELECTION (name-only), all scenarios with expected
                         tool(s). No judge LLM.
- answer_correctness  -- extracted `answer` field vs an authored `reference`,
                         any scenario carrying a `reference`. LLM judge + embeddings.
- context_precision   -- retrieved KB chunks vs `reference`, RAG scenarios only
                         (`is_rag: true`, i.e. get_technical_support). LLM judge.

Judge + embeddings run on Bedrock via the langchain-aws wrappers (no OpenAI key).

Stack (verified live against Bedrock -- see CONTEXT.md and
docs/adr/0001-local-ragas-quality-evals.md):
    ragas==0.4.0, langchain-aws, langchain-community==0.3.27
The legacy `ragas.metrics` API is used deliberately: the newer
`ragas.metrics.collections` API reaches Bedrock via LiteLLM+Instructor, which
sends temperature+top_p together and Claude Sonnet 4.5 rejects that.
"""

from __future__ import annotations

import asyncio
import re
import warnings

# RAGAS emits a cosmetic deprecation warning for the Langchain wrappers; the
# legacy API is the only one that works with a Bedrock judge today.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

from ragas import SingleTurnSample
from ragas.dataset_schema import MultiTurnSample
from ragas.messages import AIMessage, HumanMessage
from ragas.messages import ToolCall as RagasToolCall
from ragas.metrics import (
    AnswerCorrectness,
    AnswerSimilarity,
    LLMContextPrecisionWithReference,
    ToolCallAccuracy,
)

from utils.agent_config import MODEL_SONNET


# --------------------------------------------------------------------------
# Judge / embeddings on Bedrock
# --------------------------------------------------------------------------
def build_bedrock_evaluator(judge_model_id: str = MODEL_SONNET, region: str | None = None):
    """Build a RAGAS judge LLM + embeddings backed by Amazon Bedrock.

    Returns (judge, embeddings) wrapped for the legacy ragas.metrics API.
    Judge = ChatBedrockConverse (temperature 0 for stable grading), embeddings
    = Titan Text v2.
    """
    import os

    from langchain_aws import BedrockEmbeddings, ChatBedrockConverse
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    region = region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

    judge = LangchainLLMWrapper(
        ChatBedrockConverse(model=judge_model_id, region_name=region, temperature=0.0)
    )
    embeddings = LangchainEmbeddingsWrapper(
        BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", region_name=region)
    )
    return judge, embeddings


# --------------------------------------------------------------------------
# Local invocation + capturing tool calls / retrieved contexts
# --------------------------------------------------------------------------
def invoke_and_capture(agent, query: str) -> dict:
    """Invoke a Strands Agent locally and capture what RAGAS needs.

    Reads tool calls and tool results from `agent.messages` after the run:
    - tool_calls          -- names of every tool the agent invoked (toolUse blocks)
    - retrieved_contexts  -- text returned by get_technical_support (the KB chunks),
                             i.e. toolResult blocks for the RAG tool
    - response            -- the agent's final text answer

    A fresh agent per scenario keeps conversations isolated (no shared history).
    """
    result = agent(query)
    response_text = ""
    try:
        response_text = result.message["content"][0]["text"]
    except (AttributeError, KeyError, IndexError, TypeError):
        response_text = str(getattr(result, "message", result))

    tool_calls: list[str] = []
    retrieved_contexts: list[str] = []
    # Map toolUseId -> tool name so we can attribute results to the RAG tool.
    rag_tool_use_ids: set[str] = set()

    for message in agent.messages:
        for block in message.get("content") or []:
            if not isinstance(block, dict):
                continue
            if "toolUse" in block:
                name = block["toolUse"]["name"]
                tool_calls.append(name)
                if name == "get_technical_support":
                    rag_tool_use_ids.add(block["toolUse"].get("toolUseId", ""))
            if "toolResult" in block:
                tr = block["toolResult"]
                if tr.get("toolUseId", "") in rag_tool_use_ids:
                    for c in tr.get("content", []):
                        if isinstance(c, dict) and c.get("text"):
                            retrieved_contexts.append(c["text"])

    return {
        "response": response_text,
        "tool_calls": tool_calls,
        "retrieved_contexts": retrieved_contexts,
    }


# --------------------------------------------------------------------------
# Extracting the agent's answer
# --------------------------------------------------------------------------
def extract_answer(response_text: str) -> str:
    """Pull the `answer` field out of a v2+ structured response.

    v4 emits a markdown `**answer:** ...` block; v1 emits free prose. If no
    structured answer field is found, the whole response is returned so v1 and
    v4 stay comparable against the same plain-prose references.
    """
    if not response_text:
        return ""

    # Match an "answer" label with markdown bold in any position
    # (**answer:**, **answer**:, answer:) and capture up to the next field label.
    match = re.search(
        r"\**\s*answer\s*\**\s*:\s*\**\s*(.+?)(?=\n?\s*[-*]?\s*\**\s*(?:category|confidence)\s*\**\s*:|$)",
        response_text,
        re.IGNORECASE | re.DOTALL,
    )
    answer = match.group(1) if match else response_text
    # Strip stray leading/trailing markdown bold markers and list bullets.
    return answer.strip().strip("*").strip().lstrip("-").strip()


# --------------------------------------------------------------------------
# Metric runners (each returns a float in [0, 1] or None)
# --------------------------------------------------------------------------
def _run_async(coro):
    """Run an async RAGAS scorer from sync notebook code."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def score_tool_call_accuracy(query: str, actual_tools: list[str], expected_tools: list[str]) -> float | None:
    """Name-only tool selection accuracy.

    Tool calls are built with empty args so RAGAS scores tool SELECTION only --
    args are intentionally ignored (the tools normalize input, so arg matching
    would create false negatives). See CONTEXT.md.
    """
    if not expected_tools:
        return None

    ai = AIMessage(
        content="",
        tool_calls=[RagasToolCall(name=name, args={}) for name in actual_tools],
    )
    sample = MultiTurnSample(
        user_input=[HumanMessage(content=query), ai],
        reference_tool_calls=[RagasToolCall(name=name, args={}) for name in expected_tools],
    )
    return _run_async(ToolCallAccuracy().multi_turn_ascore(sample))


def score_answer_correctness(query: str, answer: str, reference: str, judge, embeddings) -> float | None:
    """Answer Correctness: extracted answer vs authored reference.

    AnswerSimilarity must be set explicitly -- AnswerCorrectness does NOT
    auto-initialize it (asserts otherwise).
    """
    scorer = AnswerCorrectness(
        llm=judge,
        embeddings=embeddings,
        answer_similarity=AnswerSimilarity(embeddings=embeddings),
    )
    sample = SingleTurnSample(user_input=query, response=answer, reference=reference)
    return _run_async(scorer.single_turn_ascore(sample))


def score_context_precision(query: str, reference: str, retrieved_contexts: list[str], judge) -> float | None:
    """Context Precision (with reference): RAG scenarios only."""
    if not retrieved_contexts:
        return None
    sample = SingleTurnSample(
        user_input=query,
        reference=reference,
        retrieved_contexts=retrieved_contexts,
    )
    return _run_async(LLMContextPrecisionWithReference(llm=judge).single_turn_ascore(sample))


# --------------------------------------------------------------------------
# Push scores to Langfuse (v3 API, attach to an existing trace_id)
# --------------------------------------------------------------------------
def push_scores_to_langfuse(trace_id: str, scores: dict[str, float]) -> None:
    """Attach numeric RAGAS scores to an existing Langfuse trace.

    Uses the v3 API client (langfuse.api.scores.create); the v2
    score_current_trace decorator API does not apply when scoring by trace_id.
    """
    import os

    import httpx
    from langfuse import Langfuse

    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    host = os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
    if not public_key or not secret_key:
        print("  (skipping Langfuse push -- keys not set)")
        return

    langfuse = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
        httpx_client=httpx.Client(timeout=60),
    )
    for name, value in scores.items():
        if value is None:
            continue
        langfuse.api.scores.create(
            trace_id=trace_id,
            name=name,
            value=float(value),
            data_type="NUMERIC",
        )
    langfuse.shutdown()
