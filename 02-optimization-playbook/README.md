# Part 2: Optimization Playbook

Sixteen cost-and-latency techniques, organized into three effort tiers. Each tier consolidates its runnable techniques into one notebook so you can step through them in sequence. Each technique is also written up as a standalone page in the walkthrough (theory + trade-offs + references) — this folder is the runnable code.

The order is the point: **start LOW, earn HIGH effort.**

## Notebooks

| Notebook | Duration | Levers covered |
|----------|----------|----------------|
| [01-low-effort](./01-low-effort.ipynb) | 45 min | Model selection · Prompt design · Parameter tuning · Prompt caching · Prompt engineering tricks (CoD, Verbalized Sampling, Self-Refine + managed Bedrock APO) · Adaptive thinking |
| [02-medium-effort](./02-medium-effort.ipynb) | 60 min | LLM routing · Bedrock Guardrails · RAG / indexing · Prompt compression (demo) · Conversation & memory (sliding window + AgentCore Memory) · Batch inference (demo) |
| [03-high-effort](./03-high-effort.ipynb) | 60 min | Sub-agent delegation (Claude Agent SDK) · Tool search via MCP Gateway (AgentCore Gateway) |

## Covered as concept / reference (no live lab)

Documented in the walkthrough; each is real but doesn't fit a live workshop slot:

- **Lever 13: Harness engineering** — a practice, mapped across the workshop's other labs
- **Lever 15: GEPA / DSPy** — open-source programmatic prompt optimization; a `compile()` run is too long for a live slot

## Prerequisites

- Completed [Part 1: Fundamentals](../01-fundamentals/)
- AWS credentials with Bedrock + AgentCore access
- All Python dependencies are pre-installed from the repo's `requirements.txt` (Strands, Claude Agent SDK, AgentCore SDKs, LLMLingua, DSPy). Running locally? From the repo root: `uv pip install -r requirements.txt`
