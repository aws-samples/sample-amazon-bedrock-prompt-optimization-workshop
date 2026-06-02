# Part 1: Fundamentals

This section gets your environment ready and gives you the vocabulary every later section assumes — token economics, latency metrics, and observability.

## Learning Objectives

After completing this section, you will:
- Be comfortable navigating Jupyter notebooks (kernels, cells, shortcuts)
- Understand tokens, pricing models, and throughput limits (TPM/RPM)
- Use the CountTokens API for accurate token estimation
- Use the Converse API and the Bedrock Mantle endpoint
- Trace LLM calls and costs in Langfuse

## Notebooks

| Notebook | Duration | Description |
|----------|----------|-------------|
| [00-jupyter-notebook-101](./00-jupyter-notebook-101.ipynb) | 15 min | Kernels, cells, shortcuts, first Bedrock call — workshop tooling onboarding |
| [01-prompts-101](./01-prompts-101.ipynb) | 30 min | Tokens, pricing, TPM/RPM, CRIS, Converse API, Bedrock Mantle |
| [02-langfuse-observability](./02-langfuse-observability.ipynb) | 15 min | LLM tracing, cost tracking, production monitoring |

## Prerequisites

- AWS Account with Amazon Bedrock access
- Python 3.10+
- `.env` file with AWS credentials (see root `.env.example`)

## Key Metrics Covered

| Metric | Description |
|--------|-------------|
| **Accuracy** | Response correctness (LLM-as-judge, human eval) |
| **Cost** | Token costs (input, output, cache) |
| **Latency** | TTFT, TTLT, generation time |
| **Throughput** | TPM, RPM |
