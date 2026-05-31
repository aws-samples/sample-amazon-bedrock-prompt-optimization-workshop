# Optimizing Cost, Latency, and Quality on Amazon Bedrock

This hands-on workshop teaches you how to cut cost and latency in production GenAI applications on Amazon Bedrock without sacrificing quality.

> **Companion code repository.** This repo holds the runnable notebooks. The narrated walkthrough lives in [AWS Workshop Studio](https://catalog.us-east-1.prod.workshops.aws/workshops/60d21a0a-c56f-47aa-9e5d-45181cd42507/en-US) — follow it alongside these notebooks.

**Target Audience**: AI/ML Developers, Software Engineers working with agentic systems, DevOps Engineers deploying GenAI workloads

---

## Workshop Objectives

This workshop focuses on optimizing three key metrics for production GenAI applications:

| Objective | Definition | Why It Matters |
|-----------|------------|----------------|
| **Accuracy** | The quality and correctness of model outputs relative to expected results | Ensures your application delivers value to users and meets business requirements |
| **Cost** | Total expenditure on model inference, including input tokens, output tokens, and cache operations | Controls operational expenses and enables sustainable scaling |
| **Latency** | Time elapsed from request initiation to response completion | Impacts user experience and application responsiveness |

---

## Workshop Structure

Four progressive tracks. Work them in this order (the folder prefixes are clone artifacts, not the sequence):

### Part 1: Fundamentals — `01-fundamentals/`
*Estimated time: 1.25 hours*

Get your tooling and vocabulary in place: notebooks, token economics, latency metrics, observability.

| Topic | Duration | Description |
|-------|----------|-------------|
| [Jupyter Notebook 101](./01-fundamentals/00-jupyter-notebook-101.ipynb) | 15 min | Kernels, cells, shortcuts, first Bedrock call |
| [Prompts 101](./01-fundamentals/01-prompts-101.ipynb) | 30 min | Tokens, pricing, TPM/RPM, CRIS, Converse API, Bedrock Mantle |
| [Langfuse Observability](./01-fundamentals/02-langfuse-observability.ipynb) | 30 min | LLM tracing, cost tracking, prompt management with Langfuse |

### Part 2: Optimization Playbook — `02-optimization-playbook/`
*Estimated time: ~2 hours*

The sixteen cost-and-latency levers, organized into three effort tiers. One consolidated notebook per tier; start LOW, earn HIGH.

| Notebook | Duration | Levers |
|----------|----------|--------|
| [LOW effort](./02-optimization-playbook/01-low-effort.ipynb) | 45 min | Model selection, prompt design, parameter tuning, prompt caching, prompt engineering tricks (+ managed APO), adaptive thinking |
| [MEDIUM effort](./02-optimization-playbook/02-medium-effort.ipynb) | 60 min | LLM routing, Guardrails, RAG, prompt compression, conversation & memory (incl. AgentCore Memory), batch inference |
| [HIGH effort](./02-optimization-playbook/03-high-effort.ipynb) | 60 min | Sub-agent delegation (Claude Agent SDK), tool search via MCP Gateway (+ concept: harness engineering, GEPA/DSPy) |

### Part 3: Developer Journey — `03-developer-journey/`
*Estimated time: 3.5 hours*

Apply the levers to one production-ready TechMart Electronics customer support agent through 7 progressive labs, each improving the last.

| Topic | Duration | Description |
|-------|----------|-------------|
| [Baseline Agent](./03-developer-journey/01-baseline-agent.ipynb) | 20 min | Build unoptimized baseline agent, establish metrics |
| [Quick Wins](./03-developer-journey/02-quick-wins.ipynb) | 20 min | Concise prompts, max_tokens, stop_sequences |
| [Prompt Caching](./03-developer-journey/03-prompt-caching.ipynb) | 30 min | System prompt and tool definition caching |
| [LLM Routing](./03-developer-journey/04-llm-routing.ipynb) | 30 min | Route queries to appropriate models by complexity |
| [Guardrails](./03-developer-journey/05-guardrails.ipynb) | 30 min | Bedrock Guardrails for topic/content filtering |
| [AgentCore Gateway](./03-developer-journey/06-agentcore-gateway.ipynb) | 45 min | Semantic tool search, centralized tool management |
| [Evaluations](./03-developer-journey/07-evaluations.ipynb) | 30 min | Systematic evaluation across all agent versions |

> **Note**: Part 3 requires infrastructure deployment. See [03-developer-journey/README.md](./03-developer-journey/README.md) for setup instructions.

### Part 4: Deep Dives — `04-deep-dive-topics/`
*Estimated time: 2.25 hours*

Two standalone deep-dive topics: deep-dive caching patterns, and the production prompt lifecycle (versioned, eval-gated, CI/CD-managed prompts).

| Topic | Duration | Description |
|-------|----------|-------------|
| [Advanced Prompt Caching](./04-deep-dive-topics/01-advanced-prompt-caching.ipynb) | 60 min | Multi-checkpoint patterns, cache strategies |
| [Production Prompt Lifecycle](./04-deep-dive-topics/02-production-prompt-lifecycle.ipynb) | 75 min | Langfuse prompt management, evaluation datasets, LLM-as-Judge, CI/CD |

---

## Prerequisites

### Required
- AWS Account with Amazon Bedrock access enabled
- Python 3.10 or higher
- Basic familiarity with Python and Jupyter notebooks

---

## Setup Instructions

### 1. Local Environment Setup

#### Option A: Using uv (Recommended - Fast!)

**Install uv** (if not already installed):

**macOS/Linux**:
```bash
# Option 1: Official installer (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Option 2: Using Homebrew
brew install uv

# Option 3: Using pipx
brew install pipx
pipx install uv
```

**Windows**:
```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Create and activate virtual environment**:
```bash
# Create virtual environment with Python 3.11
uv venv --python 3.11

# Activate virtual environment
# macOS/Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

#### Option B: Using standard pip

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# macOS/Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

### 2. Configure AWS Credentials

The workshop supports multiple credential methods:

#### Method 1: AWS CLI Credentials (Recommended)

If you already have AWS CLI configured, the notebooks will automatically use your credentials:

```bash
# No additional setup needed - boto3 uses ~/.aws/credentials
```

#### Method 2: Environment Variables (.env file)

If you prefer to use a `.env` file:

```bash
# Copy the example file
cp .env.example .env

# Edit .env and uncomment the AWS credentials
# Then add your actual credentials:
# AWS_ACCESS_KEY_ID=your-actual-access-key-id
# AWS_SECRET_ACCESS_KEY=your-actual-secret-access-key
# AWS_DEFAULT_REGION=us-east-1
```

The notebooks will automatically load credentials using `python-dotenv`.

---

### 3. Test Bedrock Connectivity

```bash
python -c "
import boto3
client = boto3.client('bedrock-runtime', region_name='us-east-1')
print('Bedrock connection successful!')
print(f'Region: {client.meta.region_name}')
"
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---