# AWS Bedrock Prompt Optimization Workshop

This hands-on workshop teaches you how to build production-grade generative AI applications with a focus on cost optimization, performance enhancement, and operational excellence.

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

This workshop is organized into progressive parts:

### Part 1: Basics - Fundamentals
*Estimated time: 1.5 hours*

Build a solid understanding of tokens, pricing, and optimization strategies.

| Topic | Duration | Description |
|-------|----------|-------------|
| [Prompts 101](./01-basics/01-prompts-101.ipynb) | 30 min | Tokens, pricing, TPM/RPM, terminology |
| [Optimization Strategy](./01-basics/02-optimization-strategy.ipynb) | 45 min | Model selection, prompt design, parameter tuning, basic caching |
| [Langfuse Observability](./01-basics/03-langfuse-observability.ipynb) | 30 min | LLM tracing, cost tracking, prompt management with Langfuse |

### Part 2: Developer Journey
*Estimated time: 3 hours*

Build a production customer support agent while applying progressive optimization techniques.

| Topic | Duration | Description |
|-------|----------|-------------|
| [Baseline Agent](./02-developer-journey/01-baseline-agent.ipynb) | 20 min | Build unoptimized baseline agent, establish metrics |
| [Quick Wins](./02-developer-journey/02-quick-wins.ipynb) | 20 min | Concise prompts, max_tokens, stop_sequences |
| [Prompt Caching](./02-developer-journey/03-prompt-caching.ipynb) | 30 min | System prompt and tool definition caching |
| [LLM Routing](./02-developer-journey/04-llm-routing.ipynb) | 30 min | Route queries to appropriate models by complexity |
| [Guardrails](./02-developer-journey/05-guardrails.ipynb) | 30 min | Bedrock Guardrails for topic/content filtering |
| [AgentCore Gateway](./02-developer-journey/06-agentcore-gateway.ipynb) | 45 min | Semantic tool search, centralized tool management |
| [Evaluations](./02-developer-journey/07-evaluations.ipynb) | 30 min | Systematic evaluation across all agent versions |

> **Note**: Part 2 requires infrastructure deployment. See [02-developer-journey/README.md](./02-developer-journey/README.md) for setup instructions.

### Part 3: Advanced Concepts
*Estimated time: 3-3.5 hours*

Advanced prompt engineering techniques, complex caching patterns, and production prompt lifecycle management.

| Topic | Duration | Description |
|-------|----------|-------------|
| [Advanced Prompt Engineering](./03-advanced-concepts/01-advanced-prompt-engineering.ipynb) | 60 min | CoT, Self-Critical, CoD, technique selection etc. |
| [Advanced Prompt Caching](./03-advanced-concepts/02-advanced-prompt-caching.ipynb) | 60 min | Multi-checkpoint patterns, cache strategies etc. |
| [Production Prompt Lifecycle](./03-advanced-concepts/03-production-prompt-lifecycle.ipynb) | 75 min | Langfuse prompt management, evaluation datasets, LLM-as-Judge, CI/CD |

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