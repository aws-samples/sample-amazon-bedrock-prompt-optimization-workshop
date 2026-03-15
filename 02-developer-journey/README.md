# Prompt Optimization Workshop - Developer Journey

## Overview

Build a production customer support agent for **TechMart Electronics** while applying progressive prompt optimization techniques to reduce cost and latency without sacrificing quality.

**Use Case**: Customer Support AI Assistant
- Product information and specifications
- Return policies and procedures
- Technical troubleshooting via Knowledge Base
- Web search for current information

**Duration**: ~3 hours

---

## Prerequisites (Admin Setup)

### Infrastructure Deployment

The Makefile reads `.env` from the project root. Ensure it's configured:

```bash
# From project root
cp .env.example .env
# Edit .env with your AWS_PROFILE and other settings
```

Deploy the infrastructure using Make (from `02-developer-journey/`):

```bash
# Check environment configuration
make check-env

# Deploy all infrastructure (Lambda, S3, Cognito)
make deploy-all

# Or deploy individually:
make deploy-infra    # Lambda, S3, IAM roles
make deploy-cognito  # Cognito User Pool for Gateway auth
```

The infrastructure stack automatically creates:
- Bedrock Knowledge Base with S3 Vectors
- Sample technical documentation uploaded to S3
- Knowledge Base ID stored in SSM Parameter Store

Check stack outputs for Knowledge Base ID:
```bash
make status
```

### Other Infrastructure Commands

```bash
make status          # Check stack status
make delete-all      # Delete all stacks (cleanup)
make help            # Show all commands
```

---

## Workshop Journey

```
01 Baseline → 02 Quick Wins → 03 Caching → 04 Routing → 05 Guardrails → 06 Gateway → 07 Evaluations
```

---

## Notebooks

### **01-baseline-agent.ipynb** (20 min)

Create an unoptimized baseline agent to establish metrics.

**What you'll learn**:
- Setting up a Strands agent with Bedrock
- Integrating Langfuse for observability
- Measuring baseline cost and latency

---

### **02-quick-wins.ipynb** (20 min)

Apply quick optimization techniques for immediate gains.

**Optimizations**:
- Concise system prompt (~60% token reduction)
- `max_tokens` limit (bounded output)
- `stop_sequences` (early termination)

---

### **03-prompt-caching.ipynb** (30 min)

Implement Bedrock prompt caching for repeated content.

**Optimizations**:
- System prompt caching with `cachePoint`
- Tool definition caching with `cache_tools="default"`
- 90% cost reduction on cached tokens

---

### **04-llm-routing.ipynb** (30 min)

Route queries to appropriate models based on complexity.

**Optimizations**:
- Haiku for simple queries (12x cheaper input tokens)
- Sonnet for complex queries (maintained quality)
- Complexity classification

---

### **05-guardrails.ipynb** (30 min)

Add Bedrock Guardrails to filter off-topic queries.

**Optimizations**:
- Topic filters (competitor questions, investment advice)
- Content filters (violence, hate speech)
- Zero tokens on blocked queries

---

### **06-agentcore-gateway.ipynb** (45 min)

Integrate AgentCore Gateway for centralized tool management.

**Optimizations**:
- Semantic tool search (load only relevant tools)
- Reduced context size (up to 75% fewer tool tokens)
- MCP-compatible tool sharing

---

### **07-evaluations.ipynb** (30 min)

Run comprehensive evaluations across all versions.

**What you'll learn**:
- Systematic evaluation across agent versions
- Side-by-side metric comparison
- Quality validation

---

## Environment Setup

1. Install dependencies (from project root):
   ```bash
   uv pip install -r requirements.txt
   ```

2. Configure environment (from project root):
   ```bash
   cp .env.example .env
   # Edit .env with your AWS_PROFILE and other settings
   ```

3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

4. Start Jupyter:
   ```bash
   jupyter lab
   ```

---

## Success Criteria

Your optimized agent should achieve:

- [ ] 50%+ cost reduction vs baseline
- [ ] 30%+ latency improvement
- [ ] 95%+ success rate maintained
- [ ] All test scenarios passing

---

## Files Structure

```
02-developer-journey/
├── Makefile                  # Infrastructure deployment (reads ../.env)
├── requirements.txt          # Python dependencies
├── pyproject.toml            # UV project config
├── 01-baseline-agent.ipynb   # Notebook 1
├── 02-quick-wins.ipynb       # Notebook 2
├── 03-prompt-caching.ipynb   # Notebook 3
├── 04-llm-routing.ipynb      # Notebook 4
├── 05-guardrails.ipynb       # Notebook 5
├── 06-agentcore-gateway.ipynb # Notebook 6
├── 07-evaluations.ipynb      # Notebook 7
├── agents/                   # Agent implementations
│   ├── v1_baseline.py
│   ├── v2_quick_wins.py
│   ├── v3_caching.py
│   ├── v4_routing.py
│   ├── v5_guardrails.py
│   └── v6_gateway.py
├── utils/                    # Helper modules
│   ├── agent_config.py
│   ├── tools.py
│   ├── metrics.py
│   └── langfuse_setup.py
├── data/                     # Test data
│   ├── test_scenarios.json
│   ├── product_catalog.json
│   └── return_policies.json
└── prerequisite/             # CloudFormation templates
    ├── infrastructure.yaml   # Lambda, S3, KB, IAM
    ├── cognito.yaml          # Cognito User Pool for Gateway
    └── lambda/               # Lambda function code
        ├── api_spec.json     # Tool schema
        └── python/           # Lambda source code
            ├── lambda_function.py
            ├── web_search.py
            └── ddgs-layer.zip
```
