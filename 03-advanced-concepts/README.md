# Part 3: Advanced Concepts

## Overview

This section covers advanced prompt engineering techniques, complex caching patterns, and production lifecycle management for GenAI systems. You will move from foundational knowledge into research-backed optimization methods, multi-checkpoint caching strategies, and a complete prompt CI/CD pipeline using Langfuse and AWS CodePipeline.

**Duration**: ~3.5 hours

---

## Learning Objectives

After completing this section, you will:
- Apply research-backed prompt engineering techniques (CoT, Self-Refine, CoD, etc.)
- Select the right technique for your scenario based on cost/latency/quality trade-offs
- Implement multi-checkpoint caching patterns with proper static/dynamic separation
- Choose appropriate cache TTL strategies (5-minute vs 1-hour)
- Version and manage prompts in Langfuse with production labels
- Build evaluation datasets and run LLM-as-Judge scoring
- Integrate prompt evaluation into a CI/CD pipeline with quality gates

## Prerequisites

- Complete [01-basics](../01-basics/) first
- Complete [02-developer-journey](../02-developer-journey/) for Langfuse setup (required for notebook 03)
- AWS Account with Amazon Bedrock access
- Python 3.10+

## Notebooks

| Notebook | Duration | Description |
|----------|----------|-------------|
| [01-advanced-prompt-engineering](./01-advanced-prompt-engineering.ipynb) | 60 min | Technique categories, optimization workflows |
| [02-advanced-prompt-caching](./02-advanced-prompt-caching.ipynb) | 60 min | Multi-checkpoint patterns, cache strategies |
| [03-production-prompt-lifecycle](./03-production-prompt-lifecycle.ipynb) | 75 min | Langfuse prompt management, evaluation datasets, LLM-as-Judge, CI/CD |

---

### **01-advanced-prompt-engineering.ipynb** (60 min)

Advanced prompt engineering techniques beyond the basics, covering reasoning enhancement, iterative refinement, efficiency optimizations, and systematic prompt improvement workflows.

**What you'll learn**:
- Chain-of-Thought (CoT) and Extended Thinking for reasoning tasks
- Self-Refine (generate/critique/refine) and Chain-of-Verification for quality-critical output
- Chain-of-Draft for production-efficient reasoning (50-70% token reduction)
- Verbalized Sampling for diverse output generation
- Prompt optimization workflows: manual iteration, LLM-assisted exploration, automated evaluation, Bedrock OptimizePrompt API

---

### **02-advanced-prompt-caching.ipynb** (60 min)

Multi-checkpoint caching patterns and best practices for Amazon Bedrock, including cache invalidation, TTL strategies, and performance monitoring.

**What you'll learn**:
- Cacheable content types (tools, system, messages) and assembly order
- Multi-checkpoint placement strategies (single, two, three checkpoints)
- Cumulative token counting and cache invalidation scenarios
- Static/dynamic separation to prevent cache thrashing
- 5-minute vs 1-hour TTL selection and mixed TTL strategies
- Cache performance monitoring and ROI analysis

---

### **03-production-prompt-lifecycle.ipynb** (75 min)

End-to-end production prompt lifecycle management using Langfuse for versioning, systematic evaluation with datasets, automated quality scoring with LLM-as-Judge, and CI/CD integration with AWS CodePipeline.

**What you'll learn**:
- Prompt versioning in Langfuse with production labels and trace linkage
- Creating and managing evaluation datasets (12-item customer support suite)
- Running experiments across prompt versions with side-by-side comparison
- Keyword-based and LLM-as-Judge scoring (using Haiku as a cost-effective judge)
- Quality gate thresholds (avg score >= 0.7, pass rate >= 80%)
- CI/CD pipeline architecture: CodePipeline, CodeBuild buildspec, Lambda test action
- CloudFormation template for deploying the full pipeline
- Alternative GitHub Actions workflow reference

---

## Files Structure

```
03-advanced-concepts/
├── README.md
├── 01-advanced-prompt-engineering.ipynb
├── 02-advanced-prompt-caching.ipynb
├── 03-production-prompt-lifecycle.ipynb
├── scripts/                          # CI/CD pipeline scripts
│   ├── evaluate_prompt.py            # Evaluation runner (Langfuse + Bedrock)
│   └── check_quality_gate.py         # Quality gate checker (no dependencies)
├── cloudformation/
│   └── prompt-pipeline.yaml          # CodePipeline CloudFormation template
├── data/                             # Sample data and evaluation datasets
│   ├── analyst_system_prompt_v1.txt  # Analyst prompt v1 (caching demos)
│   ├── analyst_system_prompt_v2.txt  # Analyst prompt v2 (invalidation demos)
│   ├── analyst_tools.json            # Analyst tool definitions
│   ├── company_policies.txt          # Company policies (system prompt demos)
│   ├── customer-support-eval.json    # 12-item evaluation dataset (notebook 03)
│   ├── intent_classification_eval.json # 40-item intent eval (notebook 01)
│   ├── technical_documentation.txt   # Technical docs (caching demos)
│   └── weather_tools.json            # Weather tool definitions
└── utils/                            # Shared helper modules
    ├── __init__.py
    └── cache_metrics.py              # Cache metric extraction and ROI analysis
```

---

## Next Steps

- Apply techniques to your production use cases
- Monitor cache hit rates and optimize checkpoint placement
- Set up CI/CD pipelines for automated prompt evaluation
- Build evaluation datasets from production failures and edge cases
