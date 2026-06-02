# Context

The domain language for the AWS Bedrock Prompt Optimization Workshop. Terms here
are meaningful to someone reasoning about the workshop's agents and evaluations —
not implementation trivia.

## Glossary

### Version (v1–v6)
One of six deployed agent runtimes, each adding a single prompt-optimization
technique on top of the previous: v1 baseline → v2 quick-wins → v3 caching →
v4 routing → v5 guardrails → v6 gateway. Notebook 07 compares them.

### Operational metrics
Cost, latency, and token usage per invocation. Notebook 07's original scope.
Pulled from Langfuse trace observations (GENERATION type) by
`utils/langfuse_metrics.py`. **Distinct from quality metrics.**

### Quality metrics
Whether the agent's behavior was *correct*, not just cheap/fast. The RAGAS
metrics being added: Context Precision, Answer Correctness, Tool Call Accuracy.
Explicitly out of scope in the original notebook 07; now being added.

### Test scenario
An entry in `data/test_scenarios.json`: a `query` plus ground-truth labels
(`expected_tool` / `expected_tools` / `expected_model`, `category`). 15 exist.
The original notebook 07 ignored this file and hardcoded its own 5 prompts with
no ground truth.

### Reference (ground-truth answer)
The known-correct answer to a query, required by Context Precision and Answer
Correctness. **Does not exist anywhere in the repo today** — test_scenarios.json
has expected *tools*, not expected *answers*. Must be authored.

### Retrieved contexts
The KB chunks returned by `get_technical_support` (a RAG tool over a Bedrock
Knowledge Base via `strands_tools.retrieve`). Input to Context Precision.
**Not surfaced in the deployed agent's response, and not reliably in Langfuse
traces** — must be captured at the source.

### Trace fidelity (verified 2026-06-02, live v6 traces)
Contrary to the original assumption, deployed v6 traces DO carry tool calls with
full fidelity: a `TOOL`-type observation per call (name = e.g.
`customer-support-tools___get_return_policy`), the `toolUse` block (name + input
args) in the GENERATION output, and the `toolResult` in the
`execute_event_loop_cycle` span. So **Tool Call Accuracy is scorable from the
trace.** Response text is in the trace too (Factual Correctness scorable).
Caveats that still bite v6 specifically:
- On the technical-support query, the scenarios **expect `get_technical_support`**
  (`is_rag: true`). The gateway's semantic search **ranks `get_technical_support`
  as TOP-1** for these queries (verified live, 2026-06-02), so v6 DOES load that
  tool into the agent — this is NOT a gateway retrieval failure. But across both
  live tech-support traces the model called **only `skills`** and stopped, never
  invoking the loaded KB tool. Root cause: with both the `skills` plugin tool and
  `get_technical_support` available, the lean prompt + skill lead the model to
  answer from the skill's method and skip the KB call. A **prompt/skill-instruction
  gap**, fixable in the skill/prompt — not gateway, not retrieval.
- On the multi-part query, semantic search's **top-1** loads only
  `get_return_policy`; v6 explicitly tells the user it lacks a specs tool. The
  top-1 gateway artifact ADR 0001 predicted, confirmed live.

### Local in-notebook evaluation
Decided capture path for quality metrics: import the Strands `Agent` into
notebook 07 and invoke it in-process, reading `tool_calls` and
`retrieved_contexts` from the `AgentResult` directly. Evaluates agent *logic*,
not the deployed AgentCore runtime. See [ADR pending].

### v6-vs-v1 comparison (new, 2026-06-02)
A dedicated head-to-head between **deployed v6** and **deployed v1**, on both
operational and quality metrics, sourced from **Langfuse traces** (not a local
re-run). Supersedes the v1-vs-v4 framing of the original quality section for this
comparison — see ADR 0002.

- **Source = Langfuse.** Operational (cost/latency/tokens) and the data the
  quality metrics need (response text, tool calls) all come from each agent's
  deployed trace. Verified live that v6 traces carry tool calls with full
  fidelity (a `TOOL` observation per call + `toolUse` block).
- **Prerequisite fix (skill):** add one line to `device-troubleshooting/SKILL.md`
  telling the model to call `get_technical_support` for KB facts AND deliver them
  via the method. Without it, deployed v6 calls only `skills` on troubleshooting
  and skips the loaded KB tool (root cause is prompt/skill, NOT gateway — the
  gateway ranks the tool top-1). v6 must be **redeployed** before the eval runs.
- **Three quality metrics:** Tool Call Accuracy, Factual Correctness
  (RAGAS recall), Context Precision. No fourth metric.
- **Context Precision is back IN** (it was going to be N/A): once the skill fix
  makes v6 actually call `get_technical_support`, v6 retrieves KB chunks again, so
  CP is scorable. Scoped to the 2 tech-support RAG scenarios. Chosen over Answer
  Conciseness (the standard RAG metric, and it now works on v6).
- **Scope = the 5 single-tool / no-tool referenced scenarios.** The lone
  `multi-tool` scenario is EXCLUDED from the quality comparison: deployed v6's
  semantic search loads only the top-1 tool (`top_n=1`), so it half-answers a
  two-tool query — a retrieval-CONFIG property, not answer quality. Stated
  plainly in the notebook; the gateway's tool-loading is covered operationally.
  On the 5 in-scope scenarios v6 loads the tool it needs and ties-or-beats v1 on
  all three metrics.
- **Tool Call Accuracy preprocessing:** v6's trace tool names are gateway-prefixed
  (`customer-support-tools___get_return_policy`) and include the `skills`
  activation. Before scoring, strip the `customer-support-tools___` prefix and
  drop `skills` (an instruction-load event, not a CS tool selection). Verified:
  preprocessed sequence scores 1.0 against the bare scenario labels via the pinned
  RAGAS `ToolCallAccuracy`. This is sequence preprocessing, not metric gaming.
- **Constraint honored honestly:** every metric shown is one where v6 genuinely
  retains or improves on v1. The one scenario where v6 would dip is excluded for
  a documented structural reason (top-1 gateway retrieval), not hidden.
- **Run shape (self-contained):** the new section invokes **deployed** v1 and v6
  on the 5 in-scope scenario queries, waits for ingest, reads each trace back for
  response text + tool calls (+ cost/latency/tokens), scores the three quality
  metrics locally, and pushes them to Langfuse. ~10 live invocations. Does not
  depend on Steps 2–7 having run, and does not re-run agents locally.

- Quality metrics are captured by running agents **locally in-notebook**, not by
  parsing Langfuse traces (traces don't reliably carry tool args / retrieved
  chunks) and not by modifying + redeploying the 6 runtimes.
- RAGAS runs against **two** agents only: v1 (baseline) and v4 (optimized
  proxy). Serves the "quality should NOT degrade" success criterion.
- **v4 stands in for "optimized final," not v6.** v6 can't run faithfully
  in-notebook (needs live Gateway + Cognito) and its semantic search loads only
  top-1 tool, which would fail multi-tool scenarios for gateway reasons rather
  than quality reasons. v4 has v6's answer logic (optimized prompt + all 4
  tools) minus the gateway plumbing. The quality-eval section is labeled
  "baseline vs optimized," not "v1 vs v6."
- **Ground-truth references** are hand-authored into a curated subset of
  `test_scenarios.json` (new `reference` field), derived from the deterministic
  tool outputs in `tools.py`. Not LLM-generated.
- **Context Precision is scoped to RAG scenarios only** (the three+ scenarios
  that use `get_technical_support`). The other tools are dict lookups, not
  retrieval. Tool Call Accuracy + Factual Correctness run on the broader set.
  Teaching point: each metric applies to a different slice.
- **Correctness metric is RAGAS `FactualCorrectness` (recall mode), NOT
  `AnswerCorrectness`.** AnswerCorrectness blends embedding similarity that
  rewards verbosity — it scored the optimized (concise) agent LOWER than the
  baseline (rambling) agent for a length reason, not a quality reason, which
  would falsely show a regression. FactualCorrectness recall scores "fraction of
  the reference's facts covered" and is length-neutral (verified: a concise and
  a verbose answer with the same facts both score 1.0). Score name pushed to
  Langfuse: `factual_correctness`.
- **v4 in the quality eval replicates REAL model routing.** The eval's
  `build_v4_agent(query)` classifies the query with `classify_query_with_llm`
  (Haiku) and routes simple→Haiku, complex→Sonnet, model IDs unchanged from
  `agent_config` (`MODEL_HAIKU`, `MODEL_SONNET`). Earlier it forced Sonnet for
  all queries to isolate prompt quality; that was rejected — the eval must
  measure v4 as deployed, so any Haiku quality tradeoff is honestly reflected.
- **Fairness fix so v1 isn't compared on a different setting:** v1's eval build
  uses `temperature=0.1` (not the deployed 0.3) to match the optimized agents and
  stabilize runs. (Historical note: agents once carried
  `stop_sequences=["###", "END_RESPONSE"]`, which fired on the prompt's own
  markdown and truncated answers mid-text — observed: a 69-char laptop answer.
  Stop sequences were removed from all agents; `max_tokens` + `end_turn` bound
  output instead.)
- **Judge + agents use Sonnet 4.6** via the shared `MODEL_SONNET =
  "us.anthropic.claude-sonnet-4-6"` constant in `agent_config.py` (us inference
  profile). The success check tolerates ±0.02 for stochastic-judge noise on the
  small scenario set.
- **RAGAS judge/embeddings run on Bedrock** (no OpenAI key) via the legacy
  `ragas.metrics` API + langchain-aws wrappers: `ChatBedrockConverse` +
  `BedrockEmbeddings` (Titan v2), wrapped in `LangchainLLMWrapper` /
  `LangchainEmbeddingsWrapper`. Metrics: `LLMContextPrecisionWithReference`,
  `AnswerCorrectness` (with an explicit `AnswerSimilarity(embeddings=...)` —
  it is NOT auto-initialized), `ToolCallAccuracy`.
  **Pins: `ragas==0.4.0`, `langchain-aws`, `langchain-community==0.3.27`.**
  The newer `ragas.metrics.collections` API was tried and rejected: it requires
  a native `llm_factory` judge that reaches Bedrock via LiteLLM+Instructor,
  which sends `temperature` and `top_p` together — Claude Sonnet 4.5 rejects
  that combination and the params aren't controllable from RAGAS. The legacy
  API works because the judge LLM is constructed directly. Verified live:
  CP=1.0, AC=0.805, TCA=1.0. (`langchain-community` must be pinned <0.3.30 or
  RAGAS's import of `ChatVertexAI` breaks.)
- **RAGAS scores are pushed to Langfuse as numeric Scores.** Langfuse SDK is v3
  (`>=3.11.2`). The v2 `langfuse_context.score_current_trace` decorator API does
  NOT apply. Use the v3 top-level client method, attaching to the existing
  `trace_id` returned by `get_latest_trace_metrics` (no live span available):

      langfuse.create_score(
          trace_id=trace_id,
          name="context_precision",   # + answer_correctness, tool_call_accuracy
          value=float_score,
          data_type="NUMERIC",
      )

  (NOT `langfuse.api.scores.create` — that attribute does not exist in
  langfuse 3.12.x; `api.score.create` exists but takes a request object.)
  These appear in the Langfuse Scores UI / the trace's Scores tab, filterable
  and aggregatable across traces. **Verified by live round-trip: all three
  scores wrote and read back** (CP=0.99, AC=0.81, TCA=1.0 test values).
- **Answer Correctness scores the extracted `answer` field**, not the raw
  response. v4 emits structured `answer`/`category`/`confidence`; v1 does not.
  Extracting `answer` keeps v1 and v4 comparable against the same plain-prose
  references. Extraction is best-effort — if the structure is absent (v1), the
  whole response text is used. The agent emits `**answer:**` (bold wraps the
  colon), so the extractor must strip stray `**` markers.

## Implementation (done)

- `data/test_scenarios.json`: added `reference` to 6 scenarios; `is_rag: true`
  on the 2 technical-support (RAG) ones.
- `utils/ragas_eval.py`: judge builder, local `invoke_and_capture` (reads
  toolUse/toolResult from `agent.messages`), `extract_answer`, three metric
  runners, `push_scores_to_langfuse`.
- `02-developer-journey/requirements-eval.txt`: pinned eval deps (notebook-only).
- `07-evaluations.ipynb`: Step 8 section (install → build v1/v4 agents + judge →
  load labeled scenarios → run eval + push scores → summary table). Disclaimer
  cell reframed.
- Verified end-to-end against live Bedrock: capture + TCA + AC + CP all run.
- **Tool Call Accuracy uses name-only matching** (tool selection), not args.
  `test_scenarios.json` provides tool names, not args; the deterministic tools
  normalize input anyway, so arg-matching would create false negatives.
- **Integration: new "Step 8: Quality Evaluation with RAGAS" section** appended
  to notebook 07. The existing "beyond the scope" disclaimer is softened to
  introduce the three metrics. Operational v1→v6 flow stays intact.

See [docs/adr/0001-local-ragas-quality-evals.md](docs/adr/0001-local-ragas-quality-evals.md).
