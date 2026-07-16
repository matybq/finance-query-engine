# Decisions

## ADR-001 — LangGraph over a linear chain

- **Date:** 2026-07-06
- **Status:** accepted
- **Context:** The system needs routing and a corrective retrieval loop, not just a single pass from question to answer.
- **Decision:** Use LangGraph as the orchestration layer.
- **Rationale:** The flow has real decision points: route the question, grade retrieved context, and rewrite when evidence is weak. That is a graph, not a straight line.
- **Alternatives considered / rejected:**
  - **Linear chain:** more honest for plain search-and-answer, but it would hide the self-correction path this project is built to demonstrate.
  - **LangChain chain only:** simpler, but not a good match for the router + loop structure.

## ADR-002 — Hybrid retrieval (dense + BM25 + RRF) over dense-only

- **Date:** 2026-07-06
- **Status:** accepted
- **Context:** Financial documents contain both paraphrased concepts and exact tokens like tickers, figures, and identifiers.
- **Decision:** Combine dense retrieval with BM25 and fuse the rankings with RRF.
- **Rationale:** Dense search handles semantic similarity; BM25 handles exact match. Retrieval is where RAG usually fails, so this is where the project should invest effort.
- **Alternatives considered / rejected:**
  - **Dense-only:** too fragile for exact terms and numerical queries.
  - **BM25-only:** too weak on paraphrase and semantic queries.

## ADR-003 — Chroma over Qdrant, pgvector, or Pinecone

- **Date:** 2026-07-06
- **Status:** accepted
- **Context:** Phase 0/1 needs a local, low-friction vector store.
- **Decision:** Use Chroma for the core implementation.
- **Rationale:** It is local, simple, and requires no extra infrastructure. That is the fastest path for a two-week core.
- **Alternatives considered / rejected:**
  - **Qdrant:** solid option, but adds extra service overhead for the current scope.
  - **pgvector:** useful later if Postgres is already the system of record.
  - **Pinecone:** production-friendly, but unnecessary infrastructure for the first version.

## ADR-004 — OpenRouter as LLM provider

- **Date:** 2026-07-06
- **Status:** accepted
- **Context:** The project needs a practical LLM surface that can change models without reworking the app.
- **Decision:** Use OpenRouter for the main LLM provider, and use OpenAI directly for embeddings with `text-embedding-3-small`.
- **Rationale:** OpenRouter gives one API over many models, which is pragmatic for the timeline. OpenAI embeddings stay separate because OpenRouter does not proxy the embeddings endpoint, and the selected model is cheap, compact, and sufficient for this corpus. This choice should be re-evaluated with RAGAS in Phase 5.
- **Alternatives considered / rejected:**
  - **Direct single-model integration:** works, but is less flexible.
  - **Local/open-source models only:** promising for later, but slower to ship for the core.

## ADR-005 — Single-file ADR log over one-file-per-decision

- **Date:** 2026-07-06
- **Status:** accepted
- **Context:** This is a solo project with a short timeline, and the docs are also interview prep.
- **Decision:** Keep all ADRs in one file.
- **Rationale:** A single scannable log is easier to maintain and easier to present than a directory of tiny decision files.
- **Alternatives considered / rejected:**
  - **One file per decision:** more modular, but too much overhead for Phase 0.

## ADR-006 — Grounding with citations + “I don't know” guardrail

- **Date:** 2026-07-06
- **Status:** accepted
- **Context:** Finance is a high-stakes domain where unsupported claims are unacceptable.
- **Decision:** Require citations and an explicit refusal path when evidence is insufficient.
- **Rationale:** A fabricated answer is worse than no answer. The system must prefer honesty over fluency.
- **Alternatives considered / rejected:**
  - **Best-effort freeform answers:** too risky for the domain.
  - **Soft warning only:** not strong enough as a guardrail.

## ADR-007 — English-only repo

- **Date:** 2026-07-06
- **Status:** accepted
- **Context:** The portfolio audience is English-speaking, but the original implementation plan is in Spanish.
- **Decision:** Write the project repo artifacts in English.
- **Rationale:** English makes the project easier to read in interviews and for external reviewers. The Spanish planning doc stays as a historical source artifact.
- **Alternatives considered / rejected:**
  - **Bilingual repo:** adds noise without helping the core audience.

## ADR-008 — Phase-gated dependencies

- **Date:** 2026-07-06
- **Status:** accepted
- **Context:** The dependency file should reflect what the current phase actually needs.
- **Decision:** Add dependencies only when the phase needs them.
- **Rationale:** A minimal `pyproject.toml` keeps the project honest and makes the build story easier to follow.
- **Alternatives considered / rejected:**
  - **Install everything up front:** hides growth, increases maintenance, and blurs the phase boundaries.

## ADR-009 — Chunking 1000 chars / 150 overlap with RecursiveCharacterTextSplitter

- **Date:** 2026-07-06
- **Status:** accepted
- **Context:** Phase 1 needs a first chunking policy for a small plain-text corpus.
- **Decision:** Use `RecursiveCharacterTextSplitter` with `chunk_size=1000` and `chunk_overlap=150`.
- **Rationale:** This is a first deliberate value, roughly ~250 tokens per chunk: enough local context without prompt noise. Treat it as provisional and tune it later with RAGAS evals in Phase 5, not by eye.
- **Alternatives considered / rejected:**
  - **Larger chunks:** more context, but more prompt noise and weaker retrieval precision.
  - **Smaller chunks:** cleaner retrieval, but too little local context for grounded answers.

## ADR-010 — Plain-text loading via pathlib, no document-loader dependency

- **Date:** 2026-07-06
- **Status:** accepted
- **Context:** The corpus is already extracted into `.txt` files, and Phase 1 only needs to read three documents.
- **Decision:** Load files directly with `pathlib` and build `Document` objects in code; do not add a `langchain-community` text loader dependency.
- **Rationale:** Plain text does not need an extra loader layer, and the phase-gated dependency policy says not to add `pypdf` until PDFs enter the corpus.
- **Alternatives considered / rejected:**
  - **TextLoader or similar loaders:** extra dependency surface with no benefit for the current corpus.
  - **Keeping `pypdf` installed now:** premature while the corpus contains no PDFs.

## ADR-011 — Recreate-on-ingest with safe swap

- **Date:** 2026-07-06
- **Status:** accepted
- **Context:** Ingestion currently rebuilds the whole index each run, and the corpus is only three files.
- **Decision:** Recreate the index on every ingest, build into a temporary directory, and swap it into place only after the build succeeds.
- **Rationale:** Incremental indexing is overengineering for this scope, and the temp-dir swap prevents a failed embedding run from destroying the existing index.
- **Alternatives considered / rejected:**
  - **Incremental updates:** more complex than needed for a three-file corpus.
  - **Writing directly to the live persist dir:** risks losing a working index on failure.

## ADR-012 — Dense retrieval k=4 default

- **Date:** 2026-07-06
- **Status:** accepted
- **Context:** Basic grounded generation needs a default dense retrieval width.
- **Decision:** Use `k=4` as the default dense retrieval count, while keeping it configurable per call.
- **Rationale:** This is a starting value only and should be revisited with evals.
- **Alternatives considered / rejected:**
  - **Lower k:** may not supply enough evidence.
  - **Higher k:** may add noise and stretch the prompt unnecessarily.

## ADR-013 — LLM provider switch (`llm_provider`: openrouter | openai)

- **Date:** 2026-07-06
- **Status:** accepted
- **Context:** OpenRouter remains the design default from ADR-004, but direct OpenAI access is useful when OpenRouter credit is unavailable.
- **Decision:** Keep OpenRouter as the default provider and add `llm_provider` with `openrouter | openai` so the generation backend can switch without changing the rest of the app.
- **Rationale:** This preserves the original provider strategy while allowing the current active OpenAI run with `gpt-4o-mini`; the model-name convention differs by provider, so the configuration needs to stay explicit about that boundary.
- **Alternatives considered / rejected:**
  - **Separate code paths:** unnecessary duplication.
  - **Dropping OpenRouter support:** would undo the original provider choice.

## ADR-014 — Umbrella `langchain` package dropped

- **Date:** 2026-07-06
- **Status:** accepted
- **Context:** No code imports the umbrella `langchain` package directly; the project uses the split packages instead.
- **Decision:** Do not depend on `langchain`; keep only `langchain-openai`, `langchain-chroma`, and `langchain-text-splitters` (plus transitive `langchain-core`).
- **Rationale:** This keeps the dependency tree honest under the phase-gated policy and avoids accidental LangGraph-related transitive baggage.
- **Alternatives considered / rejected:**
  - **Adding the umbrella package:** unnecessary overlap with the direct split packages.
  - **Retaining extra transitive deps early:** obscures what Phase 1 actually requires.

## ADR-015 — BM25 via `rank-bm25` + JSONL corpus, not `langchain-community` BM25Retriever

- **Date:** 2026-07-07
- **Status:** accepted
- **Context:** Sparse retrieval needs a transparent corpus artifact and a simple rebuild path.
- **Decision:** Persist the sparse corpus as `chunks.jsonl` with `chunk_id`, `text`, `source`, and `section`; rebuild BM25 in memory at load with `rank-bm25`, using the same simple regex-lowercase tokenizer for both corpus and queries.
- **Rationale:** The dependency stays tiny and stable, the fusion logic remains the thing we are learning from, and JSONL keeps the sparse corpus inspectable. Rebuilding BM25 from the current small corpus is cheap enough, so there is no need for a pickled index or opaque persisted structure.
- **Alternatives considered / rejected:**
  - **`langchain-community` BM25Retriever:** adds an extra abstraction without improving the learning goal.
  - **Pickled sparse index:** faster to load, but opaque and less trustworthy for debugging.

## ADR-016 — Stable chunk IDs (`source:index`) across Chroma, metadata, and JSONL

- **Date:** 2026-07-07
- **Status:** accepted
- **Context:** Reciprocal Rank Fusion needs stable identity to merge dense and sparse rankings.
- **Decision:** Assign chunk IDs once during ingestion as `source:index`, then carry the same ID through Chroma, document metadata, and `chunks.jsonl`.
- **Rationale:** A fused ranking only works when the same chunk can be recognized across both retrieval paths. Stable IDs also make debugging and cross-checking easier.
- **Alternatives considered / rejected:**
  - **Regenerate IDs at load time:** breaks identity across indexes.
  - **Position-only IDs:** too fragile if the corpus changes.

## ADR-017 — Weighted RRF (dense 1.0 / sparse 0.2, sparse candidates 10, RRF_K=60)

- **Date:** 2026-07-07
- **Status:** accepted
- **Context:** Equal-weight fusion let BM25 noise bury the best dense chunk on broad semantic queries.
- **Decision:** Use weighted RRF with `dense=1.0`, `sparse=0.2`, `SPARSE_CANDIDATES=10`, and `RRF_K=60`.
- **Rationale:** BM25's role is exact-term rescue, not an equal vote on broad paraphrases. The weighting keeps dense retrieval dominant for semantic queries while still letting sparse hits recover exact-term evidence. This is provisional pending Phase 5 RAGAS.
- **Alternatives considered / rejected:**
  - **Equal-weight RRF:** too noisy for broad questions.
  - **Sparse-heavy fusion:** would overfit keyword matches and hurt semantic recall.

## ADR-018 — Generation `k` stays 4; `k=6` tried and reverted

- **Date:** 2026-07-07
- **Status:** accepted
- **Context:** Increasing the context window to catch lower-ranked chunks can also drag in misleading evidence.
- **Decision:** Keep generation retrieval width at `k=4`.
- **Rationale:** Raising `k` to 6 introduced a real hallucination on an out-of-corpus question, showing that more context does not automatically improve grounding. Recall problems should be fixed in retrieval, not by stuffing more chunks into generation.
- **Alternatives considered / rejected:**
  - **`k=6`:** reverted after the hallucination regression.
  - **Even larger `k`:** would likely add more noise than signal.

## ADR-019 — Known limitation accepted: broad paraphrase queries may refuse

- **Date:** 2026-07-07
- **Status:** accepted
- **Context:** Some broad paraphrase questions still fail to surface the right chunk early enough in the fused ranking.
- **Decision:** Accept that queries like “How does Airbnb make money?” may still return an explicit refusal for now.
- **Rationale:** We are not patching this with query hacks. The planned Phase 3 rewrite loop — grade, rewrite, retry — is the intended fix for this failure mode.
- **Alternatives considered / rejected:**
  - **Hardcoded query expansion:** too brittle and easy to hide instead of solving retrieval.
  - **Manual prompt hacks:** would mask a retrieval problem rather than fixing it.

## ADR-020 — Ingestion artifact consistency

- **Date:** 2026-07-07
- **Status:** accepted
- **Context:** Dense and sparse artifacts must stay aligned after every ingest.
- **Decision:** Write `chunks.jsonl` atomically immediately before swapping in the new Chroma directory, then check that the JSONL record count matches the number of in-memory chunks.
- **Rationale:** If dense and sparse corpora drift, retrieval becomes untrustworthy. Atomic writes plus a loud post-ingest count check make that drift fail fast instead of silently corrupting results.
- **Alternatives considered / rejected:**
  - **Non-atomic writes:** risk partially written sparse corpora.
  - **Silent drift:** too hard to diagnose later.

## ADR-021 — Explicit StateGraph over a ReAct-style tool-calling agent

- **Date:** 2026-07-08
- **Status:** accepted
- **Context:** The agent has two known decision points: routing and self-correction.
- **Decision:** Use an explicit StateGraph with deterministic conditional edges instead of a free-form tool-calling agent.
- **Rationale:** Deterministic edges map cleanly to the two real decisions in this workflow, making the system easier to reason about, test, and defend. Free-form tool choice adds nondeterminism without adding value here.
- **Alternatives considered / rejected:**
  - **ReAct-style tool-calling agent:** considered, but too open-ended for a flow with fixed decision points.

## ADR-022 — Router with three routes; `direct` restricted to meta/greetings

- **Date:** 2026-07-08
- **Status:** accepted
- **Context:** The router only needs three outcomes: retrieve, direct, or out_of_scope.
- **Decision:** Keep `direct` for meta/greetings only; factual corpus questions always go through retrieval; `out_of_scope` returns a fixed refusal with no LLM call.
- **Rationale:** Retrieval is the normal path for factual questions, and the no-LLM refusal is the cheapest and safest handling for out-of-corpus queries.
- **Alternatives considered / rejected:**
  - **Letting `direct` answer factual questions:** would bypass retrieval and weaken grounding.
  - **LLM-generated out-of-scope refusals:** unnecessary cost and more room for variation.

## ADR-023 — Rewrite loop capped at `MAX_REWRITES=2`, then structural refusal

- **Date:** 2026-07-08
- **Status:** accepted
- **Context:** Query rewrites can improve recall, but unbounded retries can also spin.
- **Decision:** Allow at most two rewrites; if grading still says the evidence is insufficient after the cap, route to `insufficient_evidence` instead of generation.
- **Rationale:** The graph should not ask the LLM to generate when its own structured state says the evidence is insufficient. This is cheaper and safer than generating and then trying to detect whether the answer is a refusal.
- **Alternatives considered / rejected:**
  - **Unlimited rewrites:** too expensive and too easy to loop.
  - **Generate anyway after repeated failures:** relies too much on prompt obedience.
  - **Text-marker refusal detection:** fragile because valid refusals can be phrased many ways.

## ADR-024 — Agent retrieval at `RETRIEVE_K=6`; plain-RAG baseline stays at `k=4`

- **Date:** 2026-07-08
- **Status:** accepted
- **Context:** Measured fused ranks put answering chunks at roughly ranks 5–7 after rewriting.
- **Decision:** Use `RETRIEVE_K=6` inside the agent, while keeping `answer()` at `k=4` for the plain-RAG baseline.
- **Rationale:** The hallucination risk that forced `k=4` in Phase 2 is now handled structurally by the router, so deeper retrieval is safe inside the agent and helps rewritten queries reach the right evidence.
- **Alternatives considered / rejected:**
  - **Keeping the agent at `k=4`:** too shallow for the rewritten query path.
  - **Raising both agent and baseline equally:** would weaken the comparison value of the baseline.

## ADR-025 — Rewrite prompt uses grader feedback and stays keyword-style

- **Date:** 2026-07-08
- **Status:** accepted
- **Context:** The first structural guardrail correctly blocked generation when the grader remained insufficient, but it exposed that generic rewrites like “Airbnb financial performance” were too broad for the “How does Airbnb make money?” gate.
- **Decision:** Keep rewrites as concise keyword-style noun phrases, but pass prior grading feedback into the rewrite node so the model can add missing factual terms from the original information need.
- **Rationale:** The rewriter should learn from why retrieval failed instead of guessing broader queries. This keeps the behavior agentic and evidence-driven without hardcoding a query rewrite table.
- **Alternatives considered / rejected:**
  - **Hardcoded query expansions:** brittle and prone to hiding retrieval problems.
  - **Question-style rewrites:** too verbose and less effective for retrieval.
  - **Ignoring grader reasons:** caused rewrites to drift toward broad financial terms rather than the missing evidence.

## ADR-026 — Structured outputs via Pydantic `with_structured_output`, verified early

- **Date:** 2026-07-08
- **Status:** accepted
- **Context:** The router, grader, and rewriter all depend on model outputs that must parse reliably.
- **Decision:** Use Pydantic models with `with_structured_output` and verify provider/model compatibility before building the graph.
- **Rationale:** Structured outputs make the graph deterministic at the boundaries, and probing compatibility early avoids late surprises when wiring the agent together.
- **Alternatives considered / rejected:**
  - **Free-form text parsing:** too fragile for routing and control flow.
  - **Assuming model compatibility:** risky to discover only after the graph is assembled.

## ADR-027 — Functional eval harness before RAGAS

- **Date:** 2026-07-10
- **Status:** accepted
- **Context:** The agent needs a fast regression suite before adding heavier metric-based evaluation.
- **Decision:** Add a lightweight JSONL-driven eval harness covering router, grounding, guardrail, factual, exact-term, and rewrite-loop behavior.
- **Rationale:** Manual functional evals are transparent, cheap to run, and catch obvious regressions in graph control flow before introducing RAGAS complexity. They are not a replacement for RAGAS; they are the first safety net.
- **Alternatives considered / rejected:**
  - **RAGAS first:** more complete, but slower to set up and harder to debug while the agent architecture is still moving.
  - **No evals until the end:** too risky because routing, rewrite, retrieval, and guardrails can regress independently.

## ADR-028 — Direct route stays LLM-generated

- **Date:** 2026-07-10
- **Status:** accepted
- **Context:** `direct` is restricted to meta/help/greetings, so it does not answer factual corpus questions.
- **Decision:** Keep `direct` LLM-generated instead of a static string, while prompting it to avoid training-data or web-knowledge language.
- **Rationale:** A generated direct response can adapt to meta questions like greetings, usage help, or capability questions while remaining low risk because factual questions are routed to retrieval.
- **Alternatives considered / rejected:**
  - **Static direct response:** cheaper and deterministic, but less flexible and less polished for conversational UX.
  - **Allow direct factual answers:** rejected because it bypasses grounding.

## ADR-029 — Agent trace streams over SSE from graph events

- **Date:** 2026-07-16
- **Status:** accepted
- **Context:** The web UI should show the agent's decisions (routing, retrieval, grading, rewrites) and stream the answer while it generates, without changing the agent architecture.
- **Decision:** Add `run_stream()` that maps LangGraph's native `updates`/`messages` stream modes to typed events (`route`, `retrieve`, `grade`, `rewrite`, `token`, `done`), served by `POST /ask/stream` as Server-Sent Events. `POST /ask` stays unchanged for the CLI and non-streaming clients.
- **Rationale:** The graph state already records every decision with its reason, so the trace is a projection of existing state — no parallel bookkeeping. Node-level events arrive during the seconds the agent takes to run, which is when feedback matters; the refusal path becomes visible instead of implied. `X-Accel-Buffering: no` keeps nginx from buffering the stream, avoiding proxy configuration coupling.
- **Alternatives considered / rejected:**
  - **WebSockets:** bidirectional transport is unnecessary for a one-way event feed and complicates the nginx/Cloudflare path.
  - **Polling a job endpoint:** simpler transport but adds server-side state and loses token-level streaming.
  - **Returning the trace only in the final response:** loses the live feedback during the run, which is the main UX value.
