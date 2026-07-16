# Architecture

## Overview

`finance-query-engine` is an agentic RAG system for asking natural-language questions over financial filings such as annual reports and 10-Ks. The goal is not just to retrieve text, but to return grounded answers with citations and a clear refusal path when the evidence is weak. Agentic orchestration is used because the flow is not purely linear: the system can route, retrieve, grade the retrieved context, rewrite the query, and retry before answering.

## Layer breakdown

### 1) Ingestion (offline)

Documents are loaded, split into chunks, embedded, and indexed in two ways:

- **Dense index:** Chroma stores embeddings for semantic search.
- **Sparse index:** BM25 stores keyword signals for exact-match retrieval.

This layer runs offline and is rebuilt when the corpus changes.

### 2) Hybrid retrieval

At query time, the system combines:

- **Dense retrieval** for paraphrases and semantic similarity.
- **BM25** for exact terms like tickers, figures, identifiers, and named entities.
- **RRF fusion** to merge both rankings into one result list.

Optional **cross-encoder reranking** is a later improvement, not part of the Phase 0/1 core.

### 3) Agent orchestration

LangGraph owns the decision flow. The agent routes into `retrieve`, `direct`, or `out_of_scope`; factual corpus questions go through retrieval, while out-of-corpus questions return a fixed refusal without an LLM call. Retrieval is not one-and-done: the agent can inspect the result, grade the context, and loop back when needed, capped after two rewrites. If evidence remains insufficient after the cap, the graph routes to `insufficient_evidence` instead of generation.

### 4) Generation

The generator answers only from retrieved context, includes citations, and prefers refusal over guessing. In finance, a fabricated answer is worse than no answer.

### 5) Evaluation

Two layers. A lightweight functional eval harness (`evals/evaluate_agent.py`) validates routing, grounding, guardrail, factual, exact-term, and rewrite-loop behavior deterministically. A report-only RAGAS suite (`evals/evaluate_ragas.py`) measures faithfulness, context recall, and factual correctness against gold answers.

### 6) Serving

A minimal FastAPI app (`src/api/app.py`) exposes `POST /ask` and `GET /health`, packaged with Docker and docker compose. VPS deployment is the remaining roadmap step.

## Agent graph

```text
question
  │
  ▼
router ──► direct ─────────────► answer
   │
   ├──► out_of_scope ──────────► refusal
   │
   └──► retrieve (k=6) ────────► grade ──► generate ──► answer + sources
                           ▲       │
                           │       └──► insufficient_evidence ──► refusal
                           │
                           └── rewrite loop (max 2 rewrites)
```

### Node-by-node

- **router**: decides whether the question needs corpus evidence, can be answered directly, or is out of scope.
  - If evidence is needed, it sends the flow to `retrieve`.
  - If the question is out of scope, it refuses early.
- **direct**: reserved for meta/greetings only; it is not a factual-answer path.
- **out_of_scope**: returns a fixed refusal without calling the LLM.
- **retrieve**: exposed as a tool; runs hybrid search and returns candidate context.
  - The agent uses `RETRIEVE_K=6`; the plain-RAG `answer()` baseline stays at `k=4`.
- **grade**: checks whether the retrieved context is relevant and sufficient.
  - If the context is weak and rewrite budget remains, the agent rewrites the query and retries retrieval.
  - If the context is still weak after the cap, the flow routes to `insufficient_evidence`.
  - If the context is good enough, the flow continues.
- **generate**: drafts the answer from retrieved evidence only and returns the final answer plus sources.
- **insufficient_evidence**: returns a deterministic refusal without calling generation.

## Data flow

```text
user question → router → retrieve → grade → generate → answer + sources
                                      └── insufficient_evidence → refusal
```

The output should always be a grounded answer plus the source documents or sections that support it.

## Stack

| Layer | Tool | Why this choice |
|---|---|---|
| Orchestration | LangGraph | The flow has real decision points and a rewrite loop, so a graph fits better than a linear chain. |
| LLM provider | OpenRouter | One API surface for many models; pragmatic for the two-week core. |
| Embeddings | OpenAI `text-embedding-3-small` | Cheap, 1536 dims, and already a good fit for this corpus; OpenRouter does not proxy embeddings. |
| Dense store | Chroma | Local, simple, and zero-infra for the core build. |
| Sparse retrieval | BM25 | Strong on exact matches that dense search can miss. |
| Rank fusion | RRF | Simple way to combine dense and sparse rankings. |
| Evaluation | RAGAS | Measures faithfulness, context recall, and factual correctness against gold answers. |
| API | FastAPI | Lightweight serving layer for the agent. |
| Environment | uv | Fast Python dependency and environment management. |
| Deploy | Docker + VPS | Reproducible deployment with minimal infra overhead. |

## Source map

| Layer | Primary package |
|---|---|
| Configuration | `src.config` |
| Ingestion | `src.ingestion` |
| Retrieval | `src.retrieval` |
| Agent | `src.agent` |
| Generation | `src.generation` |
| Observability | `src.observability` |
| Evaluation | `evals/` (functional harness + RAGAS) |
| Serving | `src.api` |
