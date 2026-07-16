# Status

Living document. Update at the end of every phase.

## Current phase

**Phase 4 is complete as of 2026-07-16**: the structural guardrail, deterministic functional evals, the RAGAS report-only suite, LangSmith tracing, a minimal FastAPI serving layer (`POST /ask`, `GET /health`), Docker + docker compose packaging, and VPS deployment are in place. The live demo runs at http://187.127.9.91/api/docs behind nginx, with the API bound to loopback and proxied under `/api/`.

**Current eval results (2026-07-15, `gpt-4o-mini`):**

- Functional agent evals: **11/11** across router, grounding, guardrail, factual, exact-term, and rewrite-loop families.
- RAGAS (6 gold-answer cases): faithfulness **1.000**, context_recall **1.000**, factual_correctness **0.693** (F1 against reference wording; grounded-but-verbose answers score below exact-match phrasing).

**Next:** Phase 5 polish — retrieval tuning informed by RAGAS, plus a custom domain + TLS for the demo endpoint.

**Cost / latency note:** agent questions typically take 2–4 LLM calls.

## Phase table

| Phase | Scope | Status |
|---|---|---|
| Phase 0 | Setup | ✅ complete |
| Phase 1 | Basic RAG (days 2–5) | ✅ complete |
| Phase 2 | Hybrid retrieval (BM25 + RRF) (days 6–8) | ✅ complete |
| Phase 3 | LangGraph agent (days 9–11) | ✅ complete |
| Phase 4 | Grounding / guardrails / API / deploy (days 12–13) | ✅ complete |
| Phase 5 | RAGAS evals + polish (day 14+) | first RAGAS suite done; tuning pending |

## Corpus

- Corpus selected: **Airbnb 10-K FY2025** (Items **1 / 1A / 7**) as plain text in `data/raw/` (committed; SEC filings are public domain).
- Source: **SEC EDGAR**.

## Engineering quality

- CI: ruff (lint + format), mypy, pytest on every push/PR to `main`.
- Unit tests cover retrieval fusion, generation helpers, the agent graph (with fake LLMs), and the API (with a mocked agent).
- The project is an installable package with a `finance-ask` entry point.

## Post-core roadmap

- Reranking
- LangSmith eval harness
- Query transformation
- Open-source models
- UI
- GraphRAG
