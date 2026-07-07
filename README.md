# finance-query-engine

Dense financial filings are hard to query reliably: generic LLM answers tend to blur missing evidence, misread figures, or hallucinate when the filing is ambiguous. This project is a grounded agentic RAG system for asking natural-language questions over financial filings, with a refusal-over-hallucination stance: answers must cite source evidence, and when evidence is weak the system should say so explicitly instead of pretending certainty.

## Architecture

```text
ingestion → dual index (Chroma + BM25) → hybrid retrieval (dense + sparse + RRF) → LangGraph agent → grounded generation → guardrail
```

Canonical agent flow:

```text
router → retrieve (tool) → grade → (rewrite loop back to retrieve) → generate → guardrail
```

## Corpus

The current corpus is the **Airbnb 10-K for fiscal year 2025** (period ended **2025-12-31**), sourced from **SEC EDGAR**: https://www.sec.gov/ix?doc=/Archives/edgar/data/0001559720/000155972026000004/abnb-20251231.htm

Three sections were extracted as plain text into `data/raw/` (gitignored):

- Item 1 — Business
- Item 1A — Risk Factors
- Item 7 — Management's Discussion & Analysis

Why this corpus: one dense public filing with technical language, figures, and long-form structure is enough to exercise chunking and retrieval for real without multi-document complexity in the core phases. The system is corpus-agnostic, and more filings are roadmap.

## Stack

- LangGraph for orchestration
- OpenRouter for LLM access
- OpenAI `text-embedding-3-small` for embeddings
- Chroma for dense storage
- BM25 for sparse retrieval
- RRF for rank fusion
- RAGAS for evaluation
- FastAPI for serving
- uv for Python environment management
- Docker + VPS for deployment

## Status

Early development. **Fase 2 is done**: hybrid retrieval (dense + BM25 + weighted RRF) now powers generation.

Design docs (architecture, ADR log, phase status) are maintained locally and will be published when the core is complete.

## Usage

```bash
uv run python -m src.ingestion.ingest
uv run python -m src.ask "What is AirCover for Hosts?"
```

## Setup

Requirements: Python >= 3.12 and `uv`.

You also need:

- `OPENROUTER_API_KEY` for LLM calls via OpenRouter
- `OPENAI_API_KEY` for embeddings via OpenAI directly (OpenRouter does not proxy the embeddings endpoint)

```bash
uv sync
cp .env.example .env
```

Then fill in the required API keys and paths in `.env`. The corpus itself is not committed; `data/` is gitignored. See [Corpus](#corpus) for how it is sourced.

## Honesty note

This repo is intentionally not presenting fake demos or future features as finished. If something is not implemented yet, it should be treated as roadmap, not current capability.
