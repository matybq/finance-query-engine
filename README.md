# finance-query-engine

Dense financial filings are hard to query reliably: generic LLM answers tend to blur missing evidence, misread figures, or hallucinate when the filing is ambiguous. This project is a grounded agentic RAG system for asking natural-language questions over financial filings, with a refusal-over-hallucination stance: answers must cite source evidence, and when evidence is weak the system should say so explicitly instead of pretending certainty.

## Architecture

```text
ingestion → dual index (Chroma + BM25) → hybrid retrieval (dense + sparse + RRF) → LangGraph agent → grounded generation → guardrail
```

Canonical agent flow:

```text
router → retrieve → grade → generate
                  ↘ rewrite ↺
                  ↘ insufficient_evidence
```

The first guardrail is structural: if retrieved evidence is still insufficient after the rewrite budget is exhausted, the graph routes to `insufficient_evidence` instead of calling generation.

## Corpus

The current corpus is the **Airbnb 10-K for fiscal year 2025** (period ended **2025-12-31**), sourced from **SEC EDGAR**: https://www.sec.gov/ix?doc=/Archives/edgar/data/0001559720/000155972026000004/abnb-20251231.htm

Three sections were extracted as plain text into `data/raw/` (gitignored):

- Item 1 — Business
- Item 1A — Risk Factors
- Item 7 — Management's Discussion & Analysis

Why this corpus: one dense public filing with technical language, figures, and long-form structure is enough to exercise chunking and retrieval for real without multi-document complexity in the core phases. The system is corpus-agnostic, and more filings are roadmap.

## Stack

Implemented:

- LangGraph for orchestration
- OpenRouter for LLM access
- OpenAI `text-embedding-3-small` for embeddings
- Chroma for dense storage
- BM25 for sparse retrieval
- RRF for rank fusion
- RAGAS for report-only evaluation
- pytest for unit tests
- GitHub Actions for lightweight CI
- uv for Python environment management

Roadmap:

- FastAPI for serving
- Docker + VPS for deployment

## Status

Early development. **Fase 3 is done**: agentic routing + self-correcting retrieval now power generation. A first structural guardrail, deterministic functional evals, a small report-only RAGAS suite, pytest unit tests, and lightweight CI are in place.

The current interface is a CLI. FastAPI serving, Docker packaging, and VPS deployment remain roadmap items.

Design docs (architecture, ADR log, phase status) are maintained locally and will be published when the core is complete.

## Usage

```bash
uv run python -m src.ingestion.ingest
./finance-ask "What is AirCover for Hosts?"
```

For an interactive session with a short welcome guide:

```bash
./finance-ask
```

## Evaluation

The project includes two evaluation layers.

### Functional regression checks

```bash
uv run python evals/evaluate_agent.py
```

This lightweight deterministic harness validates core behavior across router decisions, out-of-corpus refusals, factual retrieval, exact-term regressions, rewrite-loop behavior, and the structural insufficient-evidence guardrail.

Current functional eval result:

| Family | Passed |
|---|---:|
| router | 3/3 |
| grounding | 2/2 |
| guardrail | 1/1 |
| factual | 2/2 |
| exact_term | 2/2 |
| rewrite_loop | 1/1 |
| overall | 11/11 |

### RAGAS report-only evals

A first RAGAS suite lives in `evals/ragas_cases.jsonl` and measures semantic RAG quality over a small set of gold-answer cases:

- `faithfulness` — whether the answer is supported by retrieved chunks
- `context_recall` — whether retrieved chunks contain the facts needed by the reference answer
- `factual_correctness` — whether the answer matches the reference answer

Install the optional eval dependencies and run:

```bash
uv sync --extra eval
uv run --extra eval python evals/evaluate_ragas.py
```

To control cost while iterating:

```bash
uv run --extra eval python evals/evaluate_ragas.py --limit 2
```

RAGAS results are diagnostic and report-only for now. CSV reports are written to `evals/experiments/` and are gitignored.

### Unit tests

```bash
uv run --group dev pytest
```

These tests cover pure retrieval and formatting helpers without API keys or corpus data.

### CI

GitHub Actions runs lightweight checks on pushes and pull requests to `main`:

- locked dependency install with eval and dev dependencies
- Python compilation for `src`, `evals`, and `tests`
- RAGAS import smoke test
- pytest unit tests

CI intentionally does not run the LangGraph agent or RAGAS scoring yet, because those require API keys plus the local corpus/index.

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
