# HH Goa 2026 — Task 2: Voice-Enabled Multilingual RAG

## Project

A voice-enabled retrieval-augmented generation system built over
[`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI),
a multilingual (Hindi/English) Indic passage-ranking dataset derived from
MS MARCO. Built for HH Goa 2026, Task 2.

**Priorities, in order:** retrieval quality, latency, robustness/reliability,
groundedness/guardrails, engineering simplicity, demo quality.

## Architecture

```
Voice
  → STT
  → Query Processing
  → Retrieval (BM25 + bge-m3/Qdrant, hybrid)
  → Context Assembly
  → Generation
  → Guardrails
  → Response
```

Full architecture, including every frozen decision and every currently
unconfirmed detail, is documented in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
That document is the source of truth — this README is a summary only.

**Confirmed technology:** BM25 (sparse), bge-m3 (dense embedding, used
off-the-shelf), Qdrant (vector store). STT provider, generation model, and
exact reranker candidates are **not yet confirmed** — see
`docs/ARCHITECTURE.md` for the full list of open items.

## Dataset

[`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI)
— a multilingual passage-ranking dataset with Hindi and English passage
pairs and query-level relevance judgments (`is_selected`). No claims about
the dataset beyond what its own card states are made here; dataset-specific
integrity findings (corpus/benchmark leakage risk, label incompleteness,
etc.) are documented in `docs/ARCHITECTURE.md`.

## Repository Structure

```
app/            FastAPI application: API routes, config, error types,
                typed models, service interfaces, orchestration harness
ingestion/      Offline: dataset loading, normalization, dedup,
                representation, indexing (NOT implemented yet)
retrieval/      Sparse/dense/fusion/reranking/routing interfaces
embeddings/     Embedding provider interface (bge-m3)
generation/     Generator interface
guardrails/     Guardrail interface (Semantic Grounding Consistency Check
                design pending confirmation)
evaluation/     Evaluation-only models and gold-label handling —
                structurally isolated from app/ and retrieval/
benchmark/      Benchmark result schemas
tests/          unit / integration / retrieval / benchmark test suites
scripts/        (empty — reserved for future CLI/utility scripts)
configs/        Non-secret experiment configuration (empty — tier and
                ablation definitions pending)
docs/           Architecture, decision records, DeepSeek implementation
                contract
data/           Dataset storage (gitignored, .gitkeep only)
artifacts/      Generated artifacts — embeddings, indexes (gitignored,
                .gitkeep only)
```

## Current Status

**Repository bootstrapped. Core implementation pending.**

This repository is a scaffold: typed models, interfaces, API skeleton,
error handling, latency instrumentation, and test structure exist. The
actual RAG pipeline — dataset ingestion, embeddings, indexing, retrieval,
reranking, generation, guardrails — is **not implemented**. `/query` and
`/voice/query` return `501 Not Implemented` honestly rather than
fabricated responses.

Several architecture details could not be scaffolded because their
specification text was not available at bootstrap time (STT provider,
generation model, guardrail internals, reranker A/B/C identities, corpus
tier sizes). These are marked `ARCHITECTURE DETAIL MISSING — REQUIRES
CONFIRMATION` throughout `docs/ARCHITECTURE.md` and the code.

## Development Setup

Requires Python 3.12+.

```bash
python -m venv .venv
source .venv/bin/activate
make install          # pip install -e ".[dev]"
cp .env.example .env  # fill in real values locally; never commit .env
make test
make lint
make typecheck
make run              # starts the API on http://localhost:8000
```

## Environment Variables

See [`.env.example`](.env.example) for the full list. Nothing in that file
is a real secret or a confirmed provider default — several fields are
intentionally left blank pending architecture confirmation.

## Development Phases

Implementation order (see `docs/DEEPSEEK_IMPLEMENTATION.md` for the
authoritative, detailed contract):

1. Phase 1 sanity checks (query-text-in-corpus scan, train/validation
   near-duplicate check) — mandatory before any threshold is set
2. Corpus construction (broad-pool sampling, decoupled from benchmark
   queries)
3. Ingestion: normalization, canonical-hash dedup, embedding, indexing
4. Retrieval: BM25, dense (bge-m3/Qdrant), fusion
5. Reranker A/B/C experiment (including "no reranker" as a legitimate
   outcome)
6. Monolingual vs. cross-lingual retrieval experiment (Config 1/Config 2)
7. Context assembly, generation, guardrails
8. Benchmark execution and reporting (validation-split, touched once)

## Benchmarking

Benchmark result schemas live in `evaluation/benchmark_models.py`. No
benchmark has been executed — no numbers in this repository are real.
Results will eventually live under `artifacts/benchmark/` (gitignored) and
be summarized in `benchmark/`.
