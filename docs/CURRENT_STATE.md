# HH Goa 2026 — Task 2: Current State

**Date**: 2026-08-21
**Test Results**: 297/297 passing
**Lint**: ruff clean (all checks passed)

---

## API Endpoints

| Method | Path | Status | Description |
|--------|------|--------|-------------|
| GET | `/health` | IMPLEMENTED, TESTED, RUNTIME VERIFIED | Returns `{"status":"ok","environment":"development"}` |
| POST | `/query` | STUB (501) | Returns 501 with `NotImplementedResponse`. TextPipeline exists but is not wired to this route. |
| POST | `/voice/query` | STUB (501) | Returns 501 with `NotImplementedResponse`. VoicePipeline exists but is not wired to this route. |
| GET | `/docs` | IMPLEMENTED, RUNTIME VERIFIED | FastAPI Swagger UI |

---

## Request/Response Schemas

### POST /query

**Request**:
```json
{
  "query_text": "string",
  "lang": "en",
  "retrieval_mode": "monolingual" | "cross_lingual"
}
```

**Response (current — 501 stub)**:
```json
{
  "detail": "Query pipeline is not implemented yet (bootstrap stage).",
  "request_id": "uuid"
}
```

**Response (when wired — planned)**:
```json
{
  "answer_text": "string",
  "grounded": true
}
```

### POST /voice/query

**Request**: multipart/form-data with audio file
**Response (current — 501 stub)**: Same as /query stub

---

## How to Start Backend

```bash
cd hh-goa-rag
pip install -e ".[dev]"
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Swagger UI: http://127.0.0.1:8000/docs

## Frontend

**No frontend exists.** This is a backend-only API project.

---

## Required Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SARVAM_API_KEY` | For voice queries | None | Sarvam AI subscription key for STT |
| `DEEPSEEK_API_KEY` | For generation | None | DeepSeek API key for LLM generation |
| `HF_EMBEDDING_DEVICE` | Optional | `cpu` | Device for bge-m3 (cpu/cuda/mps) |
| `QDRANT_URL` | Optional | `http://localhost:6333` | Qdrant server URL |

All other settings have safe defaults for development.

---

## Architecture Status

### Implemented, Tested, Runtime Verified

| Component | File(s) | Tests | Status |
|-----------|---------|-------|--------|
| BM25 sparse retrieval | `retrieval/sparse/bm25_index.py`, `bm25_retriever.py` | 34 tests | bm25 P50=0.8ms (small corpus), P50=131ms (3K docs) |
| RRF fusion | `retrieval/fusion/rrf.py` | 16 tests | Correctness verified |
| Hybrid retriever | `retrieval/fusion/hybrid_retriever.py` | 2 tests | Correctness verified |
| bge-m3 embeddings | `embeddings/bge_m3.py` | 5 tests | Runtime verified on CPU, 1024-dim, float32, L2-normalized |
| Qdrant index | `retrieval/dense/qdrant_index.py` | 10 tests | In-memory verified |
| Device detection | `embeddings/device.py` | 13 tests | CPU/CUDA/MPS auto-detect |
| Multi-resolution | `ingestion/representation/base.py` | 28 tests | T_sentence=320, +49% nDCG over passage-only |
| Language mapping | `ingestion/dataset/language.py` | 17 tests | 14 dataset languages |
| Corpus construction | `ingestion/dataset/corpus.py` | 12 tests | Dedup, canonical IDs, evaluation separation |
| Evaluation harness | `evaluation/harness.py` | 15 tests | Recall@K, MRR, nDCG |
| Context assembly | `generation/context_assembly.py` | 9 tests | Dedup, parent expansion, budget |
| Grounding validation | `generation/grounding.py` | 7 tests | Refusal detection, empty checks |
| Guardrails | `guardrails/implementation.py` | 12 tests | Empty, off-topic, unsafe, insufficient |
| TextPipeline | `app/harness/text_pipeline.py` | 6 tests | End-to-end with stub components |
| VoicePipeline | `app/harness/text_pipeline.py` | 7 tests | STT → text pipeline with stub STT |
| Sarvam STT | `app/services/sarvam_stt.py` | 0 (API call) | Interface implemented, audio validation tested |
| DeepSeek generation | `generation/deepseek_provider.py` | 0 (API call) | Interface implemented, StubGenerator tested |
| Latency models | `app/models/latency.py` | 2 tests | Monotonic clock, required fields |
| Model isolation | `app/models/retrieval.py`, `generation.py` | 13 tests | No evaluation field leakage |

### Implemented, Tested, NOT Runtime Verified

| Component | Why Not Verified |
|-----------|-----------------|
| Dense retrieval quality | bge-m3 model loading exceeds CPU timeout in this environment. Embedding generation works but full evaluation blocked. |
| Hybrid retrieval quality | Depends on dense retrieval being evaluated first. |
| DeepSeek generation | Requires DEEPSEEK_API_KEY. StubGenerator works. |
| Sarvam STT | Requires SARVAM_API_KEY. Interface and validation tested. |
| Full voice pipeline | Depends on real STT + real generation. Stub-based tests pass. |

### NOT Implemented

| Component | Reason |
|-----------|--------|
| Multi-resolution (ADR-0002) | APPROVED by user. T_sentence=320. Implementation exists but not benchmarked with dense retrieval. |
| Reranking | Architecture says 3 candidates A/B/C. Which 3 not specified. Deferred. |
| Docker deployment | Not started. |
| API route wiring | TextPipeline/VoicePipeline exist but /query and /voice/query routes still return 501. |
| Production Qdrant | Only in-memory Qdrant tested. Server-mode Qdrant not tested. |

---

## Retrieval Configuration

| Setting | Value | Source |
|---------|-------|--------|
| Sparse retriever | BM25 (rank_bm25 library) | Architecture Audit §4 |
| Dense retriever | bge-m3 (1024-dim, float32) | Architecture Audit §4/§7 |
| Vector store | Qdrant (in-memory for tests) | Architecture Audit §5/§7 |
| Fusion | Reciprocal Rank Fusion (k=60) | Implemented, not frozen by architecture |
| Multi-resolution | Adaptive sentence-level (T_sentence=320) | ADR-0002, tuned on train pool |
| Tokenization | Unicode word-boundary + bigrams for agglutinative scripts | Phase 7 |

### BM25 Performance (measured)

| Metric | Value |
|--------|-------|
| Recall@10 | 0.7451 |
| MRR | 0.3685 |
| nDCG@10 | 0.3696 |
| P50 latency | 131ms (5,973 docs, CPU) |

### Dense Retrieval Performance (measured, CPU only)

| Metric | Value |
|--------|-------|
| Embedding dimension | 1024 |
| Embedding dtype | float32 |
| Normalization | L2 (unit norm) |
| Query embedding P50 | 348ms (CPU) |
| Qdrant search P50 | 5.5ms |
| End-to-end P50 | 353ms (CPU) |

**Note**: CPU dense retrieval exceeds the 200ms controlled-path target. GPU required for production latency.

---

## Generation Model

| Setting | Value |
|---------|-------|
| Provider | DeepSeek (OpenAI-compatible API) |
| Model | deepseek-chat |
| Temperature | 0.0 (deterministic) |
| Max tokens | 1024 |
| System prompt | Answer ONLY from provided context. Refuse when insufficient. |
| Refusal detection | 15+ pattern matching (English + Hindi aware) |

---

## STT Provider

| Setting | Value |
|---------|-------|
| Provider | Sarvam AI |
| Model | Saaras v3 |
| Endpoint | https://api.sarvam.ai/speech-to-text |
| Modes | transcribe, translate, verbatim, translit, codemix |
| Max duration | 30 seconds (REST API) |
| Languages | 22 Indian + English |
| Auth | SARVAM_API_KEY env var |

---

## Known Runtime Blockers

1. **No SARVAM_API_KEY**: Voice pipeline cannot make real STT calls. Interface is complete and tested with stubs.

2. **No DEEPSEEK_API_KEY**: Generation cannot call DeepSeek API. StubGenerator works for testing.

3. **CPU-only dense retrieval**: bge-m3 embedding on CPU takes ~348ms per query. The 200ms controlled-path target requires GPU.

4. **API routes not wired**: The `/query` and `/voice/query` endpoints still return 501. TextPipeline and VoicePipeline exist as Python classes but are not connected to the FastAPI routes.

5. **No Docker deployment**: The application runs locally only.

6. **No production Qdrant**: Only in-memory Qdrant is tested. For production, a Qdrant server instance is needed.

7. **Multi-resolution not benchmarked with dense**: The T_sentence=320 adaptive strategy was benchmarked with BM25 only. Dense retrieval with multi-resolution representations has not been evaluated.

---

## Test Results (actual run)

```
297 passed, 5 warnings in 7.41s

warnings:
- PydanticDeprecatedSince211: Accessing model_fields on instance (test_hybrid_retrieval.py)
- UserWarning: Payload indexes have no effect in local Qdrant (test_qdrant_index.py)
```

```
ruff check . → All checks passed!
```
