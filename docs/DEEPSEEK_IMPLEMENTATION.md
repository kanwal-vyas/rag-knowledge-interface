# DeepSeek Implementation Handoff

**Read this before writing any implementation code.**

## The one rule that matters most

**Implement the approved architecture. Do not redesign it.** If
implementation reveals a genuine architectural problem, document it (in
`docs/decisions/`, as a new numbered ADR) and stop — do not silently change
the architecture to work around it. This applies even when the fix seems
obvious or small.

## Repository Structure

See `README.md` "Repository Structure" for the directory map. Key
boundaries to respect:

- `evaluation/` is structurally isolated from `app/`, `retrieval/`,
  `embeddings/`, `generation/`, `guardrails/`. **No file in those packages
  may import anything from `evaluation.*`, ever.** This is enforced by
  `tests/unit/test_model_isolation.py` (a static AST scan) — do not weaken
  or delete that test to make an import work.
- `ingestion/` is offline-only. `app/` (the online serving path) must not
  import from `ingestion/` at request time — ingestion produces artifacts
  (the Qdrant collection, the corpus) that `app/` then reads at query time,
  not code that runs inline in a request.

## Architecture Source of Truth

`docs/ARCHITECTURE.md`. It is derived from exactly one document: the
"Final Pre-Implementation Audit," which patches specific sections of a
prior "Architecture V2" document that is **not currently in this
repository**. Every gap where V2's actual text was needed but unavailable
is marked inline:

> **ARCHITECTURE DETAIL MISSING — REQUIRES CONFIRMATION**

**Do not fill these gaps yourself.** Do not infer the missing content from
general RAG practice, from what "would make sense," or from this
scaffold's placeholder code — the placeholders were written to be
importable, not to encode a real decision. If you need one of these
details to proceed, stop and ask for the actual V2 text or a human
decision. Filling a gap silently is the exact failure mode this whole
process (audit → freeze → scaffold → handoff) was built to prevent.

Known gaps as of bootstrap time (not exhaustive — check
`docs/ARCHITECTURE.md` directly for the current list):
- Full harness/orchestration stage list beyond what this scaffold infers
- Guardrail internals, including the Semantic Grounding Consistency Check
  design
- Which 3 rerankers make up the A/B/C experiment
- Corpus tier (T1–T5) exact sizes and the ablation matrix
- STT provider selection
- Production generation model/provider selection
- Fusion algorithm (RRF vs. weighted vs. other)
- Canonical `passage_id` hash algorithm's exact normalization rules (a
  placeholder SHA-256-over-stripped-text implementation exists in
  `ingestion/deduplication/canonical_id.py` — confirm before relying on it,
  and never change it after gold-set data has been generated against it)

## Frozen Decisions (do not reinterpret)

All of the following are locked, verbatim from the Audit. Full detail is
in `docs/ARCHITECTURE.md` §3–§10; summary:

1. Corpus is a broad pool (thousands of query rows minimum per tier),
   train+validation combined, coverage-awareness is a backstop only —
   never build the corpus by starting from benchmark queries' candidate
   passages.
2. Three query pools, never mixed: train-split for all tuning, validation-
   split for final reporting only, touched exactly once.
3. `source_query_ids`, `is_selected`, `Answer`, `Eng_Answer` are
   structurally absent from the production Qdrant payload and the
   generation-prompt path.
4. Gold-set construction uses the same canonical content-hash `passage_id`
   as corpus dedup — never row-local list indices.
5. Monolingual vs. cross-lingual retrieval is a two-config experiment
   (Config 1 / Config 2), with an expanded gold set for the cross-lingual
   condition.
6. Two Phase 1 sanity checks (query-text-in-corpus scan, train/validation
   near-duplicate check) are mandatory before any threshold is set.
7. BM25, bge-m3, and Qdrant are the confirmed retrieval technology stack —
   off-the-shelf, not fine-tuned.

## Interfaces to Implement

All defined as `ABC` subclasses with `NotImplementedError` bodies. Provide
concrete implementations behind these interfaces — do not bypass them by
calling a library directly from the harness or API layer.

| Interface | Location | Notes |
|---|---|---|
| `EmbeddingProvider` | `embeddings/base.py` | bge-m3, dimension TBC |
| `Retriever` / `SparseRetriever` / `DenseRetriever` | `retrieval/base.py` | BM25 / bge-m3+Qdrant |
| `RankFusion` | `retrieval/base.py` | Algorithm unconfirmed |
| `Reranker` | `retrieval/base.py` | Must support `NoOpReranker` as a first-class outcome |
| `STTProvider` | `app/services/stt.py` | Provider unconfirmed |
| `Generator` | `generation/base.py` | Provider unconfirmed |
| `Guardrail` | `guardrails/base.py` | Design unconfirmed |
| `Cache` | `app/services/cache.py` | Strategy unconfirmed |

## Implementation Order

1. Phase 1 sanity checks (`ingestion/dataset/phase1_checks.py`) — must run
   and be reported on before any threshold work begins.
2. Corpus construction per the broad-pool rule (§3 above).
3. Canonical `passage_id` generation, wired identically into both corpus
   dedup and gold-set joins — one shared function, not two copies that are
   "supposed to" agree.
4. Embedding + indexing (offline path — keep timing separate from online
   query latency).
5. BM25 + dense retrievers implementing `Retriever`, both honoring
   `RetrievalMode` (Config 1/2) identically.
6. Fusion.
7. Reranker A/B/C experiment on the train/tuning pool.
8. Context assembly, generation, guardrails.
9. Wire the harness (`app/harness/orchestrator.py`) end-to-end, replacing
   the `NotImplementedError` stages.
10. Benchmark execution: tuning-pool calibration first, then exactly one
    pass against the frozen validation benchmark set.

## Testing Requirements

- Every new component gets unit tests in `tests/unit/`.
- `tests/unit/test_model_isolation.py` must continue to pass — it is the
  automated check for the evaluation-field leakage rule. If a change to
  this test is genuinely required, that's an architecture-affecting
  change and needs sign-off, not a quiet edit.
- Retrieval correctness (Recall@K etc. wiring) tests go in
  `tests/retrieval/`.
- Do not write a test that asserts a specific retrieval-quality number
  before retrieval is actually implemented and measured.

## Benchmark Requirements

- Use `evaluation/benchmark_models.py:BenchmarkResult`. Set
  `is_final_reported=True` only for a run against the frozen validation
  pool, after every tunable parameter is locked.
- Never populate a `BenchmarkResult` with fabricated numbers, including
  for demo/UI purposes. An empty/null metric is honest; a placeholder
  number is not.

## Latency Instrumentation Requirements

- Use `app/models/latency.py:LatencyBreakdown`. Use a monotonic clock
  (`time.perf_counter()`), never wall-clock timestamps, for any duration.
- Set `is_estimated=True` if any field is a projection rather than a
  directly measured value — never present an estimate as a measurement.
- Keep offline ingestion timing separate from this model; it covers the
  online query path only.

## Things DeepSeek MUST NOT Change Without Approval

- Anything listed under "Frozen Decisions" above.
- The `evaluation/` ↔ production-package import isolation boundary.
- The three-way query-pool split logic, once implemented.
- The canonical `passage_id` hash function's output format, once any
  gold-set data has been generated against it.
- The `extra="forbid"` config on any typed model in `app/models/` or
  `evaluation/`.

## Definition of Done (for a given implementation phase)

- The relevant `NotImplementedError` is replaced with a real
  implementation behind the existing interface.
- Tests exist and pass, including the isolation and schema tests already
  in the repo.
- `make lint` and `make typecheck` pass.
- No `ARCHITECTURE DETAIL MISSING` marker was silently resolved by
  guessing — either the detail was genuinely confirmed (cite where), or
  the gap still exists and the code still reflects that honestly.
- Any new frozen decision made during implementation (e.g. "we tested
  fusion algorithm X vs Y and locked X") is written up as a new ADR in
  `docs/decisions/`, not just left in code/commit history.
