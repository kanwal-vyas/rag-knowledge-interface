# ADR 0001: Repository Bootstrap Against a Partial Architecture Source

**Date:** 2026-08-18
**Status:** Accepted (as a scoping decision, not an architecture decision)

## Context

The repository bootstrap task requires preserving the FROZEN Architecture V2
verbatim. Only one source document was available at bootstrap time: the
"Final Pre-Implementation Audit," which is explicitly scoped as a patch on
top of V2 and states that most of V2 (system shape, harness, guardrails,
corpus-scaling tier list, ablation matrix, reranker A/B/C options,
Semantic Grounding Consistency Check) is "unchanged... and remains locked
as previously specified" — without reproducing that text.

## Decision

Bootstrap proceeded on **verified content only**. Every component whose
specification was not present in the Audit was left as an explicit gap in
`docs/ARCHITECTURE.md`, marked `ARCHITECTURE DETAIL MISSING — REQUIRES
CONFIRMATION`, rather than inferred from:
- general RAG best practice,
- the illustrative examples given in the bootstrap task prompt itself
  (e.g. "SarvamSTT, ElevenLabsSTT" was explicit prompt-author example text,
  not a confirmed architectural decision — treated as unconfirmed),
- prior conversational memory summaries, which are lossy and not a citable
  specification.

Interfaces and placeholder implementations for components with missing
specs (STT provider, generator, guardrail internals, exact reranker
choices) were created as **empty/minimal contracts** sufficient for the
application to import and for tests to assert the isolation boundaries the
Audit *does* specify (eval-only field separation, three-way split
enforcement) — without asserting method signatures, provider names, or
endpoint contracts that were never confirmed.

## Consequence

This repository is faithful but incomplete by design. A second bootstrap
pass is required once the actual V2 document is available, to fill in:
system shape / harness stage list, guardrail internals, reranker A/B/C
identities, corpus tier sizes, ablation matrix, STT/generation provider
selection, and the full API/latency-instrumentation contract.

Until then, any code written against a gap marker in `ARCHITECTURE.md` is,
by definition, DeepSeek inventing architecture rather than implementing it —
which the process explicitly forbids. DeepSeek must stop and request the
missing V2 text rather than filling gaps independently.
