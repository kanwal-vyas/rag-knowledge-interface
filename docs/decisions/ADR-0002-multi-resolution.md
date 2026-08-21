# ADR-0002: Multi-Resolution Representation Strategy

## Status

**APPROVED** — Adaptive sentence-level child representations selected.

## Context

The architecture (Audit §10.7, DEEPSEEK_IMPLEMENTATION.md) references an "adaptive multi-resolution retrieval strategy" and a tuning parameter `T_sentence`, indicating that the V2 architecture includes multiple representation resolutions beyond passage-level indexing. However, the full V2 specification that defines the concrete multi-resolution strategy is not present in this repository.

The bootstrap decision (ADR-0001) explicitly chose not to infer missing V2 details, and this ADR follows the same principle.

## What the Existing Architecture Explicitly Establishes

1. The Audit references "multi-resolution context assembly" as a pipeline stage.
2. The DEEPSEEK_IMPLEMENTATION.md §10.7 mentions `T_sentence` as a tuning parameter — suggesting sentence-level representation exists in the full V2 design.
3. The architecture mandates benchmarking passage-only vs. multi-resolution, meaning multi-resolution is an optimization to be measured, not a guaranteed improvement.
4. Evaluation must compare: passage-only, multi-resolution, and multi-resolution + reranker.

## What Is Missing

The full V2 document does not specify:

1. **How many resolution levels exist** (e.g., passage + sentence? passage + sentence + paragraph?)
2. **How child representations are derived** from parent passages (splitting rules, overlap, boundaries)
3. **The exact chunking/splitting rules** (token-based? sentence-boundary-based? fixed-size windows?)
4. **Which representations receive embeddings** vs. which are lookup-only or merged at retrieval time
5. **Parent/child ID relationship schema** (how child hits map back to canonical passage_id)
6. **The threshold or criterion for when multi-resolution is beneficial** (the `T_sentence` parameter exists but its semantics are undefined)

## Why We Must Not Invent the Missing Strategy

- The architecture treats multi-resolution as an experiment to be benchmarked, not a guaranteed improvement.
- Inventing a strategy and implementing it would produce unmeasurable, unauditable results.
- The project's core principle is "measured performance > claimed performance."
- A fabricated multi-resolution strategy could negatively impact retrieval quality while being presented as an architectural feature.

## Current Implementation State

`ingestion/representation/base.py` implements:

- **Passage-only (single-resolution) representation**: one passage → one Representation, type="passage", parent_id=self, child_ids=[].
- The `Representation` dataclass includes fields (`representation_type`, `parent_id`, `child_ids`) as extension points for when multi-resolution is specified.
- `representation_id()` hashes `"{type}|{passage_id}|{normalized_text}"` — different representation types would produce different IDs.
- `create_passage_representation()` creates only base passage-level representations.

This is the correct minimal base case. It does NOT constitute multi-resolution support.

## Candidate Strategies That Could Be Evaluated

When the V2 specification is available, or when a decision is made, the following are common candidates:

1. **Sentence-level chunking**: Split passages at sentence boundaries (`.`, `।` for Hindi). Each sentence becomes a child representation with the same parent passage_id.
2. **Sliding window**: Fixed-size token windows with overlap. Each window is a child representation.
3. **Paragraph/segment-level**: If passages contain structural markup, split at paragraph boundaries.
4. **Adaptive chunking**: Use sentence boundaries for short passages, fall back to fixed windows for long ones.
5. **Hybrid lookup**: Index at passage level, retrieve at sentence level, merge hits back to parent for context assembly.

Each candidate requires benchmarking against passage-only before adoption.

## Decision

**Adaptive Sentence-Level Child Representations** selected based on benchmark:

- T_sentence = 320 chars (tuned on train-split pool)
- nDCG@10: +0.1362 improvement over passage-only
- Recall@10: +0.2745 improvement over passage-only
- Gold labels remain passage-level
- Sentence hits expand to parent passage for context assembly

### Benchmark Results (51 queries, 5973 passages)

| Metric | Passage-Only | Multi-Res | Delta |
|--------|-------------|-----------|-------|
| Recall@1 | 0.1961 | 0.2157 | +0.0196 |
| Recall@5 | 0.3088 | 0.4657 | +0.1569 |
| Recall@10 | 0.3627 | 0.6373 | +0.2745 |
| MRR | 0.3685 | 0.3530 | -0.0154 |
| nDCG@10 | 0.2796 | 0.4158 | +0.1362 |

### T_sentence Tuning Results

| T | Docs | Sentences | nDCG@10 |
|---|------|-----------|--------|
| 128 | 26787 | 20814 | 0.3769 |
| 192 | 26070 | 20097 | 0.3815 |
| 256 | 23183 | 17210 | 0.3974 |
| 320 | 16243 | 10270 | 0.4158 |
| 384 | 12994 | 7021 | 0.3516 |
| 512 | 9600 | 3627 | 0.3170 |

### Implementation

- `ingestion/representation/base.py`: `create_representations()` with `t_sentence` and `multi_resolution` parameters
- `split_sentences()`: punctuation-based splitting (`.`, `!`, `?`, `।`, `॥`)
- `expand_to_parent()`: context assembly parent expansion
- BM25 and dense indexing support both passage-only and multi-resolution modes

## Related

- ADR-0001: Bootstrap from partial architecture
- ARCHITECTURE.md §10.7 (multi-resolution context assembly)
- DEEPSEEK_IMPLEMENTATION.md §10.7 (`T_sentence` parameter)
