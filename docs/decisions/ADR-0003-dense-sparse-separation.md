# ADR-0003: Dense and Sparse Retrieval Separation

## Status

**APPROVED** — Implementation follows this decision.

## Context

The architecture confirms both BM25 (sparse) and bge-m3 (dense) as retrieval components (Audit §4/§5/§7). bge-m3 additionally supports a sparse output mode that could theoretically replace BM25. This ADR establishes the separation boundary.

## Decision

**Dense retrieval**: bge-m3 dense output (1024-dim L2-normalized vectors, cosine similarity via Qdrant).

**Sparse retrieval**: BM25 (implemented separately, Phase 7). NOT bge-m3 sparse output.

**Fusion**: Experimental. The architecture mandates benchmarking BM25-only, dense-only, and BM25+Dense fusion to determine whether each component contributes.

## Rationale

1. The architecture explicitly confirms BM25 as the sparse retrieval method. Replacing it with bge-m3 sparse would change an architectural constraint, not optimize within one.
2. BM25 and bge-m3-dense are independently benchmarkable. Combining the sparse output of bge-m3 with its own dense output couples two signals from the same model, making ablation difficult.
3. The retrieval evaluation must answer: "Does BM25 contribute beyond what dense retrieval provides?" Using bge-m3 sparse would make this question unanswerable.
4. If bge-m3 sparse is later shown to outperform BM25 in the fusion benchmark, this ADR can be revisited with evidence.

## Implementation Boundary

| Component | Implementation |
|-----------|---------------|
| Dense vectors | `embeddings/bge_m3.py` → `BgeM3EmbeddingProvider.embed_passages()` (dense mode) |
| Sparse index | `retrieval/sparse/` → BM25 implementation (Phase 7) |
| Dense index | `retrieval/dense/qdrant_index.py` → `QdrantIndexManager` |
| Fusion | `retrieval/fusion/` → RankFusion implementation (Phase 8) |
| Ablation benchmark | `benchmark/` → Compare BM25, Dense, BM25+Dense (Phase 9) |

## Related

- ADR-0001: Bootstrap from partial architecture
- Audit §4 (confirmed components)
- Audit §7 (Config 1 vs Config 2 experiment)
