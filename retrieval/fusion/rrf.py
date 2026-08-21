"""
Reciprocal Rank Fusion (RRF) implementation.

Combines sparse (BM25) and dense (bge-m3) ranked result lists into a
single deterministic ranking using RRF.

RRF formula:
    score(d) = sum over lists of: 1 / (k + rank_i(d))

Where k is a smoothing constant (default 60, as used in the original
RRF paper by Cormack et al. 2009). A higher k reduces the impact of
top-ranked documents.

The function:
- Deduplicates by canonical passage_id (NOT text equality)
- Preserves language metadata from whichever source provides it
- Produces deterministic ordering (ties broken by passage_id)
- Supports empty input from either retriever
"""
from __future__ import annotations

import structlog

from app.models.retrieval import Passage, RetrievalResult

logger = structlog.get_logger(__name__)

# Default RRF smoothing constant.
# k=60 is from the original Cormack et al. 2009 paper.
# Architecture does not freeze this value — it is configurable.
DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    sparse_results: list[RetrievalResult],
    dense_results: list[RetrievalResult],
    top_k: int,
    k: float = DEFAULT_RRF_K,
) -> list[RetrievalResult]:
    """
    Combine sparse and dense retrieval results using Reciprocal Rank Fusion.

    Args:
        sparse_results: Ranked results from BM25 (may be empty).
        dense_results: Ranked results from dense retrieval (may be empty).
        top_k: Maximum number of results to return.
        k: RRF smoothing constant. Higher k = less weight to top ranks.

    Returns:
        Deduplicated, ranked list of RetrievalResult with source="fused".
        Ties broken deterministically by passage_id.
    """
    # Build RRF scores: passage_id -> (rrf_score, passage, source_set)
    rrf_scores: dict[str, float] = {}
    passage_map: dict[str, Passage] = {}
    source_set: dict[str, set[str]] = {}

    # Process sparse results
    for rank, result in enumerate(sparse_results, start=1):
        pid = result.passage.passage_id
        rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (k + rank)
        passage_map[pid] = result.passage
        if pid not in source_set:
            source_set[pid] = set()
        source_set[pid].add("bm25")

    # Process dense results
    for rank, result in enumerate(dense_results, start=1):
        pid = result.passage.passage_id
        rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (k + rank)
        passage_map[pid] = result.passage
        if pid not in source_set:
            source_set[pid] = set()
        source_set[pid].add("dense")

    if not rrf_scores:
        return []

    # Sort by RRF score (descending), break ties by passage_id (ascending)
    sorted_pids = sorted(
        rrf_scores.keys(),
        key=lambda pid: (-rrf_scores[pid], pid),
    )

    # Build results
    results: list[RetrievalResult] = []
    for pid in sorted_pids[:top_k]:
        sources = source_set[pid]
        source_label = "+".join(sorted(sources)) if len(sources) > 1 else next(iter(sources))
        results.append(RetrievalResult(
            passage=passage_map[pid],
            score=rrf_scores[pid],
            source=source_label,
        ))

    logger.debug(
        "rrf_fusion",
        sparse_count=len(sparse_results),
        dense_count=len(dense_results),
        fused_count=len(results),
        top_k=top_k,
    )

    return results
