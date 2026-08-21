"""
Context assembly layer.

Takes ranked retrieval results and produces a Context suitable for
generation. Key responsibilities:

1. Deduplicate by canonical passage_id (same passage from multiple
   representations → keep the highest-scored one).
2. If a hit is a sentence representation, look up the full parent passage.
3. Apply a configurable context budget (max passages / max characters).
4. Preserve ranking information and source/language metadata.
5. NEVER include evaluation-only fields (is_selected, Answer, etc.).

The context budget is a production parameter. The architecture does not
specify an exact budget, so it is configurable.
"""
from __future__ import annotations

import structlog

from app.models.generation import Context
from app.models.retrieval import (
    Passage,
    Query,
    RetrievalResult,
)

logger = structlog.get_logger(__name__)


def assemble_context(
    query: Query,
    results: list[RetrievalResult],
    passage_store: dict[str, str],
    *,
    max_passages: int = 10,
    max_chars: int = 4000,
) -> Context:
    """
    Assemble a Context from ranked retrieval results.

    Args:
        query: The original user query.
        results: Ranked retrieval results (may include sentence-level hits).
        passage_store: Mapping of canonical passage_id → full passage text.
            This is the production corpus store, NOT evaluation data.
        max_passages: Maximum number of distinct passages to include.
        max_chars: Maximum total character count across all passages.

    Returns:
        A Context object ready for generation.

    Raises:
        ValueError: If results is empty (caller should check and handle).
    """
    if not results:
        raise ValueError(
            "Cannot assemble context from empty results. "
            "The caller should check for empty retrieval before calling."
        )

    # Step 1: Deduplicate by canonical passage_id, keeping highest score.
    seen: dict[str, tuple[int, RetrievalResult]] = {}  # passage_id → (rank, result)
    for rank, result in enumerate(results):
        pid = result.passage.passage_id
        if pid not in seen or result.score > seen[pid][1].score:
            seen[pid] = (rank, result)

    # Step 2: Sort by original rank (preserve retrieval order).
    deduped = sorted(seen.values(), key=lambda x: x[0])

    # Step 3: Expand sentence hits to parent passages.
    assembled_passages: list[RetrievalResult] = []
    for _rank, result in deduped:
        pid = result.passage.passage_id
        full_text = passage_store.get(pid, "")

        if not full_text:
            logger.warning(
                "passage_not_in_store",
                passage_id=pid,
                representation_type=result.passage.lang,
            )
            continue

        # Replace passage text with full parent text if available.
        expanded_passage = Passage(
            passage_id=pid,
            text=full_text,
            lang=result.passage.lang,
        )
        assembled_passages.append(
            RetrievalResult(
                passage=expanded_passage,
                score=result.score,
                source=result.source,
            )
        )

    if not assembled_passages:
        raise ValueError(
            "No passages survived context assembly. "
            "All retrieval results had missing passage text."
        )

    # Step 4: Apply context budget.
    budgeted: list[RetrievalResult] = []
    total_chars = 0

    for result in assembled_passages:
        passage_text = result.passage.text

        # Check passage count budget.
        if len(budgeted) >= max_passages:
            logger.info(
                "context_budget_passages_reached",
                max_passages=max_passages,
            )
            break

        # Check character budget.
        if total_chars + len(passage_text) > max_chars and budgeted:
                logger.info(
                    "context_budget_chars_reached",
                    max_chars=max_chars,
                    total_chars=total_chars,
                )
                break

        budgeted.append(result)
        total_chars += len(passage_text)

    logger.info(
        "context_assembled",
        query_lang=query.lang.value,
        input_results=len(results),
        deduped_results=len(deduped),
        assembled_results=len(assembled_passages),
        budgeted_results=len(budgeted),
        total_chars=total_chars,
    )

    return Context(query=query, passages=budgeted)
