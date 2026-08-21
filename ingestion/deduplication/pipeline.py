"""
Passage deduplication pipeline.

Architecture §3.2 (frozen):
- Deduplicate within-language and globally
- Handle exact duplicates, normalized duplicates, whitespace differences
- Unicode normalization via NFC (handled by canonical_id)
- Deterministic: same input always produces same output

The dedup pipeline operates on the raw extracted passages BEFORE
they are indexed into Qdrant or embedded. It produces a deduplicated
set of CorpusPassage objects, each with a canonical passage_id.
"""
from __future__ import annotations

from dataclasses import dataclass

import structlog

from ingestion.dataset.corpus import CorpusPassage

logger = structlog.get_logger(__name__)


@dataclass
class DedupStats:
    """Statistics from a deduplication run."""
    raw_count: int
    unique_count: int
    duplicate_count: int
    duplicate_percentage: float
    # Language breakdown of unique passages (ISO 639-1 → count)
    lang_unique: dict[str, int]


class Deduplicator:
    """
    Deterministic passage deduplicator.
    
    Usage:
        dedup = Deduplicator()
        unique_passages = dedup.deduplicate(raw_passages)
        stats = dedup.stats
    """
    
    def __init__(self) -> None:
        self._seen: dict[str, CorpusPassage] = {}  # passage_id -> first occurrence
        self._stats: DedupStats | None = None
    
    def deduplicate(
        self,
        passages: list[CorpusPassage],
    ) -> list[CorpusPassage]:
        """
        Deduplicate a list of CorpusPassage objects.
        
        Preserves the FIRST occurrence of each unique passage (by canonical_id).
        If the same passage_id appears with different source_query_ids,
        the first occurrence's source_query_ids are kept.
        
        Args:
            passages: List of CorpusPassage objects (may contain duplicates).
        
        Returns:
            Deduplicated list of CorpusPassage objects.
        """
        self._seen.clear()
        raw_count = len(passages)
        
        for passage in passages:
            pid = passage.passage_id
            if pid not in self._seen:
                self._seen[pid] = passage
        
        unique_passages = list(self._seen.values())
        unique_count = len(unique_passages)
        dup_count = raw_count - unique_count
        dup_pct = (dup_count / raw_count * 100) if raw_count > 0 else 0.0

        lang_unique: dict[str, int] = {}
        for p in unique_passages:
            lang_unique[p.lang] = lang_unique.get(p.lang, 0) + 1

        self._stats = DedupStats(
            raw_count=raw_count,
            unique_count=unique_count,
            duplicate_count=dup_count,
            duplicate_percentage=dup_pct,
            lang_unique=lang_unique,
        )
        
        logger.info(
            "dedup_complete",
            raw=raw_count,
            unique=unique_count,
            dups=dup_count,
            dup_pct=f"{dup_pct:.1f}%",
        )
        
        return unique_passages
    
    @property
    def stats(self) -> DedupStats:
        if self._stats is None:
            raise RuntimeError("Call deduplicate() before accessing stats")
        return self._stats


def deduplicate_passages(
    passages: list[CorpusPassage],
) -> tuple[list[CorpusPassage], DedupStats]:
    """
    Convenience function: deduplicate and return both passages and stats.
    """
    dedup = Deduplicator()
    unique = dedup.deduplicate(passages)
    return unique, dedup.stats
