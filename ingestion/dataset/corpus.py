"""
Corpus construction from MSMARCO-XI.

Architecture §3 (frozen):
- Corpus is a broad pool spanning far more queries than the benchmark set.
- Drawn from train + validation combined.
- Coverage-awareness is a post-hoc backstop, not a construction method.
- After building the broad-pool corpus, check which benchmark queries'
  gold passages are missing and add only the missing gold passages.
- Report coverage rate per tier — expected to be <100% at small tiers.

Key invariant: the corpus is NOT built by indexing only the passages
associated with benchmark queries. That would create an artificially
easy retrieval task (Audit §3.1).

Corpus tiers (T1-T5 naming from Architecture, exact sizes TBD):
Sizes are determined experimentally per §22. The pipeline supports
configurable tier sizes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from ingestion.dataset.loader import MSMARCORow
from ingestion.deduplication.canonical_id import canonical_passage_id
from ingestion.normalization.text import normalize_text

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Corpus entry — a single passage ready for indexing
# ---------------------------------------------------------------------------

@dataclass
class CorpusPassage:
    """A single deduplicated, indexed passage in the production corpus.
    
    Contains ONLY production-safe fields. No is_selected, no Answer,
    no Eng_Answer, no source_query_ids — those live in evaluation/ only.
    """
    passage_id: str     # Canonical content-hash ID
    text: str           # Normalized passage text
    lang: str           # "hi" or "en"
    # Off-line traceability only — NOT in Qdrant
    source_query_ids: list[int] = field(default_factory=list)


@dataclass
class CorpusStats:
    """Statistics about a constructed corpus."""
    raw_passages: int
    unique_passages: int
    duplicate_count: int
    duplicate_percentage: float
    lang_counts: dict[str, int]  # ISO 639-1 → raw count (pre-dedup)
    lang_unique_counts: dict[str, int]  # ISO 639-1 → unique count (post-dedup)
    tier_name: str
    query_count: int    # Number of source queries from which passages were drawn


# ---------------------------------------------------------------------------
# Default corpus tier definitions (configurable)
# ---------------------------------------------------------------------------

# These are initial estimates — the actual tier sizes should be determined
# experimentally per Architecture §22. The pipeline supports any tier size.
DEFAULT_TIERS: dict[str, dict[str, int]] = {
    "T1": {"target_passages": 50_000, "source_query_rows": 5_000},
    "T2": {"target_passages": 150_000, "source_query_rows": 15_000},
    "T3": {"target_passages": 500_000, "source_query_rows": 50_000},
    "T4": {"target_passages": 1_500_000, "source_query_rows": 150_000},
    "T5": {"target_passages": 5_000_000, "source_query_rows": 500_000},
}


# ---------------------------------------------------------------------------
# Corpus builder
# ---------------------------------------------------------------------------

class CorpusBuilder:
    """
    Builds a retrieval corpus from MSMARCO-XI rows.
    
    Usage:
        builder = CorpusBuilder(tier_name="T3")
        for row in msmarco_rows:
            builder.add_row(row)
        corpus, stats = builder.build()
    """
    
    def __init__(
        self,
        tier_name: str = "T3",
        target_passages: int | None = None,
        source_query_rows: int | None = None,
    ) -> None:
        if tier_name in DEFAULT_TIERS and target_passages is None:
            tier = DEFAULT_TIERS[tier_name]
            target_passages = tier["target_passages"]
            source_query_rows = tier["source_query_rows"]
        
        self.tier_name = tier_name
        self.target_passages = target_passages or 500_000
        self.source_query_rows = source_query_rows or 50_000
        
        # Passage deduplication map: normalized_text_hash -> CorpusPassage
        self._passages: dict[str, CorpusPassage] = {}
        self._raw_count = 0
        self._query_count = 0
        self._lang_counts: dict[str, int] = {}  # ISO 639-1 → raw count
    
    def add_row(self, row: MSMARCORow) -> None:
        """
        Extract and index passages from a single MSMARCO-XI row.

        Each row has parallel lists of ~10 English and target-language
        candidate passages. We index both, creating separate CorpusPassage
        objects for each language variant.
        """
        self._query_count += 1
        n_passages = len(row.english_passages)

        # Resolve languages from the row's ISO codes
        src_lang = row.source_lang  # ISO 639-1 (e.g., "en")
        tgt_lang = row.target_lang  # ISO 639-1 (e.g., "hi")

        for i in range(n_passages):
            # Source-language passage (e.g., English)
            if i < len(row.english_passages):
                en_text = row.english_passages[i]
                if en_text and en_text.strip():
                    self._add_passage(en_text, src_lang, row.query_id)
                    self._lang_counts[src_lang] = self._lang_counts.get(src_lang, 0) + 1

            # Target-language passage (translated, e.g., Hindi)
            if i < len(row.hindi_passages):
                hi_text = row.hindi_passages[i]
                if hi_text and hi_text.strip():
                    self._add_passage(hi_text, tgt_lang, row.query_id)
                    self._lang_counts[tgt_lang] = self._lang_counts.get(tgt_lang, 0) + 1

        self._raw_count += n_passages * 2  # Both languages
    
    def _add_passage(self, text: str, lang: str, query_id: int) -> None:
        """Add a single passage, deduplicating by canonical passage_id."""
        passage_id = canonical_passage_id(text)
        
        if passage_id in self._passages:
            # Duplicate — extend traceability but don't add new entry
            existing = self._passages[passage_id]
            if query_id not in existing.source_query_ids:
                existing.source_query_ids.append(query_id)
        else:
            self._passages[passage_id] = CorpusPassage(
                passage_id=passage_id,
                text=normalize_text(text),
                lang=lang,
                source_query_ids=[query_id],
            )
    
    def has_capacity(self) -> bool:
        """Check if the corpus has reached its target size."""
        return len(self._passages) < self.target_passages
    
    def query_limit_reached(self) -> bool:
        """Check if we've processed enough query rows."""
        return self._query_count >= self.source_query_rows
    
    def build(self) -> tuple[list[CorpusPassage], CorpusStats]:
        """
        Finalize the corpus and return it with statistics.
        
        Returns:
            (passages, stats) where passages is the deduplicated list
            and stats contains the build statistics.
        """
        passages = list(self._passages.values())
        unique_count = len(passages)
        dup_count = max(0, self._raw_count - unique_count)
        dup_pct = (dup_count / self._raw_count * 100) if self._raw_count > 0 else 0.0

        # Unique counts per language (post-dedup)
        lang_unique: dict[str, int] = {}
        for p in passages:
            lang_unique[p.lang] = lang_unique.get(p.lang, 0) + 1

        stats = CorpusStats(
            raw_passages=self._raw_count,
            unique_passages=unique_count,
            duplicate_count=dup_count,
            duplicate_percentage=dup_pct,
            lang_counts=dict(self._lang_counts),
            lang_unique_counts=lang_unique,
            tier_name=self.tier_name,
            query_count=self._query_count,
        )
        
        logger.info(
            "corpus_built",
            tier=self.tier_name,
            raw=self._raw_count,
            unique=unique_count,
            dup_pct=f"{dup_pct:.1f}%",
            queries=self._query_count,
        )
        
        return passages, stats
    
    def clear(self) -> None:
        """Reset the builder for a new corpus."""
        self._passages.clear()
        self._raw_count = 0
        self._query_count = 0
        self._lang_counts.clear()


# ---------------------------------------------------------------------------
# Coverage checker (post-hoc backstop, Audit §3.2)
# ---------------------------------------------------------------------------

def check_corpus_coverage(
    corpus_passages: list[CorpusPassage],
    benchmark_gold: dict[int, list[str]],  # query_id -> list of gold passage_ids
) -> dict[str, float]:
    """
    Post-hoc coverage check: which benchmark queries have their gold
    passages present in the corpus?
    
    Args:
        corpus_passages: The built corpus.
        benchmark_gold: Mapping from query_id to list of gold passage_ids
                       (using canonical passage_ids from gold-set construction).
    
    Returns:
        Dict with coverage statistics.
    """
    corpus_ids = {p.passage_id for p in corpus_passages}
    
    total_queries = len(benchmark_gold)
    fully_covered = 0
    partially_covered = 0
    total_gold = 0
    found_gold = 0
    
    for _qid, gold_ids in benchmark_gold.items():
        total_gold += len(gold_ids)
        found = sum(1 for gid in gold_ids if gid in corpus_ids)
        found_gold += found
        
        if found == len(gold_ids):
            fully_covered += 1
        elif found > 0:
            partially_covered += 1
    
    coverage_rate = found_gold / total_gold if total_gold > 0 else 0.0
    full_coverage_rate = fully_covered / total_queries if total_queries > 0 else 0.0
    
    return {
        "total_queries": total_queries,
        "fully_covered_queries": fully_covered,
        "partially_covered_queries": partially_covered,
        "uncovered_queries": total_queries - fully_covered - partially_covered,
        "coverage_rate": coverage_rate,
        "full_coverage_rate": full_coverage_rate,
        "total_gold_passages": total_gold,
        "found_gold_passages": found_gold,
    }
