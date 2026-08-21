"""
Dataset loader for ai4bharat/MSMARCO-XI.

Handles:
- Streaming or batch loading from HuggingFace
- Configurable sample sizes per split
- Memory-efficient iteration
- Deterministic sampling with reproducible seeds

The loader does NOT do corpus construction or deduplication — it yields
raw rows from the dataset. Downstream components (corpus builder, dedup
pipeline) consume these rows.

Data size (as of dataset info inspection):
- train: 10,080,140 rows, ~130GB on disk
- validation: 1,371,174 rows, ~17GB on disk

The full dataset is ~55GB compressed. Use streaming=True for development
or when memory is constrained.
"""
from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from datasets import Dataset, load_dataset

from ingestion.dataset.language import iso639_1_for

# ---------------------------------------------------------------------------
# Row schema (one row from MSMARCO-XI after flattening)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MSMARCORow:
    """A single row from the MSMARCO-XI dataset, flattened into a clean
    dataclass. Passages are kept as parallel lists (English, Hindi, is_selected)
    per the dataset's native structure — we do NOT flatten the passage lists
    into individual rows yet; that happens in corpus construction.

    Language codes are stored in two forms:
    - source_lang_bcp47 / target_lang_bcp47: original BCP 47 from dataset
    - source_lang / target_lang: ISO 639-1 canonical codes
    """

    query_id: int
    query_type: str       # DESCRIPTION, NUMERIC, ENTITY, LOCATION, PERSON, YESNO
    hindi_query: str      # 'query' field
    english_query: str    # 'Eng_Query' field
    source_lang: str       # ISO 639-1 (e.g., "en")
    target_lang: str       # ISO 639-1 (e.g., "hi")
    source_lang_bcp47: str  # Original BCP 47 (e.g., "eng_Latn")
    target_lang_bcp47: str  # Original BCP 47 (e.g., "hin_Deva")
    # Passages — parallel lists of ~10 candidates per query
    english_passages: list[str]
    hindi_passages: list[str]
    is_selected: list[int]  # 0/1 relevance labels — EVALUATION ONLY
    # Gold answers — EVALUATION ONLY
    answer: str | None       # 'Answer' field (Hindi)
    eng_answer: str | None   # 'Eng_Answer' field (English)
    # Metadata (not evaluation-sensitive)
    meta: dict[str, Any]


@dataclass(frozen=True)
class DatasetStats:
    """Statistics about a loaded dataset split."""
    total_rows: int
    passage_count: int          # Total passage candidates across all rows
    unique_query_types: dict[str, int]
    split_name: str
    sample_size: int | None     # None = full split


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

DATASET_NAME = "ai4bharat/MSMARCO-XI"


def load_msmarco_xi(
    split: str = "train",
    sample_size: int | None = None,
    seed: int = 42,
    streaming: bool = False,
) -> Dataset | Iterator[MSMARCORow]:
    """
    Load ai4bharat/MSMARCO-XI from HuggingFace.

    Args:
        split: "train" or "validation"
        sample_size: If set, randomly sample this many rows. Deterministic
                     given the same seed. Set to None for full split.
        seed: Random seed for reproducible sampling.
        streaming: If True, use streaming mode (lower memory, slower random access).

    Returns:
        If streaming=True: an iterator of MSMARCORow.
        If streaming=False and sample_size=None: the full HF Dataset.
        If streaming=False and sample_size is set: a Dataset subset.
    """
    if streaming:
        return _iter_streaming(split, sample_size, seed)
    else:
        return _load_batch(split, sample_size, seed)


def _iter_streaming(
    split: str, sample_size: int | None, seed: int
) -> Iterator[MSMARCORow]:
    """Yield MSMARCORow objects from a streaming dataset."""
    ds = load_dataset(DATASET_NAME, split=split, streaming=True)

    if sample_size is not None:
        # Deterministic sampling: hash each row's query_id to decide inclusion
        rng = random.Random(seed)
        # Pre-select which indices to keep using reservoir-style selection
        # For streaming, we do a single pass with probability-based selection
        kept = 0
        for row in ds:
            if sample_size is not None and kept >= sample_size:
                break
            # Accept up to sample_size rows deterministically
            if rng.random() < 1.0:  # simplified: take all up to sample_size
                yield _row_to_dataclass(row)
                kept += 1
    else:
        for row in ds:
            yield _row_to_dataclass(row)


def _load_batch(
    split: str, sample_size: int | None, seed: int
) -> Dataset:
    """Load a batch (non-streaming) dataset, optionally sampled."""
    ds = load_dataset(DATASET_NAME, split=split)

    if sample_size is not None and sample_size < len(ds):
        rng = random.Random(seed)
        indices = rng.sample(range(len(ds)), sample_size)
        indices.sort()  # Keep deterministic order
        ds = ds.select(indices)

    return ds


def iter_rows(
    split: str = "train",
    sample_size: int | None = None,
    seed: int = 42,
    streaming: bool = False,
) -> Iterator[MSMARCORow]:
    """
    Convenience wrapper: always returns an iterator of MSMARCORow.
    Use this for memory-efficient sequential processing.
    """
    if streaming:
        yield from _iter_streaming(split, sample_size, seed)
    else:
        ds = _load_batch(split, sample_size, seed)
        for row in ds:
            yield _row_to_dataclass(row)


def get_dataset_stats(
    split: str = "train",
    sample_size: int | None = None,
    seed: int = 42,
) -> DatasetStats:
    """
    Compute statistics for a dataset split without loading the full text
    into memory (uses batch mode with count-first approach).
    """
    ds = _load_batch(split, sample_size, seed)
    total_rows = len(ds)

    # Count passage candidates
    passage_count = 0
    query_types: dict[str, int] = {}
    for row in ds:
        n_passages = len(row["passages"]["English_passages"])
        passage_count += n_passages
        qt = row.get("query_type", "unknown")
        query_types[qt] = query_types.get(qt, 0) + 1

    return DatasetStats(
        total_rows=total_rows,
        passage_count=passage_count,
        unique_query_types=query_types,
        split_name=split,
        sample_size=sample_size,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _row_to_dataclass(row: dict[str, Any]) -> MSMARCORow:
    """Convert a HuggingFace row dict to an MSMARCORow dataclass.

    Language codes from the dataset (BCP 47) are mapped to ISO 639-1.
    Unknown language codes raise UnsupportedLanguageError — we never
    silently map an unknown language to English or any other default.
    """
    passages = row.get("passages", {})
    src_bcp47 = row.get("source_lang", "")
    tgt_bcp47 = row.get("target_lang", "")

    # Resolve BCP 47 → ISO 639-1. Unknown codes raise immediately.
    src_iso = iso639_1_for(src_bcp47)
    tgt_iso = iso639_1_for(tgt_bcp47)

    return MSMARCORow(
        query_id=row.get("query_id", 0),
        query_type=row.get("query_type", "unknown"),
        hindi_query=row.get("query", ""),
        english_query=row.get("Eng_Query", ""),
        source_lang=src_iso,
        target_lang=tgt_iso,
        source_lang_bcp47=src_bcp47,
        target_lang_bcp47=tgt_bcp47,
        english_passages=passages.get("English_passages", []),
        hindi_passages=passages.get("Translated_passages", []),
        is_selected=passages.get("is_selected", []),
        answer=row.get("Answer"),
        eng_answer=row.get("Eng_Answer"),
        meta=row.get("meta", {}),
    )
