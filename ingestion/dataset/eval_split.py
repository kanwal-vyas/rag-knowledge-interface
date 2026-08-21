"""
Evaluation dataset extraction.

Architecture §4 (frozen — three-way split):
- Tuning/calibration queries: train-split (ALL tunable parameters)
- Frozen benchmark queries: validation-split (final reporting, touched ONCE)
- Corpus: train + validation combined, NOT query-partitioned

Architecture §5 (frozen — leakage audit):
- is_selected, Answer, Eng_Answer, source_query_ids are evaluation-only
- These MUST NOT enter production Qdrant payloads, embedding text,
  BM25 documents, generation context, or prompts

This module extracts and stores evaluation data as structurally separate
files/objects. The evaluation/ package consumes these, but the production
path (app/, retrieval/, embeddings/, generation/, guardrails/) never imports
from here.
"""
from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import structlog

from ingestion.dataset.loader import MSMARCORow
from ingestion.deduplication.canonical_id import canonical_passage_id

logger = structlog.get_logger(__name__)

# Evaluation-only fields — must never appear in production models
EVALUATION_ONLY_FIELDS = frozenset({
    "is_selected",
    "Answer",
    "Eng_Answer",
    "source_query_ids",
    "query_id",  # As passage metadata (Audit §5)
})


@dataclass
class GoldLabel:
    """A single (query_id, passage_id, is_selected) evaluation triple.
    
    passage_id uses canonical content-hash IDs — same hash space as
    corpus dedup (Audit §6, freeze decision 4).
    """
    query_id: int
    passage_id: str
    is_selected: bool


@dataclass
class EvalQuery:
    """An evaluation query with all its metadata."""
    query_id: int
    query_type: str         # DESCRIPTION, NUMERIC, ENTITY, LOCATION, PERSON, YESNO
    hindi_query: str
    english_query: str
    source_lang: str
    target_lang: str
    gold_labels: list[GoldLabel] = field(default_factory=list)
    answer: str | None = None      # Hindi gold answer (EVALUATION ONLY)
    eng_answer: str | None = None  # English gold answer (EVALUATION ONLY)


@dataclass
class EvalDataset:
    """Complete evaluation dataset for a split."""
    pool: str   # "tuning" or "benchmark"
    queries: list[EvalQuery] = field(default_factory=list)
    
    def gold_for_query(self, query_id: int) -> list[str]:
        """Get canonical passage_ids for gold-relevant passages of a query."""
        for q in self.queries:
            if q.query_id == query_id:
                return [gl.passage_id for gl in q.gold_labels if gl.is_selected]
        return []


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_eval_dataset(
    rows: list[MSMARCORow] | Iterator[MSMARCORow],
    pool: str,
) -> EvalDataset:
    """
    Extract evaluation data from raw MSMARCO-XI rows.
    
    This creates the structurally isolated evaluation dataset. The gold
    passage_ids are computed using the SAME canonical_passage_id function
    used for corpus dedup — Audit §6, freeze decision 4.
    
    Args:
        rows: Iterable of MSMARCORow objects.
        pool: "tuning" (train-split) or "benchmark" (validation-split).
    
    Returns:
        EvalDataset with gold labels, answers, and query metadata.
    """
    dataset = EvalDataset(pool=pool)
    
    for row in rows:
        # Build gold labels using canonical passage_ids
        gold_labels = []
        n_passages = len(row.english_passages)
        
        for i in range(n_passages):
            # English passage gold label
            if i < len(row.english_passages):
                en_text = row.english_passages[i]
                if en_text:
                    pid = canonical_passage_id(en_text)
                    is_sel = bool(row.is_selected[i]) if i < len(row.is_selected) else False
                    gold_labels.append(GoldLabel(
                        query_id=row.query_id,
                        passage_id=pid,
                        is_selected=is_sel,
                    ))
            
            # Hindi passage gold label
            if i < len(row.hindi_passages):
                hi_text = row.hindi_passages[i]
                if hi_text:
                    pid = canonical_passage_id(hi_text)
                    is_sel = bool(row.is_selected[i]) if i < len(row.is_selected) else False
                    gold_labels.append(GoldLabel(
                        query_id=row.query_id,
                        passage_id=pid,
                        is_selected=is_sel,
                    ))
        
        eval_query = EvalQuery(
            query_id=row.query_id,
            query_type=row.query_type,
            hindi_query=row.hindi_query,
            english_query=row.english_query,
            source_lang=row.source_lang,
            target_lang=row.target_lang,
            gold_labels=gold_labels,
            answer=row.answer,        # EVALUATION ONLY
            eng_answer=row.eng_answer, # EVALUATION ONLY
        )
        dataset.queries.append(eval_query)
    
    logger.info(
        "eval_dataset_extracted",
        pool=pool,
        queries=len(dataset.queries),
        total_gold_labels=sum(len(q.gold_labels) for q in dataset.queries),
        selected_labels=sum(
            sum(1 for gl in q.gold_labels if gl.is_selected)
            for q in dataset.queries
        ),
    )
    
    return dataset


def save_eval_dataset(dataset: EvalDataset, output_dir: Path) -> None:
    """
    Persist evaluation dataset to disk as structured files.
    
    Creates:
    - {pool}_queries.json: query metadata (no evaluation-only labels)
    - {pool}_gold_labels.csv: (query_id, passage_id, is_selected)
    - {pool}_answers.csv: (query_id, answer, eng_answer)
    
    These files are loaded by the evaluation harness, never by the
    production serving path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Queries (without evaluation fields)
    queries_file = output_dir / f"{dataset.pool}_queries.json"
    queries_data = []
    for q in dataset.queries:
        queries_data.append({
            "query_id": q.query_id,
            "query_type": q.query_type,
            "hindi_query": q.hindi_query,
            "english_query": q.english_query,
            "source_lang": q.source_lang,
            "target_lang": q.target_lang,
        })
    json_str = json.dumps(queries_data, ensure_ascii=False, indent=2)
    queries_file.write_text(json_str, encoding="utf-8")
    
    # Gold labels (the core evaluation data)
    labels_file = output_dir / f"{dataset.pool}_gold_labels.csv"
    with open(labels_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "passage_id", "is_selected"])
        for q in dataset.queries:
            for gl in q.gold_labels:
                writer.writerow([gl.query_id, gl.passage_id, int(gl.is_selected)])
    
    # Answers (EVALUATION ONLY — never used in production)
    answers_file = output_dir / f"{dataset.pool}_answers.csv"
    with open(answers_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "answer", "eng_answer"])
        for q in dataset.queries:
            writer.writerow([q.query_id, q.answer or "", q.eng_answer or ""])
    
    logger.info(
        "eval_dataset_saved",
        pool=dataset.pool,
        output_dir=str(output_dir),
        queries_file=str(queries_file),
        labels_file=str(labels_file),
        answers_file=str(answers_file),
    )


def load_eval_dataset(input_dir: Path, pool: str) -> EvalDataset:
    """
    Load a previously-saved evaluation dataset from disk.
    """
    queries_file = input_dir / f"{pool}_queries.json"
    labels_file = input_dir / f"{pool}_gold_labels.csv"
    answers_file = input_dir / f"{pool}_answers.csv"
    
    # Load queries
    queries_data = json.loads(queries_file.read_text(encoding="utf-8"))
    query_map: dict[int, EvalQuery] = {}
    for qd in queries_data:
        eq = EvalQuery(
            query_id=qd["query_id"],
            query_type=qd["query_type"],
            hindi_query=qd["hindi_query"],
            english_query=qd["english_query"],
            source_lang=qd["source_lang"],
            target_lang=qd["target_lang"],
        )
        query_map[eq.query_id] = eq
    
    # Load gold labels
    with open(labels_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = int(row["query_id"])
            if qid in query_map:
                query_map[qid].gold_labels.append(GoldLabel(
                    query_id=qid,
                    passage_id=row["passage_id"],
                    is_selected=bool(int(row["is_selected"])),
                ))
    
    # Load answers
    with open(answers_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = int(row["query_id"])
            if qid in query_map:
                query_map[qid].answer = row["answer"] or None
                query_map[qid].eng_answer = row["eng_answer"] or None
    
    dataset = EvalDataset(
        pool=pool,
        queries=list(query_map.values()),
    )
    
    logger.info(
        "eval_dataset_loaded",
        pool=pool,
        queries=len(dataset.queries),
    )
    
    return dataset
