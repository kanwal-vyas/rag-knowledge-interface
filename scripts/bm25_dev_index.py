"""
Build a small BM25 development index from real MSMARCO-XI data.

Measures:
- Corpus statistics
- Index build time
- Query latency (P50/P70/P95/P100)
- Basic retrieval quality against gold passages
- Hindi-only vs Hindi+English experiment
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.retrieval import Language, Query, RetrievalMode
from ingestion.deduplication.canonical_id import canonical_passage_id
from ingestion.normalization.text import normalize_text
from retrieval.sparse.bm25_index import BM25Index
from retrieval.sparse.bm25_retriever import BM25SparseRetriever


def load_rows(parquet_path: Path, max_rows: int = 1000) -> list[dict]:
    """Load rows from parquet file using pyarrow batch iteration.

    The parquet files have a nested struct column ``passages`` with fields
    ``English_passages``, ``Translated_passages``, and ``is_selected``.
    ``batch.column('passages')[i].as_py()`` returns this as a plain dict,
    which avoids the flattening issues with fastparquet/pandas.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(parquet_path))
    rows: list[dict] = []
    for batch in pf.iter_batches(batch_size=max_rows):
        # Only read max_rows worth of rows from the first batch
        n = min(batch.num_rows, max_rows - len(rows))
        if n <= 0:
            break
        passages_col = batch.column("passages")
        query_col = batch.column("query")
        qt_col = batch.column("query_type")
        sl_col = batch.column("source_lang")
        tl_col = batch.column("target_lang")
        for i in range(n):
            rows.append({
                "query": query_col[i].as_py() or "",
                "query_type": qt_col[i].as_py() or "",
                "source_lang": sl_col[i].as_py() or "",
                "target_lang": tl_col[i].as_py() or "",
                "passages": passages_col[i].as_py() or {},
            })
    return rows[:max_rows]


def build_index_from_rows(rows: list[dict]) -> tuple[BM25Index, dict]:
    """Build BM25 index from MSMARCO-XI rows."""
    index = BM25Index(use_bigrams=True)
    stats: dict = {"total_rows": len(rows), "indexed_passages": 0, "languages": {}}

    for row in rows:
        passages = row.get("passages", {})
        en_passages = passages.get("English_passages", []) or []
        hi_passages = passages.get("Translated_passages", []) or []

        for text in en_passages:
            if isinstance(text, str) and text.strip():
                pid = canonical_passage_id(text)
                index.add_document(pid, normalize_text(text), "en")
                stats["indexed_passages"] += 1
                stats["languages"]["en"] = stats["languages"].get("en", 0) + 1

        for text in hi_passages:
            if isinstance(text, str) and text.strip():
                pid = canonical_passage_id(text)
                index.add_document(pid, normalize_text(text), "hi")
                stats["indexed_passages"] += 1
                stats["languages"]["hi"] = stats["languages"].get("hi", 0) + 1

    return index, stats


def measure_latency(
    index: BM25Index,
    queries: list[str],
    lang: str = "en",
    top_k: int = 10,
    n_runs: int = 100,
) -> dict:
    """Measure BM25 query latency."""
    retriever = BM25SparseRetriever(index=index)
    latencies: list[float] = []

    for _ in range(n_runs):
        for q in queries:
            query_obj = Query(query_text=q, lang=Language(lang))
            start = time.perf_counter()
            retriever.retrieve(query_obj, RetrievalMode.CROSS_LINGUAL, top_k)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

    latencies.sort()
    return {
        "count": len(latencies),
        "p50": latencies[len(latencies) // 2],
        "p70": latencies[int(len(latencies) * 0.7)],
        "p95": latencies[int(len(latencies) * 0.95)],
        "p100": latencies[-1],
        "mean": statistics.mean(latencies),
    }


def evaluate_retrieval(
    index: BM25Index,
    rows: list[dict],
    mode: RetrievalMode,
    lang: str,
    top_k: int = 10,
) -> dict:
    """Evaluate BM25 retrieval against gold passages."""
    retriever = BM25SparseRetriever(index=index)
    recall_at_1, recall_at_5, recall_at_10 = 0, 0, 0
    mrr = 0.0
    ndcg_scores: list[float] = []
    n_queries = 0
    n_covered = 0  # queries with at least one gold passage in corpus

    for row in rows:
        query_text = row.get("query", "")
        if not query_text or not query_text.strip():
            continue

        query_obj = Query(query_text=query_text, lang=Language(lang))
        results = retriever.retrieve(query_obj, mode, top_k)

        # Get gold passages (is_selected == 1)
        passages = row.get("passages", {})
        is_selected = passages.get("is_selected", []) or []
        en_passages = passages.get("English_passages", []) or []
        hi_passages = passages.get("Translated_passages", []) or []

        gold_ids: set[str] = set()
        for i, sel in enumerate(is_selected):
            if sel == 1 or sel == 1.0:
                if i < len(en_passages) and en_passages[i]:
                    gold_ids.add(canonical_passage_id(en_passages[i]))
                if i < len(hi_passages) and hi_passages[i]:
                    gold_ids.add(canonical_passage_id(hi_passages[i]))

        if not gold_ids:
            continue

        n_queries += 1

        # Check if any gold passage is in the corpus
        corpus_ids = {doc.passage_id for doc in index._documents}
        if not gold_ids.intersection(corpus_ids):
            continue
        n_covered += 1

        # Compute metrics
        retrieved_ids = [r.passage.passage_id for r in results]

        hits_at_1 = 1 if retrieved_ids[:1] and retrieved_ids[0] in gold_ids else 0
        hits_at_5 = 1 if any(pid in gold_ids for pid in retrieved_ids[:5]) else 0
        hits_at_10 = 1 if any(pid in gold_ids for pid in retrieved_ids[:10]) else 0

        recall_at_1 += hits_at_1
        recall_at_5 += hits_at_5
        recall_at_10 += hits_at_10

        # MRR
        for rank, pid in enumerate(retrieved_ids, 1):
            if pid in gold_ids:
                mrr += 1.0 / rank
                break

        # nDCG@10
        relevance = [1.0 if pid in gold_ids else 0.0 for pid in retrieved_ids[:10]]
        dcg = sum(rel / (i + 1) for i, rel in enumerate(relevance))
        ideal_relevance = sorted(relevance, reverse=True)
        idcg = sum(rel / (i + 1) for i, rel in enumerate(ideal_relevance))
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    if n_queries == 0:
        return {"error": "no valid queries"}

    return {
        "n_queries": n_queries,
        "n_covered": n_covered,
        "coverage_rate": n_covered / n_queries if n_queries > 0 else 0.0,
        "recall@1": recall_at_1 / n_covered if n_covered > 0 else 0.0,
        "recall@5": recall_at_5 / n_covered if n_covered > 0 else 0.0,
        "recall@10": recall_at_10 / n_covered if n_covered > 0 else 0.0,
        "mrr": mrr / n_covered if n_covered > 0 else 0.0,
        "ndcg@10": statistics.mean(ndcg_scores) if ndcg_scores else 0.0,
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="BM25 development index benchmark")
    parser.add_argument("--train-rows", type=int, default=500,
                        help="Number of training rows to index (default: 500)")
    parser.add_argument("--val-rows", type=int, default=200,
                        help="Number of validation rows for evaluation (default: 200)")
    args = parser.parse_args()

    print("=" * 60)
    print("BM25 DEVELOPMENT INDEX \u2014 PHASE 7 BENCHMARK")
    print("=" * 60)

    # --- Load data ---
    train_path = Path("data/cache/train/hintrain.parquet")
    val_path = Path("data/cache/validation/hinval.parquet")

    if not train_path.exists():
        print(f"ERROR: Train data not found at {train_path}")
        sys.exit(1)

    print(f"\nLoading {args.train_rows} train rows from {train_path}...")
    t0 = time.perf_counter()
    train_rows = load_rows(train_path, max_rows=args.train_rows)
    load_time = time.perf_counter() - t0
    print(f"  Loaded {len(train_rows)} rows in {load_time:.1f}s")

    # Debug: inspect first row passages
    if train_rows:
        sample = train_rows[0]
        p = sample.get("passages", {})
        en_sample = p.get("English_passages", []) or []
        hi_sample = p.get("Translated_passages", []) or []
        print(f"  English passages per row: {len(en_sample)}")
        print(f"  Hindi passages per row: {len(hi_sample)}")

    # --- Build index ---
    print("\nBuilding BM25 index...")
    index, corpus_stats = build_index_from_rows(train_rows)
    build_stats = index.build()

    print(f"  Documents: {corpus_stats['indexed_passages']}")
    print(f"  Languages: {corpus_stats['languages']}")
    print(f"  Vocab size: {build_stats.vocab_size}")
    print(f"  Build time: {build_stats.build_time_ms:.1f}ms")

    # --- Latency measurement ---
    print("\nMeasuring query latency...")
    en_queries = [
        "artificial intelligence",
        "capital of India",
        "machine learning",
        "natural language processing",
        "climate change",
    ]
    hi_queries = [
        "भारत की राजधानी",
        "कृत्रिम बुद्धिमत्ता",
        "मशीन लर्निंग",
        "जलवायु परिवर्तन",
        "प्राकृतिक भाषा प्रसंस्करण",
    ]

    en_latency = measure_latency(index, en_queries, lang="en", n_runs=50)
    hi_latency = measure_latency(index, hi_queries, lang="hi", n_runs=50)
    en_p50 = en_latency["p50"]
    en_p95 = en_latency["p95"]
    en_p100 = en_latency["p100"]
    print(f"  English P50={en_p50:.3f}ms P95={en_p95:.3f}ms P100={en_p100:.3f}ms")
    hi_p50 = hi_latency["p50"]
    hi_p95 = hi_latency["p95"]
    hi_p100 = hi_latency["p100"]
    print(f"  Hindi   P50={hi_p50:.3f}ms P95={hi_p95:.3f}ms P100={hi_p100:.3f}ms")

    # --- Retrieval evaluation ---
    val_rows: list[dict] = []
    if val_path.exists():
        print(f"\nLoading {args.val_rows} validation rows from {val_path}...")
        val_rows = load_rows(val_path, max_rows=args.val_rows)
        print(f"  Loaded {len(val_rows)} validation rows")

    if val_rows:
        print("\n--- EVALUATION: Hindi query -> Hindi+English (CROSS_LINGUAL) ---")
        eval_cross = evaluate_retrieval(
            index, val_rows, RetrievalMode.CROSS_LINGUAL, "hi", top_k=10
        )
        for k, v in eval_cross.items():
            print(f"  {k}: {v}")

        print("\n--- EVALUATION: Hindi query -> Hindi only (MONOLINGUAL) ---")
        eval_mono = evaluate_retrieval(index, val_rows, RetrievalMode.MONOLINGUAL, "hi", top_k=10)
        for k, v in eval_mono.items():
            print(f"  {k}: {v}")

        print("\n--- HINDI-ONLY vs HINDI+ENGLISH COMPARISON ---")
        if "error" not in eval_cross and "error" not in eval_mono:
            for metric in ["recall@1", "recall@5", "recall@10", "mrr", "ndcg@10"]:
                mono_val = eval_mono.get(metric, 0)
                cross_val = eval_cross.get(metric, 0)
                delta = cross_val - mono_val
                winner = "CROSS" if delta > 0 else "MONO" if delta < 0 else "TIE"
                print(
                    f"  {metric}: mono={mono_val:.4f}"
                    f" cross={cross_val:.4f}"
                    f" delta={delta:+.4f} winner={winner}"
                )
    else:
        print("\n  [SKIP] No validation data available for evaluation")

    # --- Save results ---
    results: dict = {
        "corpus_stats": corpus_stats,
        "build_stats": {
            "build_time_ms": build_stats.build_time_ms,
            "vocab_size": build_stats.vocab_size,
        },
        "en_latency": en_latency,
        "hi_latency": hi_latency,
    }
    if val_rows:
        results["eval_cross_lingual"] = eval_cross
        results["eval_monolingual"] = eval_mono

    output_path = Path("data/bm25_dev_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
