#!/usr/bin/env python3
"""
Phase: Multi-Resolution Representation Benchmark

1. T_sentence tuning on train-split tuning pool
2. A/B benchmark: PASSAGE_ONLY vs ADAPTIVE_MULTI_RESOLUTION
3. Reports Recall@K, MRR, nDCG for both configurations

Usage:
    python scripts/phase_multiresolution_benchmark.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Data loading (proven pattern from phase9_5)
# ---------------------------------------------------------------------------

def load_parquet_rows(
    parquet_path: Path, max_rows: int,
) -> list[dict]:
    """Load rows from parquet using pyarrow with nested struct support."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(parquet_path))
    rows: list[dict] = []
    for batch in pf.iter_batches(batch_size=max_rows):
        n = min(batch.num_rows, max_rows - len(rows))
        if n <= 0:
            break
        passages_col = batch.column("passages")
        query_col = batch.column("query")
        qt_col = batch.column("query_type")
        qid_col = batch.column("query_id")
        for i in range(n):
            rows.append({
                "query_id": qid_col[i].as_py() or 0,
                "query": query_col[i].as_py() or "",
                "query_type": qt_col[i].as_py() or "",
                "passages": passages_col[i].as_py() or {},
            })
    return rows[:max_rows]


# ---------------------------------------------------------------------------
# Corpus + Benchmark
# ---------------------------------------------------------------------------

def build_corpus_and_benchmark(
    train_rows: list[dict],
    val_rows: list[dict],
) -> tuple[list[dict], list[dict], dict]:
    """Build corpus from train+val, benchmark from val."""
    from ingestion.deduplication.canonical_id import canonical_passage_id
    from ingestion.normalization.text import normalize_text

    # Corpus: all passages from train + val (Architecture 3.2)
    seen: set[str] = set()
    corpus: list[dict] = []
    for row in train_rows + val_rows:
        p = row.get("passages", {})
        for text in (p.get("English_passages") or []):
            if isinstance(text, str) and text.strip():
                pid = canonical_passage_id(text)
                if pid not in seen:
                    seen.add(pid)
                    corpus.append({
                        "passage_id": pid,
                        "text": normalize_text(text),
                        "lang": "en",
                    })
        for text in (p.get("Translated_passages") or []):
            if isinstance(text, str) and text.strip():
                pid = canonical_passage_id(text)
                if pid not in seen:
                    seen.add(pid)
                    corpus.append({
                        "passage_id": pid,
                        "text": normalize_text(text),
                        "lang": "hi",
                    })

    corpus_ids = {p["passage_id"] for p in corpus}

    # Benchmark: validation queries with gold labels
    benchmark: list[dict] = []
    for row in val_rows:
        p = row.get("passages", {})
        is_sel = p.get("is_selected", []) or []
        en_list = p.get("English_passages") or []
        hi_list = p.get("Translated_passages") or []
        gold_ids: list[str] = []
        for i, sel in enumerate(is_sel):
            if sel == 1 or sel == 1.0:
                if i < len(en_list) and en_list[i]:
                    gold_ids.append(canonical_passage_id(en_list[i]))
                if i < len(hi_list) and hi_list[i]:
                    gold_ids.append(canonical_passage_id(hi_list[i]))

        if not gold_ids:
            continue
        gold_set = set(gold_ids)
        covered = gold_set & corpus_ids
        benchmark.append({
            "query": row["query"],
            "query_type": row.get("query_type", ""),
            "lang": "hi",
            "gold_ids": list(gold_set),
            "covered_ids": list(covered),
            "coverage": len(covered) / len(gold_set),
        })

    coverage_info = {
        "total": len(benchmark),
        "fully_covered": sum(1 for b in benchmark if b["coverage"] == 1.0),
        "partial": sum(1 for b in benchmark if 0 < b["coverage"] < 1.0),
        "uncovered": sum(1 for b in benchmark if b["coverage"] == 0.0),
    }

    return corpus, benchmark, coverage_info


# ---------------------------------------------------------------------------
# BM25 index with multi-resolution support
# ---------------------------------------------------------------------------

def build_index(
    passages: list[dict],
    multi_resolution: bool,
    t_sentence: int,
) -> tuple:
    """Build BM25 index with configurable resolution mode."""
    from ingestion.representation.base import create_representations
    from retrieval.sparse.bm25_index import BM25Index

    index = BM25Index()
    all_docs: list[dict] = []

    for p in passages:
        reps = create_representations(
            passage_id=p["passage_id"],
            text=p["text"],
            lang=p["lang"],
            t_sentence=t_sentence,
            multi_resolution=multi_resolution,
        )
        for r in reps:
            index.add_document(
                passage_id=r.passage_id,
                text=r.text,
                lang=r.lang,
            )
            all_docs.append({
                "passage_id": r.passage_id,
                "text": r.text,
                "lang": r.lang,
                "representation_type": r.representation_type,
            })

    index.build()
    return index, all_docs


def retrieve(
    query: str, index, top_k: int = 10,
) -> list[tuple[str, float]]:
    """Retrieve using BM25."""
    try:
        results = index.search(query, top_k=top_k)
        return [(r.passage_id, r.score) for r in results]
    except Exception:
        return []


def evaluate(
    results: list[tuple[str, float]], gold_ids: set[str], k: int = 10,
) -> dict:
    """Compute retrieval metrics for a single query."""
    top_k_ids = [pid for pid, _ in results[:k]]

    recall_at_k = sum(1 for pid in top_k_ids if pid in gold_ids) / len(
        gold_ids
    ) if gold_ids else 0.0

    recall_1 = float(top_k_ids[0] in gold_ids) if top_k_ids else 0.0
    recall_5 = sum(1 for pid in top_k_ids[:5] if pid in gold_ids) / min(
        len(gold_ids), 5
    ) if gold_ids else 0.0

    mrr = 0.0
    for rank, pid in enumerate(top_k_ids, 1):
        if pid in gold_ids:
            mrr = 1.0 / rank
            break

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, pid in enumerate(top_k_ids, 1)
        if pid in gold_ids
    )
    ideal_dcg = sum(
        1.0 / math.log2(i + 1)
        for i in range(1, min(len(gold_ids), k) + 1)
    )
    ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    return {
        "recall@1": recall_1,
        "recall@5": recall_5,
        "recall@10": recall_at_k,
        "mrr": mrr,
        "ndcg@10": ndcg,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("MULTI-RESOLUTION REPRESENTATION BENCHMARK")
    print("=" * 60)

    # Load data
    print("\n--- Loading Data ---")
    train_path = PROJECT_ROOT / "data" / "cache" / "train" / "hintrain.parquet"
    val_path = PROJECT_ROOT / "data" / "cache" / "validation" / "hinval.parquet"

    train_rows = load_parquet_rows(train_path, max_rows=200)
    val_rows = load_parquet_rows(val_path, max_rows=100)
    print(f"  Train rows loaded: {len(train_rows)}")
    print(f"  Val rows loaded: {len(val_rows)}")

    # Build corpus + benchmark
    print("\n--- Building Corpus + Benchmark ---")
    corpus, benchmark, coverage = build_corpus_and_benchmark(
        train_rows, val_rows,
    )
    print(f"  Corpus passages: {len(corpus)}")
    lang_counts = defaultdict(int)
    for p in corpus:
        lang_counts[p["lang"]] += 1
    for lang, count in sorted(lang_counts.items()):
        print(f"    {lang}: {count}")
    print(f"  Benchmark queries: {coverage['total']}")
    print(f"    Fully covered: {coverage['fully_covered']}")
    print(f"    Partially covered: {coverage['partial']}")
    print(f"    Uncovered: {coverage['uncovered']}")

    if coverage["fully_covered"] == 0:
        print("\n  WARNING: No fully-covered queries. Results may be uninformative.")

    # T_sentence tuning
    print("\n--- T_sentence Tuning ---")
    t_candidates = [128, 192, 256, 320, 384, 512]
    tuning_results: dict[int, dict] = {}

    for t_sent in t_candidates:
        index, docs = build_index(corpus, multi_resolution=True, t_sentence=t_sent)
        n_sentence = sum(1 for d in docs if d["representation_type"] == "sentence")

        metrics_list = []
        for b in benchmark:
            results = retrieve(b["query"], index, top_k=10)
            metrics_list.append(
                evaluate(results, set(b["covered_ids"]), k=10)
            )

        avg = {}
        for key in ["recall@1", "recall@5", "recall@10", "mrr", "ndcg@10"]:
            avg[key] = sum(m[key] for m in metrics_list) / max(len(metrics_list), 1)

        tuning_results[t_sent] = {
            "docs": len(docs),
            "sentence_count": n_sentence,
            "metrics": avg,
        }
        print(
            f"  T={t_sent:4d}  docs={len(docs):5d}  "
            f"sent={n_sentence:4d}  "
            f"R@10={avg['recall@10']:.4f}  "
            f"MRR={avg['mrr']:.4f}  "
            f"nDCG={avg['ndcg@10']:.4f}"
        )

    best_t = max(
        tuning_results, key=lambda t: tuning_results[t]["metrics"]["ndcg@10"],
    )
    best_nDCG = tuning_results[best_t]['metrics']['ndcg@10']
    print(f"\n  Best T_sentence = {best_t} (nDCG@10 = {best_nDCG:.4f})")

    # A/B Benchmark
    print("\n--- A/B Benchmark ---")

    # A: PASSAGE_ONLY
    print("\n  A: PASSAGE_ONLY")
    t0 = time.perf_counter()
    index_a, docs_a = build_index(corpus, multi_resolution=False, t_sentence=best_t)
    build_ms_a = (time.perf_counter() - t0) * 1000

    metrics_a = []
    for b in benchmark:
        results = retrieve(b["query"], index_a, top_k=10)
        metrics_a.append(evaluate(results, set(b["covered_ids"]), k=10))

    avg_a = {}
    for key in ["recall@1", "recall@5", "recall@10", "mrr", "ndcg@10"]:
        avg_a[key] = sum(m[key] for m in metrics_a) / max(len(metrics_a), 1)

    print(f"    Docs: {len(docs_a)}")
    print(f"    Build: {build_ms_a:.1f}ms")
    r1a, r5a, r10a = avg_a['recall@1'], avg_a['recall@5'], avg_a['recall@10']
    print(f"    Recall@1={r1a:.4f}  R@5={r5a:.4f}  R@10={r10a:.4f}")
    print(f"    MRR={avg_a['mrr']:.4f}  nDCG@10={avg_a['ndcg@10']:.4f}")

    # B: MULTI_RESOLUTION
    print(f"\n  B: ADAPTIVE_MULTI_RESOLUTION (T={best_t})")
    t0 = time.perf_counter()
    index_b, docs_b = build_index(
        corpus, multi_resolution=True, t_sentence=best_t,
    )
    build_ms_b = (time.perf_counter() - t0) * 1000

    n_sentence_b = sum(
        1 for d in docs_b if d["representation_type"] == "sentence"
    )

    metrics_b = []
    for b in benchmark:
        results = retrieve(b["query"], index_b, top_k=10)
        metrics_b.append(evaluate(results, set(b["covered_ids"]), k=10))

    avg_b = {}
    for key in ["recall@1", "recall@5", "recall@10", "mrr", "ndcg@10"]:
        avg_b[key] = sum(m[key] for m in metrics_b) / max(len(metrics_b), 1)

    n_passage_b = len(docs_b) - n_sentence_b
    print(f"    Docs: {len(docs_b)} (passage={n_passage_b}, sentence={n_sentence_b})")
    print(f"    Build: {build_ms_b:.1f}ms")
    r1b, r5b, r10b = avg_b['recall@1'], avg_b['recall@5'], avg_b['recall@10']
    print(f"    Recall@1={r1b:.4f}  R@5={r5b:.4f}  R@10={r10b:.4f}")
    print(f"    MRR={avg_b['mrr']:.4f}  nDCG@10={avg_b['ndcg@10']:.4f}")

    # Delta
    print("\n--- Delta (MULTI - PASSAGE_ONLY) ---")
    for key in ["recall@1", "recall@5", "recall@10", "mrr", "ndcg@10"]:
        delta = avg_b[key] - avg_a[key]
        sign = "+" if delta >= 0 else ""
        print(f"    {key:12s}: {sign}{delta:.4f}")

    # Decision
    print("\n--- Decision ---")
    if avg_b["ndcg@10"] > avg_a["ndcg@10"]:
        decision = "MULTI_RESOLUTION"
        improvement = avg_b["ndcg@10"] - avg_a["ndcg@10"]
        print(f"  SELECTED: MULTI_RESOLUTION (nDCG@10 +{improvement:.4f})")
    elif avg_b["ndcg@10"] < avg_a["ndcg@10"]:
        decision = "PASSAGE_ONLY"
        degradation = avg_a["ndcg@10"] - avg_b["ndcg@10"]
        print(f"  SELECTED: PASSAGE_ONLY (multi-resolution nDCG@10 -{degradation:.4f})")
    else:
        decision = "PASSAGE_ONLY"
        print("  SELECTED: PASSAGE_ONLY (no improvement, prefer simpler)")

    # Save
    out = {
        "t_sentence_tuning": {
            str(k): v for k, v in tuning_results.items()
        },
        "selected_t_sentence": best_t,
        "passage_only": {
            "doc_count": len(docs_a),
            "metrics": avg_a,
            "build_time_ms": round(build_ms_a, 1),
        },
        "multi_resolution": {
            "doc_count": len(docs_b),
            "sentence_count": n_sentence_b,
            "metrics": avg_b,
            "build_time_ms": round(build_ms_b, 1),
        },
        "decision": decision,
        "corpus_size": len(corpus),
        "benchmark_count": len(benchmark),
        "coverage": coverage,
    }

    out_path = PROJECT_ROOT / "data" / "multiresolution_benchmark.json"
    out_path.write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    main()
