"""
Phase 9: Retrieval Evaluation Harness

Builds a corpus from train+validation passages (architecturally correct per §3.2),
evaluates BM25-only, Dense-only, and Hybrid retrieval on validation benchmark queries.

Architecture alignment:
- §3.2: Corpus drawn from train+validation combined (passages belong in corpus)
- §4: Validation labels used only at reporting time
- §6: Gold-set uses canonical passage_ids
- §7: Monolingual vs cross-lingual experiment
"""
from __future__ import annotations

import csv
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def percentile(data: list[float], p: float) -> float:
    sorted_d = sorted(data)
    idx = int(len(sorted_d) * p / 100)
    return sorted_d[min(idx, len(sorted_d) - 1)]


def load_parquet_rows(parquet_path: Path, max_rows: int) -> list[dict]:
    """Load rows from parquet file."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(parquet_path))
    rows: list[dict] = []
    for batch in pf.iter_batches(batch_size=max_rows):
        n = min(batch.num_rows, max_rows - len(rows))
        if n <= 0:
            break
        passages_col = batch.column("passages")
        query_col = batch.column("query")
        eq_col = batch.column("Eng_Query")
        qt_col = batch.column("query_type")
        sl_col = batch.column("source_lang")
        tl_col = batch.column("target_lang")
        qid_col = batch.column("query_id")
        for i in range(n):
            rows.append({
                "query_id": qid_col[i].as_py() or 0,
                "query": query_col[i].as_py() or "",
                "Eng_Query": eq_col[i].as_py() or "",
                "query_type": qt_col[i].as_py() or "",
                "source_lang": sl_col[i].as_py() or "",
                "target_lang": tl_col[i].as_py() or "",
                "passages": passages_col[i].as_py() or {},
            })
    return rows[:max_rows]


def extract_corpus_from_rows(rows: list[dict]) -> list[dict]:
    """Extract passages from rows for corpus building (no evaluation labels)."""
    from ingestion.deduplication.canonical_id import canonical_passage_id
    from ingestion.normalization.text import normalize_text

    seen: set[str] = set()
    corpus: list[dict] = []

    for row in rows:
        passages = row.get("passages", {})
        en_list = passages.get("English_passages", []) or []
        hi_list = passages.get("Translated_passages", []) or []

        for text in en_list:
            if isinstance(text, str) and text.strip():
                pid = canonical_passage_id(text)
                if pid not in seen:
                    seen.add(pid)
                    corpus.append({"passage_id": pid, "text": normalize_text(text), "lang": "en"})

        for text in hi_list:
            if isinstance(text, str) and text.strip():
                pid = canonical_passage_id(text)
                if pid not in seen:
                    seen.add(pid)
                    corpus.append({"passage_id": pid, "text": normalize_text(text), "lang": "hi"})

    return corpus


def extract_benchmark_from_rows(rows: list[dict]) -> list[dict]:
    """Extract benchmark queries with gold labels from validation rows."""
    from ingestion.deduplication.canonical_id import canonical_passage_id

    benchmark: list[dict] = []
    for row in rows:
        passages = row.get("passages", {})
        is_selected = passages.get("is_selected", []) or []
        en_list = passages.get("English_passages", []) or []
        hi_list = passages.get("Translated_passages", []) or []

        gold_ids: list[str] = []
        for i, sel in enumerate(is_selected):
            if sel == 1 or sel == 1.0:
                if i < len(en_list) and en_list[i]:
                    gold_ids.append(canonical_passage_id(en_list[i]))
                if i < len(hi_list) and hi_list[i]:
                    gold_ids.append(canonical_passage_id(hi_list[i]))

        # Deduplicate gold IDs
        gold_ids = list(dict.fromkeys(gold_ids))

        query_text = row.get("query", "")
        if not query_text or not query_text.strip():
            continue

        benchmark.append({
            "query_id": row.get("query_id", 0),
            "query_text": query_text,
            "query_type": row.get("query_type", "UNKNOWN"),
            "source_lang": row.get("source_lang", ""),
            "target_lang": row.get("target_lang", ""),
            "gold_passage_ids": gold_ids,
        })

    return benchmark


def compute_metrics(
    retrieved_ids: list[str],
    gold_ids: set[str],
    top_k: int = 10,
) -> dict:
    """Compute retrieval metrics for a single query."""
    hits_at_1 = 1 if retrieved_ids[:1] and retrieved_ids[0] in gold_ids else 0
    hits_at_5 = 1 if any(pid in gold_ids for pid in retrieved_ids[:5]) else 0
    hits_at_10 = 1 if any(pid in gold_ids for pid in retrieved_ids[:top_k]) else 0

    # MRR
    mrr = 0.0
    for rank, pid in enumerate(retrieved_ids[:top_k], 1):
        if pid in gold_ids:
            mrr = 1.0 / rank
            break

    # nDCG@10
    relevance = [1.0 if pid in gold_ids else 0.0 for pid in retrieved_ids[:top_k]]
    dcg = sum(rel / (i + 1) for i, rel in enumerate(relevance))
    ideal_relevance = sorted(relevance, reverse=True)
    idcg = sum(rel / (i + 1) for i, rel in enumerate(ideal_relevance))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {
        "recall@1": hits_at_1,
        "recall@5": hits_at_5,
        "recall@10": hits_at_10,
        "mrr": mrr,
        "ndcg@10": ndcg,
    }


def aggregate_metrics(query_metrics: list[dict]) -> dict:
    """Aggregate per-query metrics into summary statistics."""
    n = len(query_metrics)
    if n == 0:
        return {k: 0.0 for k in ["recall@1", "recall@5", "recall@10", "mrr", "ndcg@10"]}

    return {
        "recall@1": sum(m["recall@1"] for m in query_metrics) / n,
        "recall@5": sum(m["recall@5"] for m in query_metrics) / n,
        "recall@10": sum(m["recall@10"] for m in query_metrics) / n,
        "mrr": sum(m["mrr"] for m in query_metrics) / n,
        "ndcg@10": sum(m["ndcg@10"] for m in query_metrics) / n,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Phase 9: Retrieval evaluation")
    parser.add_argument("--train-rows", type=int, default=1000, help="Training rows for corpus")
    parser.add_argument("--val-rows", type=int, default=500, help="Validation rows for benchmark")
    parser.add_argument("--device", type=str, default="cpu", help="Device for bge-m3")
    parser.add_argument("--top-k", type=int, default=10, help="Retrieval top-K")
    parser.add_argument("--benchmark-size", type=int, default=200, help="Max benchmark queries")
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 9: RETRIEVAL EVALUATION")
    print("=" * 70)

    # --- Step 1: Load data ---
    print("\n--- STEP 1: LOADING DATA ---")
    train_path = Path("data/cache/train/hintrain.parquet")
    val_path = Path("data/cache/validation/hinval.parquet")

    t0 = time.perf_counter()
    train_rows = load_parquet_rows(train_path, args.train_rows)
    print(f"  Loaded {len(train_rows)} train rows in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    val_rows = load_parquet_rows(val_path, args.val_rows)
    print(f"  Loaded {len(val_rows)} validation rows in {time.perf_counter() - t0:.1f}s")

    # --- Step 2: Build corpus from train + validation (§3.2) ---
    print("\n--- STEP 2: BUILDING CORPUS (train + validation combined) ---")
    all_rows = train_rows + val_rows
    t0 = time.perf_counter()
    corpus = extract_corpus_from_rows(all_rows)
    corpus_time = time.perf_counter() - t0

    lang_counts: dict[str, int] = {}
    for p in corpus:
        lang_counts[p["lang"]] = lang_counts.get(p["lang"], 0) + 1

    print(f"  Corpus: {len(corpus)} unique passages in {corpus_time:.1f}s")
    print(f"  Languages: {lang_counts}")

    # --- Step 3: Build benchmark from validation (§4) ---
    print("\n--- STEP 3: BUILDING BENCHMARK QUERIES ---")
    benchmark = extract_benchmark_from_rows(val_rows[:args.benchmark_size])

    # Coverage check
    corpus_ids = {p["passage_id"] for p in corpus}
    n_covered = 0
    n_partial = 0
    n_uncovered = 0
    total_gold = 0
    found_gold = 0

    for b in benchmark:
        gold_set = set(b["gold_passage_ids"])
        total_gold += len(gold_set)
        found = gold_set & corpus_ids
        found_gold += len(found)
        if len(found) == len(gold_set) and len(gold_set) > 0:
            n_covered += 1
        elif len(found) > 0:
            n_partial += 1
        else:
            n_uncovered += 1

    print(f"  Benchmark queries: {len(benchmark)}")
    print(f"  Fully covered: {n_covered}")
    print(f"  Partially covered: {n_partial}")
    print(f"  Uncovered: {n_uncovered}")
    print(f"  Gold passage coverage: {found_gold}/{total_gold} "
          f"({found_gold/total_gold*100:.1f}%)" if total_gold > 0 else "")

    # --- Step 4: Build BM25 index ---
    print("\n--- STEP 4: BUILDING BM25 INDEX ---")
    from retrieval.sparse.bm25_index import BM25Index
    from retrieval.sparse.bm25_retriever import BM25SparseRetriever

    bm25_index = BM25Index(use_bigrams=True)
    for p in corpus:
        bm25_index.add_document(p["passage_id"], p["text"], p["lang"])
    bm25_stats = bm25_index.build()
    print(f"  BM25: {bm25_stats.document_count} docs, vocab={bm25_stats.vocab_size}, "
          f"build={bm25_stats.build_time_ms:.1f}ms")

    sparse_retriever = BM25SparseRetriever(index=bm25_index)

    # --- Step 5: Build Qdrant index with bge-m3 ---
    print("\n--- STEP 5: BUILDING QDRANT INDEX ---")
    from sentence_transformers import SentenceTransformer

    t0 = time.perf_counter()
    model = SentenceTransformer("BAAI/bge-m3", device=args.device)
    load_time = time.perf_counter() - t0
    print(f"  Model loaded in {load_time:.2f}s")

    from embeddings.bge_m3 import BgeM3EmbeddingProvider
    from ingestion.representation.base import create_passage_representation
    from retrieval.dense.dense_retriever import BgeM3DenseRetriever
    from retrieval.dense.qdrant_index import QdrantIndexManager

    # Create representations
    representations = []
    for p in corpus:
        repr_obj = create_passage_representation(
            passage_id=p["passage_id"],
            text=p["text"],
            lang=p["lang"],
        )
        representations.append(repr_obj)

    # Embed all passages
    texts = [p["text"] for p in corpus]
    t0 = time.perf_counter()
    vectors = model.encode(texts, normalize_embeddings=True, batch_size=16, show_progress_bar=True)
    embed_time = time.perf_counter() - t0
    vectors_list = [v.tolist() for v in vectors]
    print(f"  Embedded {len(texts)} passages in {embed_time:.1f}s "
          f"({len(texts)/embed_time:.1f} passages/sec)")

    # Build Qdrant
    qdrant = QdrantIndexManager(collection_name="phase9_eval")
    qdrant.create_collection()
    t0 = time.perf_counter()
    qdrant.upsert_passages(representations, vectors_list, batch_size=500)
    upsert_time = (time.perf_counter() - t0) * 1000
    print(f"  Qdrant upsert: {upsert_time:.1f}ms")

    provider = BgeM3EmbeddingProvider(device=args.device)
    provider._model = model
    dense_retriever = BgeM3DenseRetriever(qdrant_manager=qdrant, embedding_provider=provider)

    from retrieval.fusion.hybrid_retriever import HybridRetriever

    hybrid = HybridRetriever(sparse_retriever=sparse_retriever, dense_retriever=dense_retriever)

    # --- Step 6: Evaluate ---
    print("\n--- STEP 6: RUNNING EVALUATION ---")
    from app.models.retrieval import Language, Query, RetrievalMode

    configs = {
        "BM25": sparse_retriever,
        "Dense": dense_retriever,
        "Hybrid": hybrid,
    }

    all_results: dict[str, dict] = {}

    for config_name, retriever in configs.items():
        print(f"\n  Evaluating {config_name}...")

        query_metrics: list[dict] = []
        per_lang: dict[str, list[dict]] = {}
        per_qtype: dict[str, list[dict]] = {}
        latencies: list[float] = []

        covered_count = 0

        for b in benchmark:
            gold_ids = set(b["gold_passage_ids"])
            if not gold_ids:
                continue

            # Check if any gold passage is in corpus
            if not gold_ids & corpus_ids:
                continue

            covered_count += 1
            query_text = b["query_text"]

            try:
                lang_enum = Language("hi")  # Validation queries are Hindi
            except ValueError:
                lang_enum = Language.EN

            query = Query(query_text=query_text, lang=lang_enum)

            t0 = time.perf_counter()
            try:
                results = retriever.retrieve(query, RetrievalMode.CROSS_LINGUAL, args.top_k)
            except Exception:
                results = []
            latency_ms = (time.perf_counter() - t0) * 1000
            latencies.append(latency_ms)

            retrieved_ids = [r.passage.passage_id for r in results]
            metrics = compute_metrics(retrieved_ids, gold_ids, args.top_k)
            query_metrics.append(metrics)

            # Per-language (all validation queries are Hindi in this dataset)
            lang_key = "hi"
            if lang_key not in per_lang:
                per_lang[lang_key] = []
            per_lang[lang_key].append(metrics)

            # Per-query-type
            qtype = b.get("query_type", "UNKNOWN")
            if qtype not in per_qtype:
                per_qtype[qtype] = []
            per_qtype[qtype].append(metrics)

        # Aggregate
        overall = aggregate_metrics(query_metrics)
        lang_agg = {k: aggregate_metrics(v) for k, v in per_lang.items()}
        qtype_agg = {k: aggregate_metrics(v) for k, v in per_qtype.items()}

        latency_stats = {}
        if latencies:
            latency_stats = {
                "p50_ms": percentile(latencies, 50),
                "p70_ms": percentile(latencies, 70),
                "p95_ms": percentile(latencies, 95),
                "p100_ms": percentile(latencies, 100),
                "mean_ms": sum(latencies) / len(latencies),
            }

        all_results[config_name] = {
            "overall": overall,
            "per_language": lang_agg,
            "per_query_type": qtype_agg,
            "latency": latency_stats,
            "covered_queries": covered_count,
            "total_benchmark_queries": len(benchmark),
        }

        print(f"    Recall@1={overall['recall@1']:.4f} "
              f"Recall@5={overall['recall@5']:.4f} "
              f"Recall@10={overall['recall@10']:.4f} "
              f"MRR={overall['mrr']:.4f} nDCG={overall['ndcg@10']:.4f} "
              f"(covered={covered_count})")
        if latency_stats:
            print(f"    Latency P50={latency_stats['p50_ms']:.1f}ms "
                  f"P95={latency_stats['p95_ms']:.1f}ms")

    # --- Step 7: Hindi monolingual vs cross-lingual ---
    print("\n--- STEP 7: HINDI-ONLY vs HINDI+ENGLISH ---")
    for mode_name, mode in [
        ("MONOLINGUAL", RetrievalMode.MONOLINGUAL),
        ("CROSS_LINGUAL", RetrievalMode.CROSS_LINGUAL),
    ]:
        query_metrics = []
        for b in benchmark:
            gold_ids = set(b["gold_passage_ids"])
            if not gold_ids or not gold_ids & corpus_ids:
                continue

            query = Query(query_text=b["query_text"], lang=Language("hi"))
            try:
                results = sparse_retriever.retrieve(query, mode, args.top_k)
            except Exception:
                results = []
            retrieved_ids = [r.passage.passage_id for r in results]
            query_metrics.append(compute_metrics(retrieved_ids, gold_ids, args.top_k))

        agg = aggregate_metrics(query_metrics)
        print(f"  BM25 {mode_name}: Recall@1={agg['recall@1']:.4f} "
              f"Recall@5={agg['recall@5']:.4f} MRR={agg['mrr']:.4f} "
              f"(n={len(query_metrics)})")

    # --- Step 8: Save results ---
    print("\n--- STEP 8: SAVING RESULTS ---")
    output = {
        "config": {
            "train_rows": args.train_rows,
            "val_rows": args.val_rows,
            "top_k": args.top_k,
            "device": args.device,
            "model": "BAAI/bge-m3",
            "rrf_k": 60,
        },
        "corpus": {
            "passages": len(corpus),
            "languages": lang_counts,
        },
        "benchmark": {
            "total_queries": len(benchmark),
            "covered_queries": n_covered,
            "partial_queries": n_partial,
            "uncovered_queries": n_uncovered,
            "gold_coverage_rate": found_gold / total_gold if total_gold > 0 else 0.0,
        },
        "results": all_results,
    }

    output_path = Path("data/phase9_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    # Save CSV ablation table
    csv_path = Path("data/phase9_ablation.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "config", "recall@1", "recall@5", "recall@10", "mrr", "ndcg@10",
            "p50_ms", "p95_ms", "p100_ms", "covered_queries",
        ])
        for name, res in all_results.items():
            lat = res["latency"]
            writer.writerow([
                name,
                f"{res['overall']['recall@1']:.4f}",
                f"{res['overall']['recall@5']:.4f}",
                f"{res['overall']['recall@10']:.4f}",
                f"{res['overall']['mrr']:.4f}",
                f"{res['overall']['ndcg@10']:.4f}",
                f"{lat.get('p50_ms', 0):.1f}",
                f"{lat.get('p95_ms', 0):.1f}",
                f"{lat.get('p100_ms', 0):.1f}",
                res["covered_queries"],
            ])

    print(f"  Results saved to {output_path}")
    print(f"  Ablation table saved to {csv_path}")

    # Cleanup
    qdrant.delete_collection()
    del model
    gc.collect()

    print("\n" + "=" * 70)
    print("PHASE 9 EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
