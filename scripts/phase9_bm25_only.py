"""
Phase 9: BM25-only evaluation (no bge-m3 needed).

Runs BM25 retrieval against a corpus built from train+validation rows,
evaluated on validation benchmark queries.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def percentile(data: list[float], p: float) -> float:
    s = sorted(data)
    return s[min(int(len(s) * p / 100), len(s) - 1)]


def load_parquet_rows(parquet_path: Path, max_rows: int) -> list[dict]:
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(str(parquet_path))
    rows: list[dict] = []
    for batch in pf.iter_batches(batch_size=max_rows):
        n = min(batch.num_rows, max_rows - len(rows))
        if n <= 0:
            break
        for i in range(n):
            rows.append({
                "query_id": batch.column("query_id")[i].as_py() or 0,
                "query": batch.column("query")[i].as_py() or "",
                "Eng_Query": batch.column("Eng_Query")[i].as_py() or "",
                "query_type": batch.column("query_type")[i].as_py() or "",
                "source_lang": batch.column("source_lang")[i].as_py() or "",
                "target_lang": batch.column("target_lang")[i].as_py() or "",
                "passages": batch.column("passages")[i].as_py() or {},
            })
    return rows[:max_rows]


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-rows", type=int, default=500)
    parser.add_argument("--val-rows", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    from app.models.retrieval import Language, Query, RetrievalMode
    from ingestion.deduplication.canonical_id import canonical_passage_id
    from ingestion.normalization.text import normalize_text
    from retrieval.sparse.bm25_index import BM25Index
    from retrieval.sparse.bm25_retriever import BM25SparseRetriever

    print("=" * 60)
    print("PHASE 9: BM25-ONLY EVALUATION")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    t0 = time.perf_counter()
    train_rows = load_parquet_rows(Path("data/cache/train/hintrain.parquet"), args.train_rows)
    val_rows = load_parquet_rows(Path("data/cache/validation/hinval.parquet"), args.val_rows)
    elapsed = time.perf_counter() - t0
    print(f"  Train: {len(train_rows)} rows, Val: {len(val_rows)} rows ({elapsed:.1f}s)")

    # Build corpus from train + validation (§3.2)
    print("\nBuilding corpus...")
    seen: set[str] = set()
    corpus: list[dict] = []
    all_rows = train_rows + val_rows
    for row in all_rows:
        p = row.get("passages", {})
        for text in (p.get("English_passages") or []):
            if isinstance(text, str) and text.strip():
                pid = canonical_passage_id(text)
                if pid not in seen:
                    seen.add(pid)
                    corpus.append({"pid": pid, "text": normalize_text(text), "lang": "en"})
        for text in (p.get("Translated_passages") or []):
            if isinstance(text, str) and text.strip():
                pid = canonical_passage_id(text)
                if pid not in seen:
                    seen.add(pid)
                    corpus.append({"pid": pid, "text": normalize_text(text), "lang": "hi"})
    print(f"  Corpus: {len(corpus)} unique passages")

    lang_counts: dict[str, int] = {}
    for c in corpus:
        lang_counts[c["lang"]] = lang_counts.get(c["lang"], 0) + 1
    print(f"  Languages: {lang_counts}")

    # Build benchmark from validation
    print("\nBuilding benchmark...")
    benchmark: list[dict] = []
    corpus_ids = {c["pid"] for c in corpus}
    for row in val_rows:
        p = row.get("passages", {})
        is_sel = p.get("is_selected", []) or []
        en_list = p.get("English_passages") or []
        hi_list = p.get("Translated_passages") or []
        gold_ids = []
        for i, sel in enumerate(is_sel):
            if sel == 1 or sel == 1.0:
                if i < len(en_list) and en_list[i]:
                    gold_ids.append(canonical_passage_id(en_list[i]))
                if i < len(hi_list) and hi_list[i]:
                    gold_ids.append(canonical_passage_id(hi_list[i]))
        gold_ids = list(dict.fromkeys(gold_ids))
        qt = row.get("query", "")
        if qt and qt.strip() and gold_ids:
            benchmark.append({
                "query_id": row.get("query_id", 0),
                "query_text": qt,
                "query_type": row.get("query_type", "UNKNOWN"),
                "gold_ids": gold_ids,
            })

    # Coverage
    n_covered = sum(1 for b in benchmark if set(b["gold_ids"]) & corpus_ids)
    n_total_gold = sum(len(set(b["gold_ids"])) for b in benchmark)
    n_found_gold = sum(len(set(b["gold_ids"]) & corpus_ids) for b in benchmark)
    print(f"  Benchmark: {len(benchmark)} queries, {n_covered} covered")
    print(f"  Gold coverage: {n_found_gold}/{n_total_gold} "
          f"({n_found_gold/n_total_gold*100:.1f}%)" if n_total_gold else "")

    # Build BM25 index
    print("\nBuilding BM25 index...")
    index = BM25Index(use_bigrams=True)
    for c in corpus:
        index.add_document(c["pid"], c["text"], c["lang"])
    stats = index.build()
    print(f"  {stats.document_count} docs, vocab={stats.vocab_size}, "
          f"build={stats.build_time_ms:.1f}ms")

    retriever = BM25SparseRetriever(index=index)

    # Evaluate: CROSS_LINGUAL
    print("\n--- BM25 CROSS_LINGUAL ---")
    for mode_name, mode in [("CROSS_LINGUAL", RetrievalMode.CROSS_LINGUAL),
                             ("MONOLINGUAL", RetrievalMode.MONOLINGUAL)]:
        query_metrics: list[dict] = []
        per_qtype: dict[str, list[dict]] = {}
        latencies: list[float] = []
        covered = 0

        for b in benchmark:
            gold_set = set(b["gold_ids"])
            if not gold_set & corpus_ids:
                continue
            covered += 1

            query = Query(query_text=b["query_text"], lang=Language.HI)
            t0 = time.perf_counter()
            results = retriever.retrieve(query, mode, args.top_k)
            latencies.append((time.perf_counter() - t0) * 1000)

            retrieved_ids = [r.passage.passage_id for r in results]

            # Metrics
            r1 = 1 if retrieved_ids[:1] and retrieved_ids[0] in gold_set else 0
            r5 = 1 if any(pid in gold_set for pid in retrieved_ids[:5]) else 0
            r10 = 1 if any(pid in gold_set for pid in retrieved_ids[:args.top_k]) else 0
            mrr = 0.0
            for rank, pid in enumerate(retrieved_ids[:args.top_k], 1):
                if pid in gold_set:
                    mrr = 1.0 / rank
                    break
            rel = [1.0 if pid in gold_set else 0.0 for pid in retrieved_ids[:args.top_k]]
            dcg = sum(r / (i + 1) for i, r in enumerate(rel))
            irl = sorted(rel, reverse=True)
            idcg = sum(r / (i + 1) for i, r in enumerate(irl))
            ndcg = dcg / idcg if idcg > 0 else 0.0

            m = {"recall@1": r1, "recall@5": r5, "recall@10": r10, "mrr": mrr, "ndcg@10": ndcg}
            query_metrics.append(m)

            qt = b["query_type"]
            per_qtype.setdefault(qt, []).append(m)

        n = len(query_metrics)
        if n > 0:
            agg = {
                "recall@1": sum(m["recall@1"] for m in query_metrics) / n,
                "recall@5": sum(m["recall@5"] for m in query_metrics) / n,
                "recall@10": sum(m["recall@10"] for m in query_metrics) / n,
                "mrr": sum(m["mrr"] for m in query_metrics) / n,
                "ndcg@10": sum(m["ndcg@10"] for m in query_metrics) / n,
            }
            print(f"\n  {mode_name} (n={n} covered queries):")
            print(f"    Recall@1={agg['recall@1']:.4f} Recall@5={agg['recall@5']:.4f} "
                  f"Recall@10={agg['recall@10']:.4f}")
            print(f"    MRR={agg['mrr']:.4f} nDCG@10={agg['ndcg@10']:.4f}")
            if latencies:
                print(f"    Latency P50={percentile(latencies, 50):.1f}ms "
                      f"P95={percentile(latencies, 95):.1f}ms "
                      f"P100={percentile(latencies, 100):.1f}ms")

            # Per query type
            print("    Per query type:")
            for qt, qm in sorted(per_qtype.items()):
                qn = len(qm)
                qa = {
                    "r1": sum(m["recall@1"] for m in qm) / qn,
                    "r5": sum(m["recall@5"] for m in qm) / qn,
                    "mrr": sum(m["mrr"] for m in qm) / qn,
                }
                print(f"      {qt} (n={qn}): R@1={qa['r1']:.3f} R@5={qa['r5']:.3f} "
                      f"MRR={qa['mrr']:.3f}")
        else:
            print(f"\n  {mode_name}: No covered queries")

    # Save results
    output = {
        "config": {"train_rows": args.train_rows, "val_rows": args.val_rows, "top_k": args.top_k},
        "corpus": {"passages": len(corpus), "languages": lang_counts},
        "benchmark": {"total": len(benchmark), "covered": n_covered},
    }
    out_path = Path("data/phase9_bm25_results.json")
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
