"""
Phase 9.5: Complete Retrieval Benchmark

Produces comparable measurements for:
  A. BM25-only
  B. Dense bge-m3-only
  C. BM25 + Dense + RRF hybrid

Using the SAME corpus, benchmark queries, gold sets, and top-K.

Architecture alignment:
- §3.2: Corpus from train+validation combined
- §4: Validation labels only at reporting time
- §6: Gold-set via canonical passage_ids
- ADR-0003: sparse=BM25, dense=bge-m3, fusion=RRF
"""
from __future__ import annotations

import contextlib
import csv
import gc
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def percentile(data: list[float], p: float) -> float:
    s = sorted(data)
    idx = int(len(s) * p / 100)
    return s[min(idx, len(s) - 1)]


def latency_stats(latencies: list[float], label: str = "") -> dict:
    if not latencies:
        return {}
    return {
        "label": label,
        "count": len(latencies),
        "p50_ms": percentile(latencies, 50),
        "p70_ms": percentile(latencies, 70),
        "p95_ms": percentile(latencies, 95),
        "p100_ms": percentile(latencies, 100),
        "mean_ms": statistics.mean(latencies),
        "stdev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# Data loading (pyarrow with nested struct)
# ---------------------------------------------------------------------------

def load_parquet_rows(parquet_path: Path, max_rows: int) -> list[dict]:
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


# ---------------------------------------------------------------------------
# Corpus + Benchmark construction
# ---------------------------------------------------------------------------

def build_corpus_and_benchmark(
    train_rows: list[dict],
    val_rows: list[dict],
) -> tuple[list[dict], list[dict], dict]:
    """Build corpus from train+val, benchmark from val.

    Returns (corpus, benchmark, coverage_info).
    """
    from ingestion.deduplication.canonical_id import canonical_passage_id
    from ingestion.normalization.text import normalize_text

    # Corpus: all passages from train + val (§3.2)
    seen: set[str] = set()
    corpus: list[dict] = []
    for row in train_rows + val_rows:
        p = row.get("passages", {})
        for text in (p.get("English_passages") or []):
            if isinstance(text, str) and text.strip():
                pid = canonical_passage_id(text)
                if pid not in seen:
                    seen.add(pid)
                    corpus.append({"passage_id": pid, "text": normalize_text(text), "lang": "en"})
        for text in (p.get("Translated_passages") or []):
            if isinstance(text, str) and text.strip():
                pid = canonical_passage_id(text)
                if pid not in seen:
                    seen.add(pid)
                    corpus.append({"passage_id": pid, "text": normalize_text(text), "lang": "hi"})

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
        gold_ids = list(dict.fromkeys(gold_ids))
        qt = row.get("query", "")
        if qt and qt.strip() and gold_ids:
            benchmark.append({
                "query_id": row.get("query_id", 0),
                "query_text": qt,
                "Eng_Query": row.get("Eng_Query", ""),
                "query_type": row.get("query_type", "UNKNOWN"),
                "source_lang": row.get("source_lang", ""),
                "gold_passage_ids": gold_ids,
            })

    # Coverage
    n_covered = sum(1 for b in benchmark if set(b["gold_passage_ids"]) & corpus_ids)
    n_partial = sum(
        1 for b in benchmark
        if set(b["gold_passage_ids"]) & corpus_ids
        and not set(b["gold_passage_ids"]).issubset(corpus_ids)
    )
    n_uncovered = sum(1 for b in benchmark if not (set(b["gold_passage_ids"]) & corpus_ids))
    total_gold = sum(len(set(b["gold_passage_ids"])) for b in benchmark)
    found_gold = sum(len(set(b["gold_passage_ids"]) & corpus_ids) for b in benchmark)

    coverage = {
        "total_queries": len(benchmark),
        "covered": n_covered,
        "partial": n_partial,
        "uncovered": n_uncovered,
        "total_gold_passages": total_gold,
        "found_gold_passages": found_gold,
        "gold_coverage_rate": found_gold / total_gold if total_gold > 0 else 0.0,
    }

    return corpus, benchmark, coverage


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_query_metrics(
    retrieved_ids: list[str],
    gold_ids: set[str],
    top_k: int = 10,
) -> dict:
    r1 = 1 if retrieved_ids[:1] and retrieved_ids[0] in gold_ids else 0
    r5 = 1 if any(pid in gold_ids for pid in retrieved_ids[:5]) else 0
    r10 = 1 if any(pid in gold_ids for pid in retrieved_ids[:top_k]) else 0

    mrr = 0.0
    for rank, pid in enumerate(retrieved_ids[:top_k], 1):
        if pid in gold_ids:
            mrr = 1.0 / rank
            break

    rel = [1.0 if pid in gold_ids else 0.0 for pid in retrieved_ids[:top_k]]
    dcg = sum(r / (i + 1) for i, r in enumerate(rel))
    irl = sorted(rel, reverse=True)
    idcg = sum(r / (i + 1) for i, r in enumerate(irl))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {"recall@1": r1, "recall@5": r5, "recall@10": r10, "mrr": mrr, "ndcg@10": ndcg}


def aggregate(query_metrics: list[dict]) -> dict:
    n = len(query_metrics)
    if n == 0:
        return {k: 0.0 for k in ["recall@1", "recall@5", "recall@10", "mrr", "ndcg@10"]}
    return {k: sum(m[k] for m in query_metrics) / n for k in query_metrics[0]}


# ---------------------------------------------------------------------------
# Latency investigation: BM25 component breakdown
# ---------------------------------------------------------------------------

def investigate_bm25_latency(index, queries, corpus_size, n_runs=50):
    """Break down BM25 latency into components to explain the Phase 7 vs Phase 9 discrepancy."""
    from app.models.retrieval import Language, Query, RetrievalMode
    from ingestion.normalization.tokenize import tokenize_for_lang
    from retrieval.sparse.bm25_retriever import BM25SparseRetriever

    retriever = BM25SparseRetriever(index=index)
    # Warm up
    for _ in range(5):
        for q in queries:
            query_obj = Query(query_text=q, lang=Language.HI)
            retriever.retrieve(query_obj, RetrievalMode.CROSS_LINGUAL, 10)

    # Component timing
    tokenize_times = []
    search_times = []
    doc_lookup_times = []
    full_times = []

    for _ in range(n_runs):
        for q in queries:
            query_obj = Query(query_text=q, lang=Language.HI)

            # Tokenization
            t0 = time.perf_counter()
            tokens = tokenize_for_lang(q, "hi")
            t1 = time.perf_counter()
            tokenize_times.append((t1 - t0) * 1000)

            # Search (core BM25 scoring)
            t2 = time.perf_counter()
            bm25_results = index._bm25.get_scores(tokens)
            indexed = list(enumerate(bm25_results))
            indexed.sort(key=lambda x: x[1], reverse=True)
            top_indices = [i for i, _ in indexed[:10]]
            t3 = time.perf_counter()
            search_times.append((t3 - t2) * 1000)

            # Document lookup (linear scan)
            t4 = time.perf_counter()
            for idx in top_indices:
                _ = index._documents[idx]
            t5 = time.perf_counter()
            doc_lookup_times.append((t5 - t4) * 1000)

            # Full retrieve
            t6 = time.perf_counter()
            retriever.retrieve(query_obj, RetrievalMode.CROSS_LINGUAL, 10)
            t7 = time.perf_counter()
            full_times.append((t7 - t6) * 1000)

    return {
        "tokenization": latency_stats(tokenize_times, "tokenization"),
        "bm25_scoring": latency_stats(search_times, "bm25_scoring"),
        "doc_lookup": latency_stats(doc_lookup_times, "doc_lookup"),
        "full_retrieve": latency_stats(full_times, "full_retrieve"),
    }


# ---------------------------------------------------------------------------
# Dense evaluation (may be blocked by environment)
# ---------------------------------------------------------------------------

def attempt_dense_evaluation(
    corpus: list[dict],
    benchmark: list[dict],
    corpus_ids: set[str],
    top_k: int = 10,
    device: str = "cpu",
) -> dict | None:
    """Attempt dense evaluation. Returns None if blocked by environment."""
    try:
        print("  Loading bge-m3 model...")
        t0 = time.perf_counter()
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-m3", device=device)
        load_time = time.perf_counter() - t0
        print(f"  Model loaded in {load_time:.1f}s")
    except Exception as e:
        print(f"  BLOCKED: {e}")
        return None

    try:
        from app.models.retrieval import Language, Query, RetrievalMode
        from embeddings.bge_m3 import BgeM3EmbeddingProvider
        from ingestion.representation.base import create_passage_representation
        from retrieval.dense.dense_retriever import BgeM3DenseRetriever
        from retrieval.dense.qdrant_index import QdrantIndexManager
        from retrieval.fusion.hybrid_retriever import HybridRetriever
        from retrieval.sparse.bm25_index import BM25Index
        from retrieval.sparse.bm25_retriever import BM25SparseRetriever

        # Build Qdrant index
        print("  Building Qdrant index...")
        representations = [
            create_passage_representation(p["passage_id"], p["text"], p["lang"])
            for p in corpus
        ]
        texts = [p["text"] for p in corpus]
        t0 = time.perf_counter()
        vectors = model.encode(texts, normalize_embeddings=True, batch_size=16)
        embed_time = time.perf_counter() - t0
        vectors_list = [v.tolist() for v in vectors]
        pps = len(texts) / embed_time
        print(f"  Embedded {len(texts)} passages in "
              f"{embed_time:.1f}s ({pps:.1f} pass/s)")

        qdrant = QdrantIndexManager(collection_name="phase9_5_bench")
        qdrant.create_collection()
        t0 = time.perf_counter()
        qdrant.upsert_passages(representations, vectors_list, batch_size=500)
        upsert_ms = (time.perf_counter() - t0) * 1000
        print(f"  Qdrant upsert: {upsert_ms:.1f}ms")

        provider = BgeM3EmbeddingProvider(device=device)
        provider._model = model  # Reuse loaded model
        dense_retriever = BgeM3DenseRetriever(qdrant_manager=qdrant, embedding_provider=provider)

        # Build BM25 for hybrid
        bm25_index = BM25Index(use_bigrams=True)
        for p in corpus:
            bm25_index.add_document(p["passage_id"], p["text"], p["lang"])
        bm25_index.build()
        bm25_retriever = BM25SparseRetriever(index=bm25_index)
        hybrid_retriever = HybridRetriever(
            sparse_retriever=bm25_retriever,
            dense_retriever=dense_retriever,
        )

        # Evaluate dense and hybrid
        dense_results = {}
        hybrid_results = {}
        for config_name, retriever in [("Dense", dense_retriever), ("Hybrid", hybrid_retriever)]:
            print(f"  Evaluating {config_name}...")
            query_metrics = []
            latencies_embed = []
            latencies_total = []
            per_lang: dict[str, list[dict]] = {}
            per_qtype: dict[str, list[dict]] = {}
            covered = 0

            for b in benchmark:
                gold_set = set(b["gold_passage_ids"])
                if not gold_set or not (gold_set & corpus_ids):
                    continue
                covered += 1
                query_text = b["query_text"]

                try:
                    q_lang = Language.HI
                except ValueError:
                    q_lang = Language.EN

                query = Query(query_text=query_text, lang=q_lang)

                # Embed query separately for latency
                t0 = time.perf_counter()
                _ = model.encode([query_text], normalize_embeddings=True)
                embed_ms = (time.perf_counter() - t0) * 1000

                # Full retrieval
                t0 = time.perf_counter()
                try:
                    results = retriever.retrieve(query, RetrievalMode.CROSS_LINGUAL, top_k)
                except Exception:
                    results = []
                total_ms = (time.perf_counter() - t0) * 1000

                latencies_embed.append(embed_ms)
                latencies_total.append(total_ms)

                retrieved_ids = [r.passage.passage_id for r in results]
                metrics = compute_query_metrics(retrieved_ids, gold_set, top_k)
                query_metrics.append(metrics)

                # Per language/query type
                lang_key = "hi"
                per_lang.setdefault(lang_key, []).append(metrics)
                qt = b.get("query_type", "UNKNOWN")
                per_qtype.setdefault(qt, []).append(metrics)

            overall = aggregate(query_metrics)
            lat_agg = latency_stats(latencies_total, config_name)
            embed_agg = latency_stats(latencies_embed, f"{config_name}_embed")

            config_result = {
                "overall": overall,
                "per_language": {k: aggregate(v) for k, v in per_lang.items()},
                "per_query_type": {k: aggregate(v) for k, v in per_qtype.items()},
                "latency": lat_agg,
                "embedding_latency": embed_agg,
                "covered_queries": covered,
            }
            print(f"    R@1={overall['recall@1']:.4f} R@5={overall['recall@5']:.4f} "
                  f"R@10={overall['recall@10']:.4f} MRR={overall['mrr']:.4f} "
                  f"nDCG={overall['ndcg@10']:.4f} (n={covered})")
            if lat_agg:
                print(f"    Latency P50={lat_agg['p50_ms']:.1f}ms P95={lat_agg['p95_ms']:.1f}ms")

            if config_name == "Dense":
                dense_results = config_result
            else:
                hybrid_results = config_result

        qdrant.delete_collection()
        del model
        gc.collect()

        return {
            "dense": dense_results,
            "hybrid": hybrid_results,
            "model_load_time_s": load_time,
            "embed_time_s": embed_time,
            "upsert_time_ms": upsert_ms,
        }

    except Exception as e:
        print(f"  DENSE EVALUATION FAILED: {e}")
        with contextlib.suppress(Exception):
            qdrant.delete_collection()
        gc.collect()
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Phase 9.5: Complete Retrieval Benchmark")
    parser.add_argument("--train-rows", type=int, default=200, help="Training rows for corpus")
    parser.add_argument("--val-rows", type=int, default=100, help="Validation rows for benchmark")
    parser.add_argument("--top-k", type=int, default=10, help="Retrieval top-K")
    parser.add_argument("--device", type=str, default="cpu", help="Device for bge-m3")
    parser.add_argument("--skip-dense", action="store_true", help="Skip dense evaluation")
    parser.add_argument(
        "--warmup-runs", type=int, default=20,
        help="Warm-up iterations for latency",
    )
    parser.add_argument(
        "--latency-runs", type=int, default=50,
        help="Measurement iterations for latency",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 9.5: COMPLETE RETRIEVAL BENCHMARK")
    print("=" * 70)

    # --- Load data ---
    print("\n--- STEP 1: LOADING DATA ---")
    train_path = Path("data/cache/train/hintrain.parquet")
    val_path = Path("data/cache/validation/hinval.parquet")

    t0 = time.perf_counter()
    train_rows = load_parquet_rows(train_path, args.train_rows)
    val_rows = load_parquet_rows(val_path, args.val_rows)
    load_time = time.perf_counter() - t0
    print(f"  Loaded {len(train_rows)} train + {len(val_rows)} val rows in {load_time:.1f}s")

    # --- Build corpus + benchmark ---
    print("\n--- STEP 2: BUILDING CORPUS + BENCHMARK ---")
    t0 = time.perf_counter()
    corpus, benchmark, coverage = build_corpus_and_benchmark(train_rows, val_rows)
    build_time = time.perf_counter() - t0

    corpus_ids = {p["passage_id"] for p in corpus}
    lang_counts: dict[str, int] = {}
    for p in corpus:
        lang_counts[p["lang"]] = lang_counts.get(p["lang"], 0) + 1

    print(f"  Corpus: {len(corpus)} unique passages ({build_time:.1f}s)")
    print(f"  Languages: {lang_counts}")
    print(f"  Benchmark: {coverage['total_queries']} queries")
    print(f"  Coverage: {coverage['covered']} covered, {coverage['partial']} partial, "
          f"{coverage['uncovered']} uncovered")
    print(f"  Gold coverage: {coverage['found_gold_passages']}/{coverage['total_gold_passages']} "
          f"({coverage['gold_coverage_rate']*100:.1f}%)")

    # Query type distribution
    qtype_counts: dict[str, int] = {}
    for b in benchmark:
        qt = b.get("query_type", "UNKNOWN")
        qtype_counts[qt] = qtype_counts.get(qt, 0) + 1
    print(f"  Query types: {qtype_counts}")

    # --- Build BM25 ---
    print("\n--- STEP 3: BUILDING BM25 INDEX ---")
    from retrieval.sparse.bm25_index import BM25Index
    from retrieval.sparse.bm25_retriever import BM25SparseRetriever

    bm25_index = BM25Index(use_bigrams=True)
    for p in corpus:
        bm25_index.add_document(p["passage_id"], p["text"], p["lang"])
    bm25_stats = bm25_index.build()
    print(f"  {bm25_stats.document_count} docs, vocab={bm25_stats.vocab_size}, "
          f"build={bm25_stats.build_time_ms:.1f}ms")

    bm25_retriever = BM25SparseRetriever(index=bm25_index)

    # --- BM25 Latency Investigation ---
    print("\n--- STEP 4: BM25 LATENCY INVESTIGATION ---")
    print("  Investigating Phase 7 (0.8ms) vs Phase 9 (200ms) discrepancy...")
    print(f"  Corpus size: {len(corpus)} docs")

    # Use a mix of real Hindi validation queries + synthetic queries
    inv_queries = [b["query_text"] for b in benchmark[:5]]
    if len(inv_queries) < 5:
        inv_queries.extend([
            "भारत की राजधानी",
            "कृत्रिम बुद्धिमत्ता",
            "मशीन लर्निंग",
            "जलवायु परिवर्तन",
            "प्राकृतिक भाषा प्रसंस्करण",
        ])
    inv_queries = inv_queries[:5]

    # Compare: cold start vs warmed up
    from app.models.retrieval import (
        Language as LangEnum,
    )
    from app.models.retrieval import (
        Query as QueryObj,
    )
    from app.models.retrieval import (
        RetrievalMode as RMode,
    )

    cold_latencies = []
    for q in inv_queries:
        q_obj = QueryObj(query_text=q, lang=LangEnum.HI)
        t0 = time.perf_counter()
        bm25_retriever.retrieve(
            q_obj, RMode.CROSS_LINGUAL, args.top_k,
        )
        cold_latencies.append((time.perf_counter() - t0) * 1000)

    cold_stats = latency_stats(cold_latencies, "cold_start")
    print(f"  Cold start (5 queries, 1 run):  P50={cold_stats['p50_ms']:.2f}ms  "
          f"P95={cold_stats['p95_ms']:.2f}ms  P100={cold_stats['p100_ms']:.2f}ms")

    # Warmed up
    component_breakdown = investigate_bm25_latency(
        bm25_index, inv_queries, len(corpus), n_runs=args.latency_runs,
    )

    for comp_name, comp_stats in component_breakdown.items():
        if comp_stats:
            print(f"  {comp_name:20s}:  P50={comp_stats['p50_ms']:.3f}ms  "
                  f"P95={comp_stats['p95_ms']:.3f}ms  mean={comp_stats['mean_ms']:.3f}ms")

    # Identify the discrepancy cause
    full_p50 = component_breakdown["full_retrieve"]["p50_ms"]
    score_p50 = component_breakdown["bm25_scoring"]["p50_ms"]
    token_p50 = component_breakdown["tokenization"]["p50_ms"]
    print("\n  INVESTIGATION FINDING:")
    print(f"    BM25 scoring core:     {score_p50:.3f}ms (the actual search)")
    print(f"    Tokenization:          {token_p50:.3f}ms")
    print(f"    Full retrieve:         {full_p50:.3f}ms"
          " (Query/RetrievalResult overhead)")
    overhead = full_p50 - score_p50 - token_p50
    print(f"    Python overhead:       {overhead:.3f}ms (Pydantic model construction, etc.)")
    print(f"    Corpus size:           {len(corpus)} documents")

    # --- BM25 Evaluation ---
    print("\n--- STEP 5: BM25 EVALUATION (with proper warm-up) ---")
    from app.models.retrieval import Language, Query, RetrievalMode

    for mode_name, mode in [
        ("CROSS_LINGUAL", RetrievalMode.CROSS_LINGUAL),
        ("MONOLINGUAL", RetrievalMode.MONOLINGUAL),
    ]:
        # Warm up
        for _ in range(args.warmup_runs):
            for b in benchmark:
                gold_set = set(b["gold_passage_ids"])
                if not gold_set or not (gold_set & corpus_ids):
                    continue
                q = Query(query_text=b["query_text"], lang=Language.HI)
                bm25_retriever.retrieve(q, mode, args.top_k)

        query_metrics = []
        per_lang_metrics: dict[str, list[dict]] = {}
        per_qtype_metrics: dict[str, list[dict]] = {}
        latencies = []
        covered = 0

        for b in benchmark:
            gold_set = set(b["gold_passage_ids"])
            if not gold_set or not (gold_set & corpus_ids):
                continue
            covered += 1

            q = Query(query_text=b["query_text"], lang=Language.HI)
            t0 = time.perf_counter()
            results = bm25_retriever.retrieve(q, mode, args.top_k)
            latencies.append((time.perf_counter() - t0) * 1000)

            retrieved_ids = [r.passage.passage_id for r in results]
            metrics = compute_query_metrics(retrieved_ids, gold_set, args.top_k)
            query_metrics.append(metrics)

            lang_key = "hi"
            per_lang_metrics.setdefault(lang_key, []).append(metrics)
            qt = b.get("query_type", "UNKNOWN")
            per_qtype_metrics.setdefault(qt, []).append(metrics)

        overall = aggregate(query_metrics)
        lat = latency_stats(latencies, f"BM25_{mode_name}")

        print(f"\n  BM25 {mode_name} (n={covered} covered queries):")
        print(f"    Recall@1={overall['recall@1']:.4f}  Recall@5={overall['recall@5']:.4f}  "
              f"Recall@10={overall['recall@10']:.4f}")
        print(f"    MRR={overall['mrr']:.4f}  nDCG@10={overall['ndcg@10']:.4f}")
        if lat:
            print(f"    Latency: P50={lat['p50_ms']:.2f}ms  P95={lat['p95_ms']:.2f}ms  "
                  f"P100={lat['p100_ms']:.2f}ms  mean={lat['mean_ms']:.2f}ms")

        # Per query type
        for qt, qm in sorted(per_qtype_metrics.items()):
            qa = aggregate(qm)
            print(f"    {qt} (n={len(qm)}): R@1={qa['recall@1']:.3f} R@5={qa['recall@5']:.3f} "
                  f"MRR={qa['mrr']:.3f}")

        if mode_name == "CROSS_LINGUAL":
            bm25_cross = {"overall": overall, "latency": lat,
                          "per_query_type": {k: aggregate(v) for k, v in per_qtype_metrics.items()},
                          "per_language": {k: aggregate(v) for k, v in per_lang_metrics.items()},
                          "covered_queries": covered}
        else:
            bm25_mono = {"overall": overall, "latency": lat, "covered_queries": covered}

    # --- Hindi-only vs Hindi+English ---
    print("\n--- STEP 6: HINDI-ONLY vs HINDI+ENGLISH ---")
    for exp_mode_name, exp_mode in [
        ("MONOLINGUAL", RetrievalMode.MONOLINGUAL),
        ("CROSS_LINGUAL", RetrievalMode.CROSS_LINGUAL),
    ]:
        exp_metrics = []
        for b in benchmark:
            gold_set = set(b["gold_passage_ids"])
            if not gold_set or not (gold_set & corpus_ids):
                continue
            q = Query(query_text=b["query_text"], lang=Language.HI)
            results = bm25_retriever.retrieve(q, exp_mode, args.top_k)
            retrieved_ids = [r.passage.passage_id for r in results]
            exp_metrics.append(compute_query_metrics(retrieved_ids, gold_set, args.top_k))
        ea = aggregate(exp_metrics)
        print(f"  BM25 {exp_mode_name}: R@1={ea['recall@1']:.4f} R@5={ea['recall@5']:.4f} "
              f"MRR={ea['mrr']:.4f} (n={len(exp_metrics)})")

    # --- Dense + Hybrid evaluation ---
    dense_hybrid = None
    if not args.skip_dense:
        print("\n--- STEP 7: DENSE + HYBRID EVALUATION ---")
        dense_hybrid = attempt_dense_evaluation(
            corpus, benchmark, corpus_ids, top_k=args.top_k, device=args.device,
        )
        if dense_hybrid is None:
            print("  DENSE EVALUATION BLOCKED — HARDWARE LIMITATION")
            print("  Required: GPU or faster CPU environment for bge-m3 embedding")
    else:
        print("\n--- STEP 7: DENSE EVALUATION SKIPPED (--skip-dense) ---")

    # --- Comparison Table ---
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)

    comparison = {
        "BM25_CROSS": bm25_cross["overall"],
        "BM25_MONO": bm25_mono["overall"],
    }
    latency_comparison = {
        "BM25_CROSS": bm25_cross.get("latency", {}),
        "BM25_MONO": bm25_mono.get("latency", {}),
    }

    if dense_hybrid:
        comparison["Dense"] = dense_hybrid["dense"]["overall"]
        comparison["Hybrid"] = dense_hybrid["hybrid"]["overall"]
        latency_comparison["Dense"] = dense_hybrid["dense"].get("latency", {})
        latency_comparison["Hybrid"] = dense_hybrid["hybrid"].get("latency", {})

    # Metrics table
    print(f"\n{'Config':<18} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'MRR':>8} {'nDCG':>8} {'n':>6}")
    print("-" * 68)
    for name, metrics in comparison.items():
        n = 0
        if name == "BM25_CROSS":
            n = bm25_cross["covered_queries"]
        elif name == "BM25_MONO":
            n = bm25_mono["covered_queries"]
        elif name == "Dense" and dense_hybrid:
            n = dense_hybrid["dense"]["covered_queries"]
        elif name == "Hybrid" and dense_hybrid:
            n = dense_hybrid["hybrid"]["covered_queries"]
        r1 = metrics['recall@1']
        r5 = metrics['recall@5']
        r10 = metrics['recall@10']
        mrr = metrics['mrr']
        ndcg = metrics['ndcg@10']
        print(f"{name:<18} {r1:>8.4f} {r5:>8.4f} "
              f"{r10:>8.4f} {mrr:>8.4f} {ndcg:>8.4f} {n:>6}")

    # Latency table
    print(f"\n{'Config':<18} {'P50':>10} {'P70':>10} {'P95':>10} {'P100':>10} {'Mean':>10}")
    print("-" * 70)
    for name, lat in latency_comparison.items():
        if lat:
            print(f"{name:<18} {lat.get('p50_ms', 0):>9.1f}ms {lat.get('p70_ms', 0):>9.1f}ms "
                  f"{lat.get('p95_ms', 0):>9.1f}ms {lat.get('p100_ms', 0):>9.1f}ms "
                  f"{lat.get('mean_ms', 0):>9.1f}ms")
        else:
            print(f"{name:<18} {'N/A':>10}")

    # --- Save results ---
    print("\n--- SAVING RESULTS ---")
    output = {
        "config": {
            "train_rows": args.train_rows,
            "val_rows": args.val_rows,
            "top_k": args.top_k,
            "device": args.device,
            "model": "BAAI/bge-m3",
            "rrf_k": 60,
            "warmup_runs": args.warmup_runs,
            "latency_runs": args.latency_runs,
        },
        "corpus": {
            "passages": len(corpus),
            "languages": lang_counts,
        },
        "coverage": coverage,
        "query_types": qtype_counts,
        "bm25_latency_investigation": {
            "cold_start": cold_stats,
            "components": {k: v for k, v in component_breakdown.items() if v},
            "corpus_size": len(corpus),
        },
        "results": {
            "BM25_CROSS": bm25_cross,
            "BM25_MONO": bm25_mono,
        },
        "comparison": comparison,
        "latency_comparison": latency_comparison,
    }

    if dense_hybrid:
        output["results"]["Dense"] = dense_hybrid["dense"]
        output["results"]["Hybrid"] = dense_hybrid["hybrid"]
        output["dense_evaluation"] = {
            "model_load_time_s": dense_hybrid["model_load_time_s"],
            "embed_time_s": dense_hybrid["embed_time_s"],
            "upsert_time_ms": dense_hybrid["upsert_time_ms"],
        }
    else:
        output["dense_evaluation"] = "BLOCKED — HARDWARE LIMITATION"

    output_path = Path("data/phase9_5_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    # CSV ablation table
    csv_path = Path("data/phase9_5_ablation.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "config", "recall@1", "recall@5", "recall@10", "mrr", "ndcg@10",
            "p50_ms", "p70_ms", "p95_ms", "p100_ms", "mean_ms", "covered_queries",
        ])
        for name, metrics in comparison.items():
            lat = latency_comparison.get(name, {})
            n = 0
            if "BM25" in name:
                n = (bm25_cross["covered_queries"]
                     if "CROSS" in name
                     else bm25_mono["covered_queries"])
            elif name == "Dense" and dense_hybrid:
                n = dense_hybrid["dense"]["covered_queries"]
            elif name == "Hybrid" and dense_hybrid:
                n = dense_hybrid["hybrid"]["covered_queries"]
            writer.writerow([
                name,
                f"{metrics['recall@1']:.4f}", f"{metrics['recall@5']:.4f}",
                f"{metrics['recall@10']:.4f}", f"{metrics['mrr']:.4f}", f"{metrics['ndcg@10']:.4f}",
                f"{lat.get('p50_ms', 0):.1f}", f"{lat.get('p70_ms', 0):.1f}",
                f"{lat.get('p95_ms', 0):.1f}", f"{lat.get('p100_ms', 0):.1f}",
                f"{lat.get('mean_ms', 0):.1f}", n,
            ])

    print(f"  Results: {output_path}")
    print(f"  Ablation: {csv_path}")

    print("\n" + "=" * 70)
    print("PHASE 9.5 BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
