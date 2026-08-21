"""
Phase 5: bge-m3 Runtime Validation

Attempts to load BAAI/bge-m3, embed real multilingual MSMARCO-XI passages,
build a small Qdrant index, and run real vector retrieval.

Reports actual measured numbers. Does NOT fabricate results.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Fix Windows console encoding for Unicode output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def load_real_passages(max_rows: int = 200) -> list[dict]:
    """Load real passages from downloaded MSMARCO-XI parquet files."""
    import pyarrow.parquet as pq

    train_path = Path("data/cache/train/hintrain.parquet")
    if not train_path.exists():
        print(f"ERROR: {train_path} not found")
        sys.exit(1)

    pf = pq.ParquetFile(str(train_path))
    passages: list[dict] = []
    seen_texts: set[str] = set()

    for batch in pf.iter_batches(batch_size=max_rows):
        passages_col = batch.column("passages")
        lang_col = batch.column("source_lang")
        for i in range(batch.num_rows):
            p = passages_col[i].as_py() or {}
            en_list = p.get("English_passages", []) or []
            hi_list = p.get("Translated_passages", []) or []
            lang_code = lang_col[i].as_py() or "unknown"
            for text in en_list:
                if isinstance(text, str) and text.strip() and text not in seen_texts:
                    passages.append({"text": text, "lang": "en", "src_lang": lang_code})
                    seen_texts.add(text)
            for text in hi_list:
                if isinstance(text, str) and text.strip() and text not in seen_texts:
                    passages.append({"text": text, "lang": "hi", "src_lang": lang_code})
                    seen_texts.add(text)
            if len(passages) >= max_rows:
                break
        if len(passages) >= max_rows:
            break

    return passages[:max_rows]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Phase 5: bge-m3 validation")
    parser.add_argument("--num-passages", type=int, default=50, help="Number of passages to embed")
    parser.add_argument("--batch-size", type=int, default=16, help="Embedding batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu/cuda/mps")
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 5: BGE-M3 RUNTIME VALIDATION")
    print("=" * 70)

    # --- Step 1: Model Loading ---
    print("\n--- STEP 1: MODEL LOADING ---")
    t0 = time.perf_counter()
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("BAAI/bge-m3", device=args.device)
        load_time = time.perf_counter() - t0
        print(f"  Model loaded in {load_time:.2f}s")
        print(f"  Device: {args.device}")

        # Check dimension from actual model
        test_vec = model.encode(["test"], normalize_embeddings=True)
        actual_dim = test_vec.shape[1]
        actual_dtype = str(test_vec.dtype)
        print(f"  Actual embedding dimension: {actual_dim}")
        print(f"  Actual dtype: {actual_dtype}")

        # Check normalization
        import numpy as np

        norm = np.linalg.norm(test_vec[0])
        print(f"  Vector L2 norm: {norm:.6f} (should be ~1.0 if normalized)")

        # Memory estimate
        try:
            import psutil

            proc = psutil.Process()
            mem_mb = proc.memory_info().rss / 1024 / 1024
            print(f"  Process memory: {mem_mb:.1f} MB")
        except ImportError:
            print("  Memory: psutil not available")

    except Exception as e:
        load_time = time.perf_counter() - t0
        print(f"  MODEL LOADING FAILED after {load_time:.2f}s")
        print(f"  Error: {type(e).__name__}: {e}")
        print("\nPHASE 5 BLOCKED — Model loading failed.")
        sys.exit(1)

    # --- Step 2: Load real passages ---
    print("\n--- STEP 2: LOADING REAL MULTILINGUAL PASSAGES ---")
    passages = load_real_passages(max_rows=args.num_passages)
    print(f"  Loaded {len(passages)} unique passages")
    lang_counts: dict[str, int] = {}
    for p in passages:
        lang_counts[p["lang"]] = lang_counts.get(p["lang"], 0) + 1
    print(f"  Language distribution: {lang_counts}")

    # --- Step 3: Embed passages ---
    print("\n--- STEP 3: EMBEDDING PASSAGES ---")
    texts = [p["text"] for p in passages]
    batch_size = args.batch_size

    t0 = time.perf_counter()
    all_vectors = []
    batch_latencies: list[float] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        bt0 = time.perf_counter()
        vecs = model.encode(
            batch,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        bt = (time.perf_counter() - bt0) * 1000
        batch_latencies.append(bt)
        all_vectors.extend([v.tolist() for v in vecs])

    embed_time = time.perf_counter() - t0
    throughput = len(texts) / embed_time if embed_time > 0 else 0

    print(f"  Embedded {len(texts)} passages in {embed_time:.2f}s")
    print(f"  Throughput: {throughput:.1f} passages/sec")
    print(f"  Batch latencies: min={min(batch_latencies):.1f}ms "
          f"max={max(batch_latencies):.1f}ms "
          f"mean={sum(batch_latencies)/len(batch_latencies):.1f}ms")

    # Verify all vectors have correct dimension
    bad_dims = [i for i, v in enumerate(all_vectors) if len(v) != actual_dim]
    if bad_dims:
        print(f"  ERROR: {len(bad_dims)} vectors have wrong dimension!")
    else:
        print(f"  All {len(all_vectors)} vectors have correct dimension: {actual_dim}")

    # --- Step 4: Query embedding ---
    print("\n--- STEP 4: QUERY EMBEDDING ---")
    test_queries = [
        ("artificial intelligence", "en"),
        ("\u092d\u093e\u0930\u0924 \u0915\u0940 \u0930\u093e\u091c\u0927\u093e\u0928\u0940", "hi"),
    ]
    query_vectors: list[list[float]] = []
    for q_text, _q_lang in test_queries:
        t0 = time.perf_counter()
        q_vec = model.encode([q_text], normalize_embeddings=True)
        q_time = (time.perf_counter() - t0) * 1000
        q_list = q_vec[0].tolist()
        query_vectors.append(q_list)
        print(f"  Query '{q_text[:40]}...' -> dim={len(q_list)}, time={q_time:.1f}ms")

    # --- Step 5: Build Qdrant index ---
    print("\n--- STEP 5: BUILDING QDRANT INDEX ---")
    from ingestion.representation.base import create_passage_representation
    from retrieval.dense.qdrant_index import QdrantIndexManager

    representations = []
    for i, p in enumerate(passages):
        repr_obj = create_passage_representation(
            passage_id=f"passage_{i:06d}",
            text=p["text"],
            lang=p["lang"],
        )
        representations.append(repr_obj)

    manager = QdrantIndexManager(collection_name="phase5_validation")
    t0 = time.perf_counter()
    manager.create_collection()
    collection_time = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    n_upserted = manager.upsert_passages(representations, all_vectors, batch_size=100)
    upsert_time = (time.perf_counter() - t0) * 1000

    info = manager.get_collection_info()
    print(f"  Collection created in {collection_time:.1f}ms")
    print(f"  Upserted {n_upserted} points in {upsert_time:.1f}ms")
    print(f"  Collection info: {info}")

    # --- Step 6: Real vector retrieval ---
    print("\n--- STEP 6: REAL VECTOR RETRIEVAL ---")
    retrieval_latencies_embed: list[float] = []
    retrieval_latencies_search: list[float] = []
    retrieval_latencies_total: list[float] = []

    for q_text, _q_lang in test_queries:
        # Measure embedding latency
        t0 = time.perf_counter()
        q_vec = model.encode([q_text], normalize_embeddings=True)[0].tolist()
        embed_ms = (time.perf_counter() - t0) * 1000

        # Measure search latency
        t0 = time.perf_counter()
        results = manager.search(q_vec, top_k=5)
        search_ms = (time.perf_counter() - t0) * 1000

        total_ms = embed_ms + search_ms
        retrieval_latencies_embed.append(embed_ms)
        retrieval_latencies_search.append(search_ms)
        retrieval_latencies_total.append(total_ms)

        print(f"\n  Query: '{q_text[:50]}'")
        print(
            f"  Embedding: {embed_ms:.1f}ms"
            f" | Search: {search_ms:.1f}ms"
            f" | Total: {total_ms:.1f}ms"
        )
        for j, r in enumerate(results[:3]):
            text_preview = r["text"][:60].replace("\n", " ")
            print(f"    #{j+1} score={r['score']:.4f} lang={r['lang']} "
                  f"passage_id={r['passage_id'][:20]}...")
            print(f"       \"{text_preview}...\"")

    # Also run latency benchmark across all passages
    print("\n--- LATENCY BENCHMARK (100 runs) ---")
    bench_latencies_embed: list[float] = []
    bench_latencies_search: list[float] = []
    bench_latencies_total: list[float] = []

    bench_query = "machine learning algorithms"
    for _ in range(100):
        t0 = time.perf_counter()
        q_vec = model.encode([bench_query], normalize_embeddings=True)[0].tolist()
        embed_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        manager.search(q_vec, top_k=10)
        search_ms = (time.perf_counter() - t0) * 1000

        bench_latencies_embed.append(embed_ms)
        bench_latencies_search.append(search_ms)
        bench_latencies_total.append(embed_ms + search_ms)

    def percentile(data: list[float], p: float) -> float:
        sorted_d = sorted(data)
        idx = int(len(sorted_d) * p / 100)
        return sorted_d[min(idx, len(sorted_d) - 1)]

    print("\n  Query embedding latency:")
    print(f"    P50={percentile(bench_latencies_embed, 50):.1f}ms "
          f"P70={percentile(bench_latencies_embed, 70):.1f}ms "
          f"P95={percentile(bench_latencies_embed, 95):.1f}ms "
          f"P100={percentile(bench_latencies_embed, 100):.1f}ms")

    print("\n  Qdrant search latency:")
    print(f"    P50={percentile(bench_latencies_search, 50):.1f}ms "
          f"P70={percentile(bench_latencies_search, 70):.1f}ms "
          f"P95={percentile(bench_latencies_search, 95):.1f}ms "
          f"P100={percentile(bench_latencies_search, 100):.1f}ms")

    print("\n  End-to-end (embed + search):")
    print(f"    P50={percentile(bench_latencies_total, 50):.1f}ms "
          f"P70={percentile(bench_latencies_total, 70):.1f}ms "
          f"P95={percentile(bench_latencies_total, 95):.1f}ms "
          f"P100={percentile(bench_latencies_total, 100):.1f}ms")

    # --- Step 7: Save results ---
    results_data = {
        "model": "BAAI/bge-m3",
        "load_time_s": load_time,
        "device": args.device,
        "actual_dimension": actual_dim,
        "dtype": actual_dtype,
        "passages_embedded": len(texts),
        "embed_time_s": embed_time,
        "throughput_per_sec": throughput,
        "collection_info": info,
        "upsert_count": n_upserted,
        "upsert_time_ms": upsert_time,
        "latency_benchmark": {
            "n_runs": 100,
            "embed_p50_ms": percentile(bench_latencies_embed, 50),
            "embed_p70_ms": percentile(bench_latencies_embed, 70),
            "embed_p95_ms": percentile(bench_latencies_embed, 95),
            "embed_p100_ms": percentile(bench_latencies_embed, 100),
            "search_p50_ms": percentile(bench_latencies_search, 50),
            "search_p70_ms": percentile(bench_latencies_search, 70),
            "search_p95_ms": percentile(bench_latencies_search, 95),
            "search_p100_ms": percentile(bench_latencies_search, 100),
            "total_p50_ms": percentile(bench_latencies_total, 50),
            "total_p70_ms": percentile(bench_latencies_total, 70),
            "total_p95_ms": percentile(bench_latencies_total, 95),
            "total_p100_ms": percentile(bench_latencies_total, 100),
        },
        "quality_evaluation": "NOT MEANINGFUL — ZERO GOLD COVERAGE",
    }

    output_path = Path("data/phase5_validation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {output_path}")

    # Cleanup
    manager.delete_collection()
    del model
    gc.collect()

    print("\n" + "=" * 70)
    print("PHASE 5 VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
