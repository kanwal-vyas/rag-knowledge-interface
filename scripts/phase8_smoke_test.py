"""
Phase 8: Hybrid Retrieval Smoke Test

Runs BM25 + bge-m3 dense + RRF on real MSMARCO-XI data.
Measures component and end-to-end latency.
"""
from __future__ import annotations

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


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Phase 8: Hybrid retrieval smoke test")
    parser.add_argument("--num-passages", type=int, default=30, help="Number of passages to embed")
    parser.add_argument("--device", type=str, default="cpu", help="Device for bge-m3")
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 8: HYBRID RETRIEVAL SMOKE TEST")
    print("=" * 70)

    # --- Step 1: Load real passages ---
    print("\n--- STEP 1: LOADING REAL PASSAGES ---")
    import pyarrow.parquet as pq

    train_path = Path("data/cache/train/hintrain.parquet")
    if not train_path.exists():
        print(f"ERROR: {train_path} not found")
        sys.exit(1)

    pf = pq.ParquetFile(str(train_path))
    passages: list[dict] = []
    seen_texts: set[str] = set()

    for batch in pf.iter_batches(batch_size=200):
        passages_col = batch.column("passages")
        for i in range(batch.num_rows):
            p = passages_col[i].as_py() or {}
            en_list = p.get("English_passages", []) or []
            hi_list = p.get("Translated_passages", []) or []
            for text in en_list:
                if isinstance(text, str) and text.strip() and text not in seen_texts:
                    passages.append({"text": text, "lang": "en"})
                    seen_texts.add(text)
            for text in hi_list:
                if isinstance(text, str) and text.strip() and text not in seen_texts:
                    passages.append({"text": text, "lang": "hi"})
                    seen_texts.add(text)
            if len(passages) >= args.num_passages:
                break
        if len(passages) >= args.num_passages:
            break

    passages = passages[:args.num_passages]
    print(f"  Loaded {len(passages)} passages")

    # --- Step 2: Build BM25 index ---
    print("\n--- STEP 2: BUILDING BM25 INDEX ---")
    from ingestion.normalization.text import normalize_text
    from retrieval.sparse.bm25_index import BM25Index
    from retrieval.sparse.bm25_retriever import BM25SparseRetriever

    bm25_index = BM25Index(use_bigrams=True)
    for i, p in enumerate(passages):
        bm25_index.add_document(f"passage_{i:06d}", normalize_text(p["text"]), p["lang"])
    bm25_stats = bm25_index.build()
    print(f"  BM25: {bm25_stats.document_count} docs, vocab={bm25_stats.vocab_size}, "
          f"build={bm25_stats.build_time_ms:.1f}ms")

    sparse_retriever = BM25SparseRetriever(index=bm25_index)

    # --- Step 3: Load bge-m3 and build Qdrant index ---
    print("\n--- STEP 3: LOADING BGE-M3 AND BUILDING QDRANT INDEX ---")
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
    for i, p in enumerate(passages):
        repr_obj = create_passage_representation(
            passage_id=f"passage_{i:06d}",
            text=p["text"],
            lang=p["lang"],
        )
        representations.append(repr_obj)

    # Embed all passages
    texts = [p["text"] for p in passages]
    t0 = time.perf_counter()
    vectors = model.encode(texts, normalize_embeddings=True, batch_size=8, show_progress_bar=False)
    embed_time = time.perf_counter() - t0
    vectors_list = [v.tolist() for v in vectors]
    print(f"  Embedded {len(texts)} passages in {embed_time:.2f}s")

    # Build Qdrant index
    qdrant = QdrantIndexManager(collection_name="phase8_smoke")
    qdrant.create_collection()
    t0 = time.perf_counter()
    qdrant.upsert_passages(representations, vectors_list, batch_size=100)
    upsert_time = (time.perf_counter() - t0) * 1000
    print(f"  Qdrant upsert: {upsert_time:.1f}ms")

    # Create embedding provider wrapper
    provider = BgeM3EmbeddingProvider(device=args.device)
    provider._model = model  # Inject already-loaded model
    dense_retriever = BgeM3DenseRetriever(qdrant_manager=qdrant, embedding_provider=provider)

    # --- Step 4: Hybrid retrieval ---
    print("\n--- STEP 4: HYBRID RETRIEVAL ---")
    from app.models.retrieval import Language, Query, RetrievalMode
    from retrieval.fusion.hybrid_retriever import HybridRetriever

    hybrid = HybridRetriever(
        sparse_retriever=sparse_retriever,
        dense_retriever=dense_retriever,
    )

    test_queries = [
        ("artificial intelligence", "en"),
        ("capital of India", "en"),
    ]

    for q_text, q_lang in test_queries:
        query = Query(query_text=q_text, lang=Language(q_lang))

        # Sparse only
        t0 = time.perf_counter()
        sparse_results = sparse_retriever.retrieve(query, RetrievalMode.CROSS_LINGUAL, top_k=5)
        sparse_ms = (time.perf_counter() - t0) * 1000

        # Dense only
        t0 = time.perf_counter()
        dense_results = dense_retriever.retrieve(query, RetrievalMode.CROSS_LINGUAL, top_k=5)
        dense_ms = (time.perf_counter() - t0) * 1000

        # Hybrid
        t0 = time.perf_counter()
        hybrid_results = hybrid.retrieve(query, RetrievalMode.CROSS_LINGUAL, top_k=5)
        hybrid_ms = (time.perf_counter() - t0) * 1000

        print(f"\n  Query: '{q_text}'")
        print(f"  Sparse: {len(sparse_results)} results in {sparse_ms:.1f}ms")
        for j, r in enumerate(sparse_results[:3]):
            print(f"    #{j+1} score={r.score:.4f} pid={r.passage.passage_id[:20]}...")
        print(f"  Dense:  {len(dense_results)} results in {dense_ms:.1f}ms")
        for j, r in enumerate(dense_results[:3]):
            print(f"    #{j+1} score={r.score:.4f} pid={r.passage.passage_id[:20]}...")
        print(f"  Hybrid: {len(hybrid_results)} results in {hybrid_ms:.1f}ms")
        for j, r in enumerate(hybrid_results[:3]):
            pid = r.passage.passage_id[:20]
            print(
                f"    #{j+1} score={r.score:.4f}"
                f" source={r.source} pid={pid}..."
            )

    # --- Step 5: Latency benchmark ---
    print("\n--- STEP 5: LATENCY BENCHMARK (20 runs) ---")
    bench_query = Query(query_text="machine learning algorithms", lang=Language.EN)

    sparse_latencies: list[float] = []
    dense_latencies: list[float] = []
    hybrid_latencies: list[float] = []

    for _ in range(20):
        t0 = time.perf_counter()
        sparse_retriever.retrieve(bench_query, RetrievalMode.CROSS_LINGUAL, top_k=10)
        sparse_latencies.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        dense_retriever.retrieve(bench_query, RetrievalMode.CROSS_LINGUAL, top_k=10)
        dense_latencies.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        hybrid.retrieve(bench_query, RetrievalMode.CROSS_LINGUAL, top_k=10)
        hybrid_latencies.append((time.perf_counter() - t0) * 1000)

    print("\n  BM25 latency:")
    print(f"    P50={percentile(sparse_latencies, 50):.1f}ms "
          f"P95={percentile(sparse_latencies, 95):.1f}ms "
          f"P100={percentile(sparse_latencies, 100):.1f}ms")
    print("  Dense latency (embed+search):")
    print(f"    P50={percentile(dense_latencies, 50):.1f}ms "
          f"P95={percentile(dense_latencies, 95):.1f}ms "
          f"P100={percentile(dense_latencies, 100):.1f}ms")
    print("  Hybrid latency (BM25+dense+RRF):")
    print(f"    P50={percentile(hybrid_latencies, 50):.1f}ms "
          f"P95={percentile(hybrid_latencies, 95):.1f}ms "
          f"P100={percentile(hybrid_latencies, 100):.1f}ms")

    # --- Save results ---
    results_data = {
        "passages": len(passages),
        "load_time_s": load_time,
        "embed_time_s": embed_time,
        "upsert_time_ms": upsert_time,
        "latency": {
            "n_runs": 20,
            "bm25_p50_ms": percentile(sparse_latencies, 50),
            "bm25_p95_ms": percentile(sparse_latencies, 95),
            "bm25_p100_ms": percentile(sparse_latencies, 100),
            "dense_p50_ms": percentile(dense_latencies, 50),
            "dense_p95_ms": percentile(dense_latencies, 95),
            "dense_p100_ms": percentile(dense_latencies, 100),
            "hybrid_p50_ms": percentile(hybrid_latencies, 50),
            "hybrid_p95_ms": percentile(hybrid_latencies, 95),
            "hybrid_p100_ms": percentile(hybrid_latencies, 100),
        },
        "quality_evaluation": "NOT MEANINGFUL — ZERO GOLD COVERAGE",
    }
    output_path = Path("data/phase8_smoke_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results_data, indent=2), encoding="utf-8")
    print(f"\nResults saved to {output_path}")

    # Cleanup
    qdrant.delete_collection()
    del model
    gc.collect()

    print("\n" + "=" * 70)
    print("PHASE 8 SMOKE TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
