"""
Prepare a small demo index for the RAG API.

Loads from cached JSON samples (data/cache/sample_train.json, sample_val.json),
builds a BM25 index, and saves everything to artifacts/demo/.

Usage:
    python -m scripts.prepare_demo_index
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

DEMO_OUTPUT_DIR = Path("artifacts/demo")
TRAIN_JSON = Path("data/cache/sample_train.json")
VAL_JSON = Path("data/cache/sample_val.json")
SAMPLE_ROWS = 100  # rows from each split to use


def build_demo_index(output_dir: Path = DEMO_OUTPUT_DIR) -> dict:
    """Build a demo BM25 index from cached JSON samples."""
    start_time = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load JSON samples ──
    logger.info("loading_json_samples", rows=SAMPLE_ROWS)
    with open(TRAIN_JSON, encoding="utf-8") as f:
        train_data = json.load(f)[:SAMPLE_ROWS]
    with open(VAL_JSON, encoding="utf-8") as f:
        val_data = json.load(f)[:SAMPLE_ROWS]
    logger.info("samples_loaded", train=len(train_data), val=len(val_data))

    # ── Extract passages ──
    from ingestion.dataset.language import iso639_1_for
    from ingestion.deduplication.canonical_id import canonical_passage_id
    from ingestion.normalization.text import normalize_text

    passages: dict[str, dict] = {}
    lang_counts: dict[str, int] = {}
    raw_count = 0

    for row in train_data + val_data:
        src_bcp47 = row.get("source_lang", "")
        tgt_bcp47 = row.get("target_lang", "")
        try:
            src_lang = iso639_1_for(src_bcp47) if src_bcp47 else "en"
        except Exception:
            src_lang = "en"
        try:
            tgt_lang = iso639_1_for(tgt_bcp47) if tgt_bcp47 else "hi"
        except Exception:
            tgt_lang = "hi"

        en_passages = row.get("English_passages", [])
        hi_passages = row.get("Translated_passages", [])

        if not isinstance(en_passages, list):
            en_passages = []
        if not isinstance(hi_passages, list):
            hi_passages = []

        for p_text in en_passages:
            if p_text and isinstance(p_text, str) and p_text.strip():
                raw_count += 1
                normalized = normalize_text(p_text)
                pid = canonical_passage_id(normalized)
                if pid not in passages:
                    passages[pid] = {"text": normalized, "lang": src_lang}
                    lang_counts[src_lang] = lang_counts.get(src_lang, 0) + 1

        for p_text in hi_passages:
            if p_text and isinstance(p_text, str) and p_text.strip():
                raw_count += 1
                normalized = normalize_text(p_text)
                pid = canonical_passage_id(normalized)
                if pid not in passages:
                    passages[pid] = {"text": normalized, "lang": tgt_lang}
                    lang_counts[tgt_lang] = lang_counts.get(tgt_lang, 0) + 1

    logger.info("passages_extracted", unique=len(passages), raw=raw_count, lang_counts=lang_counts)

    # ── Build BM25 index ──
    logger.info("building_bm25_index", passages=len(passages))
    from retrieval.sparse.bm25_index import BM25Index

    bm25_index = BM25Index()
    for pid, p in passages.items():
        bm25_index.add_document(pid, p["text"], p["lang"])
    bm25_stats = bm25_index.build()

    # ── Save BM25 index ──
    bm25_path = output_dir / "bm25_index.json"
    bm25_index.save(bm25_path)

    # ── Save passage store ──
    passage_store = {pid: p["text"] for pid, p in passages.items()}
    store_path = output_dir / "passage_store.json"
    store_path.write_text(
        json.dumps(passage_store, ensure_ascii=False),
        encoding="utf-8",
    )

    # ── Build demo queries ──
    demo_queries = _build_demo_queries(passages)
    queries_path = output_dir / "demo_queries.json"
    queries_path.write_text(
        json.dumps(demo_queries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ── Save manifest ──
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    manifest = {
        "sample_rows_per_split": SAMPLE_ROWS,
        "total_passages": len(passages),
        "raw_passages": raw_count,
        "language_counts": lang_counts,
        "bm25_documents": bm25_stats.document_count,
        "bm25_vocab_size": bm25_stats.vocab_size,
        "bm25_build_time_ms": round(bm25_stats.build_time_ms, 1),
        "demo_queries_count": len(demo_queries),
        "total_build_time_ms": round(elapsed_ms, 1),
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info(
        "demo_index_prepared",
        passages=len(passages),
        bm25_docs=bm25_stats.document_count,
        queries=len(demo_queries),
        build_time_ms=round(elapsed_ms, 1),
    )

    return manifest


def _build_demo_queries(passages: dict[str, dict]) -> list[dict]:
    """Build curated demo queries from actual corpus content."""
    en_pids = [pid for pid, p in passages.items() if p["lang"] == "en"]
    hi_pids = [pid for pid, p in passages.items() if p["lang"] == "hi"]

    queries = []

    # Query 1: English — generic
    queries.append({
        "query": "What information is available about this topic?",
        "lang": "en",
        "expected_retrieval": True,
        "note": "Generic English query",
    })

    # Query 2: Hindi — simple
    queries.append({
        "query": "यह क्या है",
        "lang": "hi",
        "expected_retrieval": True,
        "note": "Simple Hindi query",
    })

    # Query 3: English — use actual passage content
    if en_pids:
        sample_text = passages[en_pids[0]]["text"]
        words = [w for w in sample_text.split() if len(w) > 3][:4]
        if words:
            queries.append({
                "query": " ".join(words),
                "lang": "en",
                "expected_retrieval": True,
                "note": "Uses terms from actual corpus",
            })

    # Query 4: Hindi — use actual passage content
    if hi_pids:
        sample_text = passages[hi_pids[0]]["text"]
        queries.append({
            "query": sample_text[:60],
            "lang": "hi",
            "expected_retrieval": True,
            "note": "Uses text from actual Hindi passage",
        })

    # Query 5: Out-of-domain
    queries.append({
        "query": "quantum computing superposition entanglement qubits",
        "lang": "en",
        "expected_retrieval": False,
        "note": "Out-of-domain — should find nothing",
    })

    return queries


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(_logging.INFO),
    )
    manifest = build_demo_index()
    print(json.dumps(manifest, indent=2))
