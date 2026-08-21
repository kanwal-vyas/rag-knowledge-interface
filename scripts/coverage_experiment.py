"""
Coverage experiment: measure corpus size → validation gold coverage.

Architecture §3.2: coverage-awareness is a post-hoc backstop, not a
construction method. This script measures the empirical relationship
between corpus tier size and benchmark gold-passage coverage.

The exact tier sizes remain UNFROZEN. This experiment informs tier selection.

Usage:
    python -m scripts.coverage_experiment [--rows 1000] [--seeds 42,123]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.dataset.corpus import CorpusBuilder, check_corpus_coverage
from ingestion.dataset.eval_split import extract_eval_dataset
from ingestion.dataset.loader import iter_rows


def run_coverage_experiment(
    max_rows: int = 1000,
    seeds: list[int] | None = None,
    output_path: str | None = None,
) -> dict:
    """
    Run the coverage experiment.

    Args:
        max_rows: Number of training rows to process for corpus building.
        seeds: Random seeds for reproducibility (unused currently, reserved).
        output_path: Where to save JSON results.

    Returns:
        Dict with experiment results.
    """
    if seeds is None:
        seeds = [42]

    print(f"Coverage experiment: processing {max_rows} rows...")
    t0 = time.time()

    # Step 1: Load training rows for corpus building
    print("[1] Loading training rows...")
    train_rows = []
    for row in iter_rows(split="train", sample_size=max_rows, seed=seeds[0]):
        train_rows.append(row)
    print(f"    Loaded {len(train_rows)} training rows")

    # Step 2: Load validation rows for benchmark gold set
    print("[2] Loading validation rows for benchmark...")
    val_rows = []
    for row in iter_rows(split="validation", sample_size=500, seed=seeds[0]):
        val_rows.append(row)
    print(f"    Loaded {len(val_rows)} validation rows")

    # Step 3: Build eval dataset from validation rows (benchmark pool)
    print("[3] Extracting benchmark eval dataset...")
    eval_dataset = extract_eval_dataset(iter(val_rows), pool="benchmark")
    # Build gold map: query_id -> list of gold passage_ids
    benchmark_gold: dict[int, list[str]] = {}
    for q in eval_dataset.queries:
        gold_ids = [gl.passage_id for gl in q.gold_labels if gl.is_selected]
        if gold_ids:
            benchmark_gold[q.query_id] = gold_ids
    print(f"    {len(benchmark_gold)} benchmark queries with gold passages")

    # Step 4: Build corpus at increasing sizes and measure coverage
    print("[4] Building corpora at increasing sizes...")
    tier_sizes = [100, 250, 500, 750, 1000, min(2000, max_rows)]

    results: list[dict] = []
    for tier_size in tier_sizes:
        if tier_size > len(train_rows):
            break

        builder = CorpusBuilder(
            tier_name=f"T_exp_{tier_size}",
            target_passages=tier_size * 20,  # generous capacity
            source_query_rows=tier_size,
        )
        for row in train_rows[:tier_size]:
            builder.add_row(row)

        corpus, stats = builder.build()
        coverage = check_corpus_coverage(corpus, benchmark_gold)

        entry = {
            "tier_size_rows": tier_size,
            "corpus_unique_passages": stats.unique_passages,
            "corpus_raw_passages": stats.raw_passages,
            "duplicate_pct": round(stats.duplicate_percentage, 1),
            "lang_counts": stats.lang_counts,
            "lang_unique": stats.lang_unique_counts,
            "benchmark_queries": coverage["total_queries"],
            "fully_covered": coverage["fully_covered_queries"],
            "partially_covered": coverage["partially_covered_queries"],
            "uncovered": coverage["uncovered_queries"],
            "coverage_rate": round(coverage["coverage_rate"], 4),
            "full_coverage_rate": round(coverage["full_coverage_rate"], 4),
        }
        results.append(entry)
        print(
            f"    Rows={tier_size:>5} | Passages={stats.unique_passages:>6} | "
            f"Coverage={coverage['coverage_rate']:.1%} | "
            f"Full={coverage['full_coverage_rate']:.1%}"
        )

    elapsed = time.time() - t0

    experiment = {
        "max_rows_processed": max_rows,
        "benchmark_queries": len(benchmark_gold),
        "seeds": seeds,
        "elapsed_seconds": round(elapsed, 1),
        "tiers": results,
    }

    # Save results
    if output_path is None:
        output_path = "data/coverage_experiment.json"

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(experiment, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[5] Results saved to {out}")

    return experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Coverage experiment")
    parser.add_argument("--rows", type=int, default=1000, help="Max training rows")
    parser.add_argument("--seeds", type=str, default="42", help="Comma-separated seeds")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    run_coverage_experiment(
        max_rows=args.rows,
        seeds=seeds,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
