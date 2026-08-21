"""
Phase 2 smoke test: Download exactly 5 rows from MSMARCO-XI via streaming
to verify schema, pairing, and assumptions before building any corpus.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def main() -> None:
    print("=" * 60)
    print("MSMARCO-XI SCHEMA SMOKE TEST (5 rows, streaming)")
    print("=" * 60)

    # Step 1: Load 5 rows via streaming
    print("\n[1] Loading 5 rows from train split (streaming)...")
    t0 = time.time()

    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
        ds = load_dataset("ai4bharat/MSMARCO-XI", split="train", streaming=True)
    except Exception as e:
        print(f"FAILED to load dataset: {e}")
        sys.exit(1)

    rows = []
    for i, row in enumerate(ds):
        rows.append(row)
        if i >= 4:
            break

    elapsed = time.time() - t0
    print(f"    Loaded {len(rows)} rows in {elapsed:.1f}s")

    if len(rows) == 0:
        print("FAILED: No rows returned")
        sys.exit(1)

    # Step 2: Print raw schema of first row
    print("\n[2] RAW SCHEMA (first row keys):")
    first = rows[0]
    for key in first:
        val = first[key]
        if isinstance(val, dict):
            print(f"  {key}: dict with keys {list(val.keys())}")
        elif isinstance(val, list):
            item_type = type(val[0]).__name__ if val else 'empty'
            print(f"  {key}: list of {len(val)} items, type={item_type}")
        else:
            print(f"  {key}: {type(val).__name__} = {repr(val)[:80]}")

    # Step 3: Verify passages structure
    print("\n[3] PASSAGE STRUCTURE:")
    for idx, row in enumerate(rows):
        passages = row.get("passages", {})
        en_psgs = passages.get("English_passages", [])
        hi_psgs = passages.get("Translated_passages", [])
        is_sel = passages.get("is_selected", [])

        print(f"\n  Row {idx}: query_id={row.get('query_id')}, query_type={row.get('query_type')}")
        print(f"    English_passages: {len(en_psgs)} items")
        print(f"    Translated_passages: {len(hi_psgs)} items")
        print(f"    is_selected: {is_sel}")
        print(f"    Parallel length match: en={len(en_psgs)}, hi={len(hi_psgs)}, sel={len(is_sel)}")

        if len(en_psgs) != len(hi_psgs):
            print("    WARNING: English/Hindi passage count MISMATCH!")
        if len(en_psgs) != len(is_sel):
            print("    WARNING: Passage/is_selected count MISMATCH!")

        # Show first passage pair
        if en_psgs:
            print(f"    en[0]: {repr(en_psgs[0][:80])}")
        if hi_psgs:
            print(f"    hi[0]: {repr(hi_psgs[0][:80])}")

    # Step 4: Verify is_selected semantics
    print("\n[4] IS_SELECTED SEMANTICS:")
    for idx, row in enumerate(rows):
        passages = row.get("passages", {})
        is_sel = passages.get("is_selected", [])
        en_psgs = passages.get("English_passages", [])
        hi_psgs = passages.get("Translated_passages", [])

        selected_indices = [i for i, v in enumerate(is_sel) if v == 1]
        print(f"  Row {idx}: {len(selected_indices)} selected out of {len(is_sel)} passages")
        for si in selected_indices:
            en_text = en_psgs[si] if si < len(en_psgs) else "N/A"
            hi_text = hi_psgs[si] if si < len(hi_psgs) else "N/A"
            print(f"    is_selected[{si}]=1:")
            print(f"      en: {repr(en_text[:100])}")
            print(f"      hi: {repr(hi_text[:100])}")

    # Step 5: Verify query/language fields
    print("\n[5] QUERY & LANGUAGE FIELDS:")
    for idx, row in enumerate(rows):
        print(f"  Row {idx}:")
        print(f"    query_id: {row.get('query_id')} (type={type(row.get('query_id')).__name__})")
        print(f"    query_type: {row.get('query_type')}")
        print(f"    source_lang: {row.get('source_lang')}")
        print(f"    target_lang: {row.get('target_lang')}")
        print(f"    query (hindi): {repr(row.get('query', '')[:60])}")
        print(f"    Eng_Query: {repr(row.get('Eng_Query', '')[:60])}")
        print(f"    Answer: {repr(str(row.get('Answer', ''))[:60])}")
        print(f"    Eng_Answer: {repr(str(row.get('Eng_Answer', ''))[:60])}")

    # Step 6: Verify dual-language gold-label assumption
    print("\n[6] DUAL-LANGUAGE GOLD-LABEL ASSUMPTION CHECK:")
    print("  Architecture §7 says G_cross(q) = { passage_id(hi), passage_id(en) }")
    print("  for each selected passage index.")
    print()

    from ingestion.deduplication.canonical_id import canonical_passage_id
    from ingestion.normalization.text import normalize_text

    for idx, row in enumerate(rows):
        passages = row.get("passages", {})
        en_psgs = passages.get("English_passages", [])
        hi_psgs = passages.get("Translated_passages", [])
        is_sel = passages.get("is_selected", [])

        selected_indices = [i for i, v in enumerate(is_sel) if v == 1]
        if not selected_indices:
            print(f"  Row {idx}: No selected passages — skipping")
            continue

        print(f"  Row {idx}: {len(selected_indices)} selected passage(s)")
        for si in selected_indices:
            en_text = en_psgs[si] if si < len(en_psgs) else ""
            hi_text = hi_psgs[si] if si < len(hi_psgs) else ""

            en_id = canonical_passage_id(en_text) if en_text else None
            hi_id = canonical_passage_id(hi_text) if hi_text else None

            # Check: are en and hi the same passage (same hash)?
            same_hash = en_id == hi_id if en_id and hi_id else False
            # Check: are en and hi the same text (normalized)?
            en_norm = normalize_text(en_text) if en_text else ""
            hi_norm = normalize_text(hi_text) if hi_text else ""
            same_norm = en_norm == hi_norm

            print(f"    Index {si}:")
            print(f"      en passage_id: {en_id[:16] if en_id else 'N/A'}...")
            print(f"      hi passage_id: {hi_id[:16] if hi_id else 'N/A'}...")
            print(f"      Same canonical ID: {same_hash}")
            print(f"      Same normalized text: {same_norm}")
            print(f"      → Gold set for Config 2 should include BOTH: {not same_hash}")

    # Step 7: Check meta field
    print("\n[7] META FIELD:")
    for idx, row in enumerate(rows):
        meta = row.get("meta", {})
        print(f"  Row {idx}: {json.dumps(meta, indent=4) if meta else '(empty)'}")

    # Step 8: Summary
    print("\n" + "=" * 60)
    print("SMOKE TEST COMPLETE")
    print("=" * 60)
    print(f"\nRows loaded: {len(rows)}")
    all_passages = all(
        len(r.get('passages', {}).get('English_passages', [])) > 0
        for r in rows
    )
    all_selected = all(
        len(r.get('passages', {}).get('is_selected', [])) > 0
        for r in rows
    )
    print(f"All rows have passages: {all_passages}")
    print(f"All rows have is_selected: {all_selected}")

    # Check parallel structure
    parallel_ok = True
    for r in rows:
        p = r.get("passages", {})
        if len(p.get("English_passages", [])) != len(p.get("is_selected", [])):
            parallel_ok = False
    print(f"Parallel structure (en == is_selected length): {parallel_ok}")

    # Write schema to file for reference
    schema = {
        "row_keys": list(first.keys()),
        "passages_keys": list(first.get("passages", {}).keys()),
        "rows_loaded": len(rows),
        "parallel_ok": parallel_ok,
    }
    out = Path("data") / "schema_report.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(schema, indent=2))
    print(f"\nSchema report written to {out}")


if __name__ == "__main__":
    main()
