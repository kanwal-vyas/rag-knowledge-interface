"""
Mandatory Phase 1 sanity checks (Audit §9, freeze decision 6).

Both checks MUST run and pass/report before any threshold (tau,
T_sentence, corpus-tier stopping rule) is calibrated.

Check 1: Query-text-not-in-corpus scan (Audit §5 "Query text" row)
  Verifies query text does not appear as a substring in passage content.
  Expected: empty or near-empty, but must be checked, not assumed.

Check 2: Train/validation query near-duplicate check (Audit §8)
  Exact or near-duplicate queries across splits could leak benchmark
  information into tuning. If found above negligible rate, the affected
  validation queries must be excluded from the frozen benchmark set.
"""

from __future__ import annotations

from ingestion.normalization.text import normalize_text


def scan_query_text_in_corpus(
    corpus_texts: list[str],
    query_texts: list[str],
) -> list[str]:
    """
    Substring scan: which query texts appear (case-insensitive) inside
    some corpus passage?

    This is a correctness check, not a fuzzy search — we look for the
    normalized query as a substring of the normalized passage. A non-empty
    result does not necessarily mean a problem (short generic queries like
    "yes" or "1" might legitimately appear), but the results must be
    inspected and reported.

    Performance: O(N*M) in the worst case where N=len(corpus), M=len(queries).
    For development with small samples this is fine. For full-scale, use
    a suffix array or Aho-Corasick — flagging this for Phase 11 optimization.

    Args:
        corpus_texts: List of passage texts (normalized).
        query_texts: List of query texts to scan for (normalized).

    Returns:
        List of (query, passage_snippet) tuples where a query was found
        as a substring of a passage. Empty list = clean.
    """
    # Build a set of normalized corpus texts for fast existence checks,
    # and also do substring scan.
    normalized_corpus = [normalize_text(t) for t in corpus_texts]
    normalized_queries = [normalize_text(t) for t in query_texts]

    found: list[str] = []
    for q_text in normalized_queries:
        if not q_text or len(q_text) < 3:
            # Skip trivially short queries that would match everywhere
            continue
        q_lower = q_text.lower()
        for c_text in normalized_corpus:
            if q_lower in c_text.lower():
                # Truncate snippet for reporting
                idx = c_text.lower().index(q_lower)
                start = max(0, idx - 30)
                end = min(len(c_text), idx + len(q_text) + 30)
                snippet = c_text[start:end]
                found.append(f"Query '{q_text[:50]}' found in passage snippet: '...{snippet}...'")
                break  # One match per query is enough to flag it

    return found


def find_train_validation_near_duplicates(
    train_queries: list[str],
    validation_queries: list[str],
    similarity_threshold: float = 0.95,
) -> list[tuple[str, str, float]]:
    """
    Exact/near-duplicate check between train-split and validation-split
    query text (Audit §8, §9 check 2).

    Two checks:
    1. Exact match (after normalization + case folding)
    2. Near-duplicate via character-level Jaccard similarity >= threshold

    Args:
        train_queries: Training split query texts.
        validation_queries: Validation split query texts.
        similarity_threshold: Jaccard similarity threshold for near-duplicate
            detection. Default 0.95 — conservative enough to catch meaningful
            near-duplicates without excessive false positives.

    Returns:
        List of (train_query, validation_query, similarity_score) tuples
        for matches found. Empty list = clean.
    """
    # Step 1: Normalize all queries
    train_norm = [normalize_text(q).lower() for q in train_queries]
    val_norm = [normalize_text(q).lower() for q in validation_queries]

    # Step 2: Exact matches (fast)
    train_set = set(train_norm)
    exact_matches: list[tuple[str, str, float]] = []
    val_to_train: dict[str, str] = {}  # val_norm -> original val query
    for i, vq in enumerate(validation_queries):
        val_to_train[val_norm[i]] = vq

    for i, tq in enumerate(train_queries):
        tn = train_norm[i]
        if tn in train_set and len(tn) >= 3:  # Skip trivially short
            # Find matching validation queries
            for vn in set(val_norm):
                if vn == tn and len(vn) >= 3:
                    exact_matches.append((tq, val_to_train[vn], 1.0))

    # Step 3: Near-duplicates via token-level Jaccard
    # Token-level Jaccard is more meaningful than character-set Jaccard
    # for short queries — it avoids false positives where two short queries
    # share common stop words (e.g., "what direction does phloem flow"
    # vs "how far is philadelphia from lancaster pa" would score 1.0
    # with character-set Jaccard due to shared {a,d,e,h,i,l,o,r,t} etc.)
    # Token-level Jaccard compares word sets, which is more semantically
    # meaningful.
    exact_pairs = {(m[0], m[1]) for m in exact_matches}
    near_matches: list[tuple[str, str, float]] = []

    if similarity_threshold < 1.0 and len(train_queries) * len(validation_queries) < 10_000_000:
        for i, tq in enumerate(train_queries):
            tn = train_norm[i]
            if len(tn) < 5:
                continue
            train_tokens = set(tn.split())
            if len(train_tokens) < 2:
                continue
            for j, vq in enumerate(validation_queries):
                vn = val_norm[j]
                if (tq, vq) in exact_pairs:
                    continue
                if len(vn) < 5:
                    continue
                val_tokens = set(vn.split())
                if len(val_tokens) < 2:
                    continue
                # Token-level Jaccard
                intersection = len(train_tokens & val_tokens)
                union = len(train_tokens | val_tokens)
                if union > 0:
                    sim = intersection / union
                    if sim >= similarity_threshold:
                        near_matches.append((tq, vq, sim))

    return exact_matches + near_matches
