# HH Goa 2026 — Task 2: Architecture (Source of Truth)

**Status:** FROZEN. Approved for DeepSeek implementation.

**Provenance of this document.** This file is derived from exactly one source
document: `HH Goa 2026 — Task 2: Final Pre-Implementation Audit` (the "Audit"),
which is itself explicitly scoped as a **patch** on top of a prior document,
"Architecture V2" (§5, §16, §21, §22, §24 of V2 are patched; the rest of V2 is
declared unchanged but is **not reproduced** in the Audit).

**The full text of Architecture V2 is not available in this repository.**
Every section below is either (a) transcribed from content the Audit actually
states, or (b) explicitly marked as missing. Nothing below was inferred,
reconstructed, or guessed. Do not fill in a marked gap without going back to
the human owner for the actual V2 text — do not derive it from this repo's
code, from general RAG practice, or from what "would make sense."

Gaps are marked inline exactly like this:

> **ARCHITECTURE DETAIL MISSING — REQUIRES CONFIRMATION**

---

## 1. System Shape

> **ARCHITECTURE DETAIL MISSING — REQUIRES CONFIRMATION**
> The Audit states the following are frozen in V2 and unchanged by this audit,
> but does not reproduce their specification text: system shape, harness
> structure, guardrail structure (including the "Semantic Grounding
> Consistency Check" design), offline/online separation, hand-written
> orchestration, corpus-scaling tier list (T1–T5 confirmed to exist by name;
> exact tier sizes not confirmed), ablation matrix, reranker A/B/C options
> (the existence of three options is confirmed; which three rerankers is not).

The only shape-level facts this repository can assert with confidence, because
they appear as supporting detail inside the Audit's prose rather than as its
subject:

- The pipeline has a **production/online retrieval-serving path** that talks
  to a **Qdrant** vector store via a defined payload contract, separate from
  an **offline/eval-only** data layer.
- The retrieval-serving payload is *not* the same object as the evaluation
  join table — several fields are deliberately excluded from the former (see
  §4 Data Model Contracts below).
- There is a tiered corpus (T1..T5 naming implied), used to study a
  corpus-size vs. retrieval-quality curve (§6 below).
- A voice pipeline exists upstream of retrieval (implied by the project brief
  and this task's own directory request), but its component boundaries
  (STT provider, query processing steps) are not specified in the Audit.

---

## 2. Retrieval Technology (confirmed by name in the Audit)

| Component | Choice | Source |
|---|---|---|
| Sparse retriever | **BM25** | Audit §4: "bge-m3 and BM25 are used off-the-shelf" |
| Dense embedding model | **bge-m3** | Audit §4, §7 (cross-lingual alignment relied upon) |
| Vector store | **Qdrant** | Audit §5, §7 ("production Qdrant payload", "Qdrant actually searches") |
| Reranker | Off-the-shelf, **not** fine-tuned on this dataset; exactly 3 candidate options exist (A/B/C) | Audit §4, §10.7 — **which 3 rerankers: MISSING** |
| Fusion / hybrid strategy | Hybrid retrieval (sparse + dense) is referenced as an existing V2 concept | **exact fusion method: MISSING** |

**bge-m3 and BM25 are explicitly off-the-shelf, not fine-tuned** on
MSMARCO-XI in the current design. The Audit notes (§4) that *if* fine-tuning
is introduced later as an optimization, it must use train-split labels only —
this is a forward-looking constraint, not a statement that fine-tuning is
planned.

---

## 3. Corpus Construction (frozen, this is the Audit's primary subject)

### 3.1 The flaw that was fixed

V2's original corpus-scaling design made benchmark-query coverage a
**sampling requirement**: build the corpus by prioritizing passages
referenced by the validation query sample, then pad outward. This is
invalid — MS MARCO's ~10 candidate passages per query are already
pre-filtered by a first-stage retriever, so building a corpus from the
benchmark queries' own candidate pools shrinks the "haystack" to mostly
plausible-passage content, systematically inflating measured Recall@K.

### 3.2 The fix (locked)

- The retrieval corpus at each tier is sampled from a **broad pool spanning
  many thousands of queries' worth of passages** — not the ~100–500 queries
  used for benchmark reporting. Example given in the Audit (illustrative,
  not a confirmed tier size): pull passages from on the order of 10,000+
  query rows to build a 250K-passage tier.
- Corpus is drawn from **train + validation splits combined**. Validation
  *passages* belong in the corpus always — excluding them would make
  Recall@K measure passage-reachability rather than retrieval quality.
  Validation *labels* (`is_selected` for validation rows) are touched only
  at final reporting time (§4 below), never during tuning.
- Deduplicate as before (within-language + globally for English).
- Coverage-awareness is a **post-hoc backstop, not a construction method**:
  after building the broad-pool corpus, check which benchmark queries' gold
  passages are missing, and add only the *missing* gold passages. Report
  resulting coverage rate per tier as a metric — this is expected to be
  <100% at small tiers, and that is correct, realistic behavior, not a bug
  to be engineered away.

---

## 4. Evaluation Integrity — Three-Way Split (frozen)

| Pool | Source | Used for |
|---|---|---|
| **Tuning/calibration queries** | Train split | τ (grounding threshold) calibration, reranker A/B/C decision, `T_sentence` threshold, corpus-tier stopping-rule curve-fitting, Hindi-vs-cross-lingual experiment — **every** tunable parameter in the system |
| **Frozen benchmark (reporting) queries** | Validation split | Final reported Recall@K/MRR/nDCG, final latency numbers, final answer-quality numbers — touched exactly once, after every tuning decision above is locked |
| **Corpus (documents to retrieve)** | Train + validation combined, broad pool | Not query-partitioned — a passage's presence in the corpus is independent of its originating query's split |

**Hard rule for DeepSeek:** no code path may compute a metric against the
frozen benchmark set until every tunable parameter in the system is already
fixed from the train-split tuning pool. If a threshold needs re-tuning
later, it goes back through the train pool. The validation reporting set is
never touched a second time to "improve" a number.

---

## 5. Leakage Audit (frozen, item by item)

| Source | Risk | Fix / verdict |
|---|---|---|
| Query text | Could query text appear inside the passage corpus? | Not indexed by construction. Phase 1 sanity check required (substring scan) rather than assumed safe. |
| `query_id` / `source_query_ids` | Metadata that could influence retrieval/generation | **Removed from production Qdrant payload.** Offline passage store / eval-traceability layer only. |
| `Answer` / `Eng_Answer` | Gold reference answers — if they reach the generation context, this directly leaks the answer key | **New eval-only reference table**, structurally separate from both the retrieval corpus and the generation prompt construction path. Same isolation pattern as `is_selected`. This is a more severe leak than `is_selected` if it ever reached generation — it wouldn't just inflate a metric, it would fabricate answer quality. |
| `is_selected` | Gold relevance label | Structurally separate table, never in production payload (unchanged from prior design). |
| Duplicated passages (content-hash dedup) | Could collapse cross-query relevance incorrectly | Not a flaw if the join logic is correct — see §6 below. |
| English/Hindi translated equivalents | Same underlying passage in two languages | Not leakage — intentional, used deliberately by the cross-lingual gold-mapping (§7). |

---

## 6. Canonical Passage ID / Gold-Set Mapping (frozen implementation rule)

Relevance in MS MARCO-style data is a property of a **(query, passage)
pair**, not of a passage alone. Global dedup means one corpus entry can
appear in multiple queries' gold sets, or as a hard negative for another
query — this is correct MS MARCO semantics.

**Mandatory rule:** the evaluation table (`is_selected` joined to
`passage_id`) must be built by running every original per-query passage
through the **same content-hash function** used to dedup the corpus, so
`G(q)` (gold set for query q) is expressed in canonical `passage_id`s that
match what the retriever can actually return.

**Explicitly forbidden:** keeping gold labels indexed by the original
per-row list position (e.g. `passages.is_selected[i]` tied to
`passages.Translated_passages[i]`'s row-local index). This silently breaks
Recall@K the moment two queries share a passage, because the retriever
returns a canonical `passage_id` that won't match a row-local index.

---

## 7. Monolingual vs. Cross-Lingual Retrieval Experiment (frozen)

This was previously an *asserted* default in V2 ("Hindi-only, safer
default"), not a tested decision. The Audit reclassifies it as a mandatory
experiment because English passages are the **original, untranslated**
MS MARCO source text, while Hindi passages are **machine-translated** — it's
plausible bge-m3 retrieves the correct English passage more reliably than
the translation-degraded Hindi version of the same content.

**Config 1 (monolingual):** Hindi query → retrieval restricted to
`lang == hi` partition.

**Config 2 (cross-lingual):** Hindi query → retrieval over `lang ∈ {hi, en}`
combined, no language filter, relying on bge-m3's cross-lingual alignment.
Context assembly must handle possibly-mixed-language context (added
engineering cost specific to Config 2). Whether the generation model can
ground a Hindi answer in English source context is an assumption to verify,
not assume.

**Gold-set correction (mandatory):** gold-set construction for this
experiment cannot use the Hindi row's `is_selected` passages alone as
`G(q)`, or Config 2 is unfairly penalized whenever it correctly retrieves
the English original instead of the (possibly lower-quality) Hindi
translation. Because `English_passages[i]` and `Translated_passages[i]` are
paired by list index in the same row, the **expanded gold set for the
cross-lingual condition** is:

```
G_cross(q) = { passage_id(hi translation of gold passage),
               passage_id(en original of gold passage) }
```

Both count as correct hits under Config 2.

**Metrics:** Recall@K/MRR/nDCG per config (correct gold-set definition per
config), generation-stage groundedness/answer-quality under mixed-language
context, and latency (this experiment doubles as the filtered-search vs.
unfiltered-search Qdrant latency check).

**v1 default (a hypothesis for the tuning pool to confirm or overturn, not a
final answer):** start with Config 1 for guardrail/latency-simplicity
reasons, but the experiment runs regardless.

---

## 8. Known Methodology Risks (to be stated proactively, not discovered by judges)

- **`is_selected` label incompleteness** is a known MS MARCO-lineage
  property — a non-selected passage isn't guaranteed irrelevant, and the
  true relevant set may exceed the ~10-candidate pool. This puts a soft
  ceiling on achievable Recall@K that is not a retrieval-system failure.
- **Train/validation query near-duplication risk**: MS MARCO-family
  datasets have occasionally had near-duplicate queries across splits. If
  present above a negligible rate, either exclude overlapping validation
  queries from the frozen benchmark set or disclose the overlap rate.
  (→ mandatory Phase 1 check, §9.)
- **MT-quality noise in gold `Answer`/`Eng_Answer`** affects
  answer-relevance scoring (previously flagged in V1/V2, restated here as
  adjacent to the leakage/label-quality theme).

---

## 9. Mandatory Phase 1 Sanity Checks (frozen, must run before any threshold is set)

1. **Query-text-not-in-corpus scan** — substring scan verifying query text
   does not leak into passage content.
2. **Train/validation query near-duplicate check** — exact or near-duplicate
   `query`/`Eng_Query` text between train and validation splits; if found
   above a negligible rate, exclude overlapping validation queries from the
   frozen benchmark set or disclose the overlap rate.

Neither check has an implementation yet in this repository (correctly — see
`DEFINITION OF DONE` in `docs/DEEPSEEK_IMPLEMENTATION.md`). Stubs exist in
`ingestion/dataset/` marked `NotImplementedError`.

---

## 10. FINAL FREEZE DECISIONS (verbatim from the Audit, §"FINAL FREEZE DECISIONS")

1. Corpus is built from a broad pool spanning far more queries than the
   benchmark set (thousands of query rows minimum per tier), drawn from
   train + validation combined. Coverage-awareness is a post-hoc backstop
   only.
2. Three query pools, never mixed: train-split for all tuning/calibration;
   validation-split for the final reported benchmark only, touched exactly
   once.
3. `source_query_ids`, `is_selected`, `Answer`, and `Eng_Answer` are all
   structurally absent from the production Qdrant payload and from any
   generation-prompt construction path.
4. Gold-set construction must use the same canonical content-hash
   `passage_id` as corpus deduplication — never row-local list indices.
5. Monolingual vs. cross-lingual retrieval is an experiment with two
   configs, not a preset default, using an expanded gold set for the
   cross-lingual condition.
6. Two Phase 1 sanity checks are mandatory before any threshold is set:
   query-text-not-in-corpus scan, and train/validation query near-duplicate
   check.
7. Everything else frozen in Architecture V2 (system shape, harness,
   guardrail structure, offline/online separation, hand-written
   orchestration, corpus-scaling tier list, ablation matrix, reranker A/B/C
   options, Semantic Grounding Consistency Check design) is unchanged by
   this audit and remains locked as previously specified —
   **but the specification text itself is not available in this repository.
   See gap markers throughout this document.**

---

## 11. Explicit Conflicts Found

None. The Audit is internally consistent and does not contradict itself.
No conflict was found between this document and any other document, because
no second source document (V2 itself) was available to compare against.
If V2's actual text is provided later, **re-run this comparison** — do not
assume consistency was verified against V2; it was only verified against
what the Audit itself claims about V2.
