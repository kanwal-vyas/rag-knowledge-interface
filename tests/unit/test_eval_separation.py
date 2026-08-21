"""Tests for evaluation dataset separation (ingestion/dataset/eval_split.py)."""
from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from ingestion.dataset.eval_split import (
    extract_eval_dataset,
    load_eval_dataset,
    save_eval_dataset,
)
from ingestion.dataset.loader import MSMARCORow
from ingestion.deduplication.canonical_id import canonical_passage_id

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_row(
    query_id: int = 1,
    en_passages: list[str] | None = None,
    hi_passages: list[str] | None = None,
    is_selected: list[int] | None = None,
    answer: str = "Gold answer",
    eng_answer: str = "English gold answer",
) -> MSMARCORow:
    return MSMARCORow(
        query_id=query_id,
        query_type="DESCRIPTION",
        hindi_query=f"प्रश्न {query_id}",
        english_query=f"Query {query_id}",
        source_lang="hi",
        target_lang="en",
        source_lang_bcp47="hin_Deva",
        target_lang_bcp47="eng_Latn",
        english_passages=(
            en_passages if en_passages is not None
            else [f"English passage {query_id}"]
        ),
        hindi_passages=hi_passages if hi_passages is not None else [f"हिंदी पाठ {query_id}"],
        is_selected=is_selected if is_selected is not None else [1],
        answer=answer,
        eng_answer=eng_answer,
        meta={},
    )


# ---------------------------------------------------------------------------
# Gold label tests
# ---------------------------------------------------------------------------

class TestGoldLabelExtraction:
    def test_gold_labels_use_canonical_ids(self) -> None:
        """Gold passage_ids must be canonical content-hash IDs (Audit §6)."""
        passage_text = "Some passage content"
        expected_en_id = canonical_passage_id(passage_text)
        expected_hi_id = canonical_passage_id("हिंदी पाठ 1")  # default hi_passages

        row = _make_row(en_passages=[passage_text])
        dataset = extract_eval_dataset([row], pool="benchmark")

        assert len(dataset.queries) == 1
        gold = dataset.queries[0].gold_labels
        # Both en and hi passages produce gold labels
        assert len(gold) == 2
        gold_ids = {gl.passage_id for gl in gold}
        assert expected_en_id in gold_ids
        assert expected_hi_id in gold_ids

    def test_both_languages_indexed(self) -> None:
        """Both English and Hindi passage gold labels are extracted."""
        row = _make_row(
            en_passages=["English passage"],
            hi_passages=["हिंदी पाठ"],
            is_selected=[1],
        )
        dataset = extract_eval_dataset([row], pool="benchmark")
        gold = dataset.queries[0].gold_labels
        assert len(gold) == 2  # 1 en + 1 hi

    def test_is_selected_preserved(self) -> None:
        """is_selected labels are correctly mapped."""
        row = _make_row(
            en_passages=["passage 1", "passage 2", "passage 3"],
            hi_passages=["p1", "p2", "p3"],
            is_selected=[1, 0, 0],
        )
        dataset = extract_eval_dataset([row], pool="benchmark")
        # English labels
        en_labels = [gl for gl in dataset.queries[0].gold_labels
                     if gl.passage_id == canonical_passage_id("passage 1")]
        assert len(en_labels) == 1
        assert en_labels[0].is_selected is True

    def test_answers_stored_eval_only(self) -> None:
        """Answer/Eng_Answer are stored in eval dataset, not production."""
        row = _make_row(answer="Hindi answer", eng_answer="English answer")
        dataset = extract_eval_dataset([row], pool="benchmark")
        q = dataset.queries[0]
        assert q.answer == "Hindi answer"
        assert q.eng_answer == "English answer"

    def test_gold_for_query(self) -> None:
        """gold_for_query returns canonical IDs of selected passages."""
        passage = "Selected passage"
        row = _make_row(en_passages=[passage], hi_passages=[], is_selected=[1])
        dataset = extract_eval_dataset([row], pool="benchmark")
        gold_ids = dataset.gold_for_query(query_id=1)
        assert gold_ids == [canonical_passage_id(passage)]


# ---------------------------------------------------------------------------
# Save/load roundtrip
# ---------------------------------------------------------------------------

class TestSaveLoadRoundtrip:
    def test_roundtrip(self) -> None:
        """Save and load an eval dataset, verify data integrity."""
        row = _make_row(
            en_passages=["passage a", "passage b"],
            hi_passages=["पाठ a", "पाठ b"],
            is_selected=[1, 0],
            answer="उत्तर",
            eng_answer="Answer",
        )
        dataset = extract_eval_dataset([row], pool="tuning")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            save_eval_dataset(dataset, output_dir)
            loaded = load_eval_dataset(output_dir, pool="tuning")

        assert len(loaded.queries) == 1
        q = loaded.queries[0]
        assert q.query_id == 1
        assert q.answer == "उत्तर"
        assert q.eng_answer == "Answer"
        # Gold labels preserved: 2 en + 2 hi passages
        assert len(q.gold_labels) == 4
        selected = [gl for gl in q.gold_labels if gl.is_selected]
        assert len(selected) == 2  # passage a (en) + पाठ a (hi) both is_selected=1

    def test_files_created(self) -> None:
        """Save creates the expected files."""
        row = _make_row()
        dataset = extract_eval_dataset([row], pool="benchmark")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            save_eval_dataset(dataset, output_dir)
            assert (output_dir / "benchmark_queries.json").exists()
            assert (output_dir / "benchmark_gold_labels.csv").exists()
            assert (output_dir / "benchmark_answers.csv").exists()

    def test_gold_labels_csv_format(self) -> None:
        """Gold labels CSV has correct columns."""
        row = _make_row(en_passages=["p1"], hi_passages=[], is_selected=[1])
        dataset = extract_eval_dataset([row], pool="test")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            save_eval_dataset(dataset, output_dir)
            labels_file = output_dir / "test_gold_labels.csv"
            with open(labels_file, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                assert reader.fieldnames == ["query_id", "passage_id", "is_selected"]


# ---------------------------------------------------------------------------
# Structural isolation
# ---------------------------------------------------------------------------

class TestStructuralIsolation:
    def test_eval_dataset_not_importable_from_production(self) -> None:
        """
        Audit §5 / DEEPSEEK_IMPLEMENTATION.md: production packages
        must not import from evaluation.*. This is enforced by
        test_model_isolation.py's AST scan. Here we verify the
        eval module's exports contain evaluation-only fields.
        """
        from evaluation.models import AnswerReference, GoldRelevanceLabel
        # These models contain the fields that MUST NOT appear in production
        assert "is_selected" in GoldRelevanceLabel.model_fields
        assert (
            "answer" in AnswerReference.model_fields
            or "query_id" in AnswerReference.model_fields
        )

    def test_production_models_reject_evaluation_fields(self) -> None:
        """Production models with extra='forbid' reject smuggled eval fields."""

        from app.models.retrieval import Passage, Query

        # Passage should reject is_selected
        try:
            Passage(
                passage_id="test",
                text="test",
                lang="hi",
                is_selected=True,  # type: ignore
            )
            raise AssertionError("Should have raised ValidationError")
        except Exception as e:
            assert "is_selected" in str(e) or "extra" in str(e).lower()

        # Query should reject query_id
        try:
            Query(
                query_text="test",
                lang="hi",
                query_id=123,  # type: ignore
            )
            raise AssertionError("Should have raised ValidationError")
        except Exception as e:
            assert "query_id" in str(e) or "extra" in str(e).lower()
