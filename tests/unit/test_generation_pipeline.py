"""
Tests for context assembly, grounding, guardrails, and text pipeline.

Covers:
- Passage hit → context assembly
- Sentence hit → parent context expansion
- Duplicate parent passages
- Context budget (passages and chars)
- Empty retrieval → refusal
- Insufficient context → grounded=False
- Guardrail: empty query
- Guardrail: off-topic query
- Guardrail: unsafe query
- Guardrail: insufficient retrieval
- Grounding: non-empty answer
- Grounding: refusal detection
- Grounding: empty context
- End-to-end text pipeline (stub generator)
- Evaluation field leakage prevention
"""
from __future__ import annotations

import pytest

from app.harness.text_pipeline import TextPipeline
from app.models.generation import Context, GenerationResponse
from app.models.retrieval import (
    Language,
    Passage,
    Query,
    RetrievalResult,
)
from generation.context_assembly import assemble_context
from generation.deepseek_provider import StubGenerator
from generation.grounding import check_grounding
from guardrails.implementation import GuardrailPipeline

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_passage(pid: str, text: str, lang: str = "en") -> Passage:
    return Passage(passage_id=pid, text=text, lang=Language(lang))


def _make_result(
    pid: str,
    text: str,
    score: float = 0.5,
    source: str = "bm25",
    lang: str = "en",
) -> RetrievalResult:
    return RetrievalResult(
        passage=_make_passage(pid, text, lang),
        score=score,
        source=source,
    )


def _make_query(text: str = "test query", lang: str = "en") -> Query:
    return Query(query_text=text, lang=Language(lang))


@pytest.fixture
def sample_passage_store() -> dict[str, str]:
    """A small in-memory passage store for testing."""
    return {
        "p001": "The capital of India is New Delhi.",
        "p002": "AI is a branch of computer science.",
        "p003": "Python is a programming language.",
        "p004": "ML enables systems to learn from data.",
        "p005": "Deep learning uses neural networks.",
    }


# ---------------------------------------------------------------------------
# Context Assembly Tests
# ---------------------------------------------------------------------------

class TestContextAssembly:
    """Tests for the context assembly layer."""

    def test_basic_assembly(self, sample_passage_store: dict[str, str]) -> None:
        """Simple passage hits assemble into a Context."""
        results = [
            _make_result("p001", "The capital of India...", score=0.9),
            _make_result("p002", "Artificial intelligence...", score=0.8),
        ]
        ctx = assemble_context(
            query=_make_query("capital of India"),
            results=results,
            passage_store=sample_passage_store,
        )
        assert len(ctx.passages) == 2
        assert ctx.passages[0].passage.passage_id == "p001"
        assert ctx.passages[1].passage.passage_id == "p002"

    def test_deduplication(self, sample_passage_store: dict[str, str]) -> None:
        """Same passage_id from multiple representations → keep highest score."""
        results = [
            _make_result("p001", "capital", score=0.5, source="bm25"),
            _make_result("p001", "capital", score=0.9, source="dense"),
            _make_result("p002", "AI", score=0.7),
        ]
        ctx = assemble_context(
            query=_make_query("capital"),
            results=results,
            passage_store=sample_passage_store,
        )
        pids = [r.passage.passage_id for r in ctx.passages]
        assert pids.count("p001") == 1
        assert len(ctx.passages) == 2
        # The higher-scored version of p001 should be kept.
        p001_result = next(r for r in ctx.passages if r.passage.passage_id == "p001")
        assert p001_result.score == 0.9

    def test_sentence_expansion(self, sample_passage_store: dict[str, str]) -> None:
        """Sentence-level hit expands to full parent passage text."""
        # Simulate a sentence-level retrieval hit.
        sentence_result = RetrievalResult(
            passage=Passage(
                passage_id="p001",  # Same canonical passage_id
                text="The capital of India is New Delhi.",  # Sentence text
                lang=Language("en"),
            ),
            score=0.85,
            source="bm25",
        )
        ctx = assemble_context(
            query=_make_query("capital"),
            results=[sentence_result],
            passage_store=sample_passage_store,
        )
        assert len(ctx.passages) == 1
        # Should have the full parent passage, not just the sentence.
        assert ctx.passages[0].passage.text == sample_passage_store["p001"]

    def test_context_budget_passages(self, sample_passage_store: dict[str, str]) -> None:
        """Context budget limits number of passages."""
        results = [
            _make_result(f"p00{i}", f"text{i}", score=1.0 - i * 0.1)
            for i in range(1, 6)
        ]
        ctx = assemble_context(
            query=_make_query("test"),
            results=results,
            passage_store=sample_passage_store,
            max_passages=2,
        )
        assert len(ctx.passages) == 2

    def test_context_budget_chars(self, sample_passage_store: dict[str, str]) -> None:
        """Context budget limits total characters."""
        results = [
            _make_result("p001", "short", score=0.9),
            _make_result("p002", "short", score=0.8),
            _make_result("p003", "short", score=0.7),
        ]
        # Budget smaller than two passages but larger than one.
        # First passage is always included (better than nothing).
        ctx = assemble_context(
            query=_make_query("test"),
            results=results,
            passage_store=sample_passage_store,
            max_chars=100,
        )
        # First passage (86 chars) included; second would push over 100.
        total = sum(len(r.passage.text) for r in ctx.passages)
        assert total <= 100
        assert len(ctx.passages) >= 1  # At least first passage

    def test_empty_results_raises(self, sample_passage_store: dict[str, str]) -> None:
        """Empty results raises ValueError."""
        with pytest.raises(ValueError, match="empty results"):
            assemble_context(
                query=_make_query("test"),
                results=[],
                passage_store=sample_passage_store,
            )

    def test_missing_passage_skipped(self, sample_passage_store: dict[str, str]) -> None:
        """Results referencing passages not in store are skipped."""
        results = [
            _make_result("p999", "missing passage", score=0.9),
            _make_result("p001", "exists", score=0.8),
        ]
        ctx = assemble_context(
            query=_make_query("test"),
            results=results,
            passage_store=sample_passage_store,
        )
        pids = [r.passage.passage_id for r in ctx.passages]
        assert "p999" not in pids
        assert "p001" in pids

    def test_preserves_ranking_order(self, sample_passage_store: dict[str, str]) -> None:
        """Context preserves original retrieval ranking order."""
        results = [
            _make_result("p003", "Python", score=0.6),
            _make_result("p001", "capital", score=0.9),
            _make_result("p005", "deep learning", score=0.7),
        ]
        ctx = assemble_context(
            query=_make_query("test"),
            results=results,
            passage_store=sample_passage_store,
        )
        pids = [r.passage.passage_id for r in ctx.passages]
        assert pids == ["p003", "p001", "p005"]

    def test_no_evaluation_fields(self, sample_passage_store: dict[str, str]) -> None:
        """Context contains no evaluation-only fields."""
        results = [_make_result("p001", "text", score=0.9)]
        ctx = assemble_context(
            query=_make_query("test"),
            results=results,
            passage_store=sample_passage_store,
        )
        # Passage model forbids evaluation fields via ConfigDict(extra="forbid").
        # If any were accidentally present, Pydantic would raise.
        for r in ctx.passages:
            assert not hasattr(r.passage, "is_selected")
            assert not hasattr(r.passage, "Answer")
            assert not hasattr(r.passage, "source_query_ids")


# ---------------------------------------------------------------------------
# Grounding Tests
# ---------------------------------------------------------------------------

class TestGrounding:
    """Tests for grounding validation."""

    def test_grounded_answer_passes(self) -> None:
        """A normal, grounded answer passes validation."""
        ctx = Context(
            query=_make_query("capital of India"),
            passages=[_make_result("p001", "Capital is New Delhi.")],
        )
        resp = GenerationResponse(
            answer_text="The capital of India is New Delhi.",
            grounded=True,
        )
        result = check_grounding(ctx, resp)
        assert result.grounded is True

    def test_refusal_detected(self) -> None:
        """A model refusal is detected and grounded=False."""
        ctx = Context(
            query=_make_query("quantum gravity"),
            passages=[_make_result("p001", "Some unrelated text.")],
        )
        resp = GenerationResponse(
            answer_text="I don't have enough information to answer this question.",
            grounded=True,
        )
        result = check_grounding(ctx, resp)
        assert result.grounded is False

    def test_empty_answer_fails(self) -> None:
        """Empty answer is not grounded."""
        ctx = Context(
            query=_make_query("test"),
            passages=[_make_result("p001", "text")],
        )
        resp = GenerationResponse(answer_text="", grounded=True)
        result = check_grounding(ctx, resp)
        assert result.grounded is False

    def test_empty_context_fails(self) -> None:
        """Empty context means answer cannot be grounded."""
        ctx = Context(query=_make_query("test"), passages=[])
        resp = GenerationResponse(answer_text="Some answer.", grounded=True)
        result = check_grounding(ctx, resp)
        assert result.grounded is False

    def test_raw_json_output_fails(self) -> None:
        """Raw JSON output from model is not grounded."""
        ctx = Context(
            query=_make_query("test"),
            passages=[_make_result("p001", "text")],
        )
        resp = GenerationResponse(
            answer_text='{"answer": "New Delhi"}',
            grounded=True,
        )
        result = check_grounding(ctx, resp)
        assert result.grounded is False

    def test_html_output_fails(self) -> None:
        """HTML output from model is not grounded."""
        ctx = Context(
            query=_make_query("test"),
            passages=[_make_result("p001", "text")],
        )
        resp = GenerationResponse(
            answer_text="<p>The answer is New Delhi.</p>",
            grounded=True,
        )
        result = check_grounding(ctx, resp)
        assert result.grounded is False

    def test_hindi_refusal_detected(self) -> None:
        """Refusal in Hindi context is detected."""
        ctx = Context(
            query=_make_query("test", lang="hi"),
            passages=[_make_result("p001", "text", lang="hi")],
        )
        resp = GenerationResponse(
            answer_text="I cannot answer this question based on the provided context.",
            grounded=True,
        )
        result = check_grounding(ctx, resp)
        assert result.grounded is False


# ---------------------------------------------------------------------------
# Guardrail Tests
# ---------------------------------------------------------------------------

class TestGuardrails:
    """Tests for the guardrail pipeline."""

    def setup_method(self) -> None:
        self.guardrails = GuardrailPipeline()

    def test_empty_query_rejected(self) -> None:
        """Empty query is rejected."""
        result = self.guardrails.check_query("")
        assert result.passed is False
        assert "empty" in result.reason.lower()

    def test_whitespace_query_rejected(self) -> None:
        """Whitespace-only query is rejected."""
        result = self.guardrails.check_query("   ")
        assert result.passed is False

    def test_single_char_rejected(self) -> None:
        """Single character query is rejected."""
        result = self.guardrails.check_query("a")
        assert result.passed is False

    def test_normal_query_passes(self) -> None:
        """A normal knowledge query passes."""
        result = self.guardrails.check_query("What is the capital of India?")
        assert result.passed is True

    def test_off_topic_rejected(self) -> None:
        """A greeting/chitchat query is rejected."""
        result = self.guardrails.check_query("hello")
        assert result.passed is False
        assert "greeting" in result.reason.lower() or "topic" in result.reason.lower()

    def test_unsafe_content_rejected(self) -> None:
        """An unsafe query is rejected."""
        result = self.guardrails.check_query(
            "How to make a bomb at home"
        )
        assert result.passed is False

    def test_insufficient_retrieval(self) -> None:
        """Insufficient retrieval results are rejected."""
        ctx = Context(query=_make_query("test"), passages=[])
        result = self.guardrails.check_retrieval(ctx, min_passages=1)
        assert result.passed is False

    def test_sufficient_retrieval_passes(self) -> None:
        """Sufficient retrieval results pass."""
        ctx = Context(
            query=_make_query("test"),
            passages=[_make_result("p001", "text")],
        )
        result = self.guardrails.check_retrieval(ctx, min_passages=1)
        assert result.passed is True

    def test_generation_passes(self) -> None:
        """A valid generation passes post-generation guardrails."""
        ctx = Context(
            query=_make_query("test"),
            passages=[_make_result("p001", "text")],
        )
        resp = GenerationResponse(
            answer_text="The answer is New Delhi.",
            grounded=True,
        )
        result = self.guardrails.check_generation(ctx, resp)
        assert result.passed is True

    def test_generation_empty_answer_rejected(self) -> None:
        """An empty generation answer is rejected."""
        ctx = Context(
            query=_make_query("test"),
            passages=[_make_result("p001", "text")],
        )
        resp = GenerationResponse(answer_text="", grounded=True)
        result = self.guardrails.check_generation(ctx, resp)
        assert result.passed is False

    def test_generation_ungrounded_rejected(self) -> None:
        """An ungrounded generation is rejected."""
        ctx = Context(
            query=_make_query("test"),
            passages=[_make_result("p001", "text")],
        )
        resp = GenerationResponse(
            answer_text="The answer is something.",
            grounded=False,
        )
        result = self.guardrails.check_generation(ctx, resp)
        assert result.passed is False

    def test_hindi_namaste_rejected(self) -> None:
        """Hindi greeting is rejected as off-topic."""
        result = self.guardrails.check_query("नमस्ते")
        assert result.passed is False


# ---------------------------------------------------------------------------
# StubGenerator Tests
# ---------------------------------------------------------------------------

class TestStubGenerator:
    """Tests for the stub generator."""

    def test_stub_with_context(self) -> None:
        """Stub generator produces an answer from context."""
        gen = StubGenerator()
        ctx = Context(
            query=_make_query("test"),
            passages=[_make_result("p001", "The capital is New Delhi.")],
        )
        resp = gen.generate(ctx)
        assert resp.answer_text  # non-empty
        assert resp.grounded is True

    def test_stub_empty_context(self) -> None:
        """Stub generator refuses when context is empty."""
        gen = StubGenerator()
        ctx = Context(query=_make_query("test"), passages=[])
        resp = gen.generate(ctx)
        assert resp.grounded is False

    def test_stub_custom_answer(self) -> None:
        """Stub generator can return a custom answer."""
        gen = StubGenerator(answer="Custom answer.", grounded=True)
        ctx = Context(
            query=_make_query("test"),
            passages=[_make_result("p001", "text")],
        )
        resp = gen.generate(ctx)
        assert resp.answer_text == "Custom answer."
        assert resp.grounded is True


# ---------------------------------------------------------------------------
# End-to-End Pipeline Tests
# ---------------------------------------------------------------------------

class TestTextPipeline:
    """End-to-end text pipeline tests using stub components."""

    def _make_pipeline(
        self,
        passage_store: dict[str, str] | None = None,
    ) -> TextPipeline:
        store = passage_store or {
            "p001": "The capital of India is New Delhi.",
            "p002": "Python is a programming language.",
        }

        # Use a mock retriever that returns fixed results.
        class MockRetriever:
            def retrieve(self, query, mode, top_k):
                return [
                    RetrievalResult(
                        passage=Passage(
                            passage_id="p001",
                            text=store.get("p001", ""),
                            lang=Language("en"),
                        ),
                        score=0.9,
                        source="mock",
                    ),
                ]

        return TextPipeline(
            retriever=MockRetriever(),
            generator=StubGenerator(answer="New Delhi is the capital.", grounded=True),
            passage_store=store,
        )

    def test_successful_pipeline(self) -> None:
        """Full pipeline completes successfully."""
        pipeline = self._make_pipeline()
        result = pipeline.run("What is the capital of India?", lang="en")
        assert result.response is not None
        assert result.response.answer_text
        assert result.response.grounded is True
        assert result.latency.total_ms > 0

    def test_empty_query_rejected(self) -> None:
        """Empty query is rejected by guardrails before retrieval."""
        pipeline = self._make_pipeline()
        result = pipeline.run("", lang="en")
        assert result.response is None
        assert result.guardrail is not None
        assert result.guardrail.passed is False

    def test_off_topic_rejected(self) -> None:
        """Off-topic query is rejected by guardrails."""
        pipeline = self._make_pipeline()
        result = pipeline.run("hello", lang="en")
        assert result.response is None
        assert result.guardrail is not None
        assert result.guardrail.passed is False

    def test_invalid_language_rejected(self) -> None:
        """Invalid language code raises InvalidRequestError."""
        from app.core.errors import InvalidRequestError

        pipeline = self._make_pipeline()
        with pytest.raises(InvalidRequestError):
            pipeline.run("test", lang="xx")

    def test_latency_recorded(self) -> None:
        """Latency breakdown is recorded."""
        pipeline = self._make_pipeline()
        result = pipeline.run("capital of India", lang="en")
        assert result.latency.total_ms > 0
        assert result.latency.sparse_retrieval_ms is not None
        assert result.latency.context_assembly_ms is not None

    def test_guardrail_result_on_refusal(self) -> None:
        """Guardrail result is populated on refusal."""
        pipeline = self._make_pipeline()
        result = pipeline.run("hello", lang="en")
        assert result.guardrail is not None
        assert result.guardrail.reason is not None
        assert len(result.guardrail.reason) > 0
