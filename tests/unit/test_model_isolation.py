from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app.models.generation import Context, GenerationRequest, GenerationResponse
from app.models.retrieval import FORBIDDEN_EVALUATION_FIELDS, Passage, Query, RetrievalResult

PRODUCTION_MODEL_CLASSES = [
    Passage,
    Query,
    RetrievalResult,
    Context,
    GenerationRequest,
    GenerationResponse,
]

PRODUCTION_PACKAGE_ROOTS = ["app", "retrieval", "embeddings", "generation", "guardrails"]
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("model_cls", PRODUCTION_MODEL_CLASSES)
def test_no_forbidden_evaluation_fields_on_production_models(model_cls: type) -> None:
    field_names = set(model_cls.model_fields.keys())
    leaked = field_names & FORBIDDEN_EVALUATION_FIELDS
    assert not leaked, (
        f"{model_cls.__name__} has forbidden evaluation field(s) {leaked}. "
        "is_selected / Answer / Eng_Answer / source_query_ids / query_id must never "
        "appear on production models — see Audit §5 and app/models/retrieval.py."
    )


@pytest.mark.parametrize("model_cls", PRODUCTION_MODEL_CLASSES)
def test_production_models_forbid_extra_fields(model_cls: type) -> None:
    """
    extra='forbid' is what actually prevents a caller from smuggling an
    evaluation field in via **kwargs/dict-unpacking at runtime, even if
    the field was never declared. This test guards against someone
    quietly relaxing that config.
    """
    assert model_cls.model_config.get("extra") == "forbid", (
        f"{model_cls.__name__} must set model_config = ConfigDict(extra='forbid') "
        "to structurally block evaluation-field smuggling."
    )


def _imports_evaluation_module(py_path: Path) -> bool:
    tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "evaluation" or alias.name.startswith("evaluation."):
                    return True
        if isinstance(node, ast.ImportFrom) and node.module and (
            node.module == "evaluation" or node.module.startswith("evaluation.")
        ):
            return True
    return False


def test_no_production_module_imports_evaluation_package() -> None:
    """
    Structural isolation rule from evaluation/models.py: nothing in the
    production packages may import evaluation.* at all. Static AST scan
    rather than a runtime check, so it also catches an import that's never
    exercised by other tests.
    """
    offenders = []
    for root_name in PRODUCTION_PACKAGE_ROOTS:
        root = REPO_ROOT / root_name
        if not root.exists():
            continue
        for py_file in root.rglob("*.py"):
            if _imports_evaluation_module(py_file):
                offenders.append(str(py_file.relative_to(REPO_ROOT)))

    assert not offenders, (
        f"Production modules importing evaluation.*: {offenders}. "
        "This is the exact leakage pattern the architecture audit was written to prevent."
    )
