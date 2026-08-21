"""Tests for device detection utility."""
from __future__ import annotations

from unittest.mock import patch


class TestDetectDevice:
    """Tests for the detect_device function."""

    def test_cpu_only_environment(self) -> None:
        """When no GPU is available, should return 'cpu'."""
        from embeddings.device import detect_device

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
            patch.dict("os.environ", {}, clear=True),
        ):
            result = detect_device(None)
            assert result == "cpu"

    def test_cuda_available(self) -> None:
        """When CUDA is available, should return 'cuda'."""
        from embeddings.device import detect_device

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.backends.mps.is_available", return_value=False),
            patch.dict("os.environ", {}, clear=True),
        ):
            result = detect_device(None)
            assert result == "cuda"

    def test_mps_available_no_cuda(self) -> None:
        """When MPS is available but not CUDA, should return 'mps'."""
        from embeddings.device import detect_device

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=True),
            patch.dict("os.environ", {}, clear=True),
        ):
            result = detect_device(None)
            assert result == "mps"

    def test_explicit_cpu(self) -> None:
        """Explicitly requesting 'cpu' should return 'cpu'."""
        from embeddings.device import detect_device

        result = detect_device("cpu")
        assert result == "cpu"

    def test_explicit_cuda_when_unavailable(self) -> None:
        """Requesting CUDA when unavailable should fall back to CPU."""
        from embeddings.device import detect_device

        with patch("torch.cuda.is_available", return_value=False):
            result = detect_device("cuda")
            assert result == "cpu"

    def test_explicit_mps_when_unavailable(self) -> None:
        """Requesting MPS when unavailable should fall back to CPU."""
        from embeddings.device import detect_device

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            result = detect_device("mps")
            assert result == "cpu"

    def test_auto_keyword(self) -> None:
        """'auto' should trigger auto-detection."""
        from embeddings.device import detect_device

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
            patch.dict("os.environ", {}, clear=True),
        ):
            result = detect_device("auto")
            assert result == "cpu"

    def test_env_variable_override(self) -> None:
        """HF_EMBEDDING_DEVICE env var should be respected."""
        from embeddings.device import detect_device

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
            patch.dict("os.environ", {"HF_EMBEDDING_DEVICE": "cpu"}),
        ):
            result = detect_device(None)
            assert result == "cpu"

    def test_return_type_is_string(self) -> None:
        """Result should always be a string."""
        from embeddings.device import detect_device

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            result = detect_device(None)
            assert isinstance(result, str)
            assert result in ("cpu", "cuda", "mps")


class TestReportDevice:
    """Tests for the report_device function."""

    def test_report_contains_required_keys(self) -> None:
        """Report should always contain device, gpu_available, torch_version."""
        from embeddings.device import report_device

        report = report_device()
        assert "device" in report
        assert "gpu_available" in report
        assert "torch_version" in report
        assert isinstance(report["gpu_available"], bool)

    def test_report_cpu_device(self) -> None:
        """CPU report should not have GPU-specific keys."""
        from embeddings.device import report_device

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            report = report_device()
            assert report["device"] == "cpu"
            assert report["gpu_available"] is False
            # GPU-specific keys should not be present
            assert "gpu_name" not in report
            assert "cuda_version" not in report

    def test_report_is_serializable(self) -> None:
        """Report should be JSON-serializable."""
        import json

        from embeddings.device import report_device

        report = report_device()
        # Should not raise
        serialized = json.dumps(report)
        assert isinstance(serialized, str)


class TestEvaluateOnlyFieldsNotInDeviceReport:
    """Verify device report does not leak evaluation-only fields."""

    def test_no_evaluation_fields_in_report(self) -> None:
        """Device report must never contain evaluation-only fields."""
        from app.models.retrieval import FORBIDDEN_EVALUATION_FIELDS
        from embeddings.device import report_device

        report = report_device()
        leaked = set(report.keys()) & FORBIDDEN_EVALUATION_FIELDS
        assert not leaked, f"Device report contains forbidden fields: {leaked}"
