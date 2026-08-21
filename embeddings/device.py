"""
Device detection and configuration for embedding inference.

Supports:
- CPU (always available)
- CUDA (NVIDIA GPU)
- MPS (Apple Silicon GPU)

The detected device is logged explicitly so benchmarks report
whether GPU was available.
"""
from __future__ import annotations

import os

import structlog

logger = structlog.get_logger(__name__)


def detect_device(requested: str | None = None) -> str:
    """
    Determine the compute device for embedding inference.

    Priority:
    1. Explicit `requested` device (if valid)
    2. HF_EMBEDDING_DEVICE environment variable
    3. Auto-detect: CUDA > MPS > CPU

    Returns a string: "cuda", "mps", or "cpu".
    """
    import torch

    # 1. Explicit request
    if requested and requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            logger.warning("cuda_requested_but_unavailable", falling_back_to="cpu")
            return "cpu"
        if requested == "mps" and not torch.backends.mps.is_available():
            logger.warning("mps_requested_but_unavailable", falling_back_to="cpu")
            return "cpu"
        return requested

    # 2. Environment variable
    env_device = os.environ.get("HF_EMBEDDING_DEVICE", "").strip().lower()
    if env_device in ("cuda", "mps", "cpu"):
        return detect_device(env_device)  # recurse with explicit value for validation

    # 3. Auto-detect
    if torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name(0)
        except (AssertionError, RuntimeError):
            device_name = "unknown"
        logger.info("gpu_detected", device="cuda", gpu_name=device_name)
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("gpu_detected", device="mps")
        return "mps"

    logger.info("cpu_only", device="cpu")
    return "cpu"


def report_device() -> dict:
    """
    Return a detailed device report for benchmark logging.

    Includes: device, GPU name, CUDA version, memory, torch version.
    """
    import torch

    device = detect_device()
    report: dict = {
        "device": device,
        "gpu_available": device in ("cuda", "mps"),
        "torch_version": torch.__version__,
    }

    if device == "cuda":
        report["gpu_name"] = torch.cuda.get_device_name(0)
        report["cuda_version"] = torch.version.cuda or "unknown"
        report["gpu_memory_total_gb"] = round(
            torch.cuda.get_device_properties(0).total_mem / (1024**3), 1
        )
        # Don't report allocated memory on cold start
        report["gpu_memory_allocated_gb"] = round(
            torch.cuda.memory_allocated(0) / (1024**3), 2
        )
    elif device == "mps":
        report["gpu_name"] = "Apple Silicon (MPS)"

    return report


def log_device_report(prefix: str = "") -> dict:
    """Log and return the device report."""
    report = report_device()
    tag = f"{prefix}_" if prefix else ""
    logger.info(
        f"{tag}device_report",
        **{k: v for k, v in report.items()},
    )
    return report
