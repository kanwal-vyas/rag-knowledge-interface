from __future__ import annotations

import time

from app.models.latency import LatencyBreakdown


def test_latency_breakdown_requires_total_and_is_estimated_flag() -> None:
    lb = LatencyBreakdown(total_ms=123.4, is_estimated=False)
    assert lb.total_ms == 123.4
    assert lb.is_estimated is False
    assert lb.stt_ms is None  # unset stages are None, not 0


def test_monotonic_timing_pattern() -> None:
    """
    Sanity check that the monotonic-clock pattern the architecture
    requires actually behaves as expected in this environment — not a
    test of real pipeline latency, just of the timing primitive.
    """
    start = time.perf_counter()
    time.sleep(0.001)
    elapsed_ms = (time.perf_counter() - start) * 1000
    lb = LatencyBreakdown(total_ms=elapsed_ms, is_estimated=False)
    assert lb.total_ms > 0
