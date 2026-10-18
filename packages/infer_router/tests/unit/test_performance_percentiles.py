"""Latency percentiles must not collapse to the minimum for small samples."""

from openjiuwentools.infer_router.monitoring.performance_stats import PerformanceStats


def test_percentiles_use_nearest_rank() -> None:
    p = PerformanceStats._calculate_percentiles([1.0, 2.0])

    assert p[50] == 1.0
    assert p[90] == 2.0
    assert p[99] == 2.0


def test_higher_percentiles_do_not_understate_three_samples() -> None:
    p = PerformanceStats._calculate_percentiles([1.0, 2.0, 3.0])

    assert p[50] == 2.0
    assert p[70] == 3.0
    assert p[90] == 3.0
