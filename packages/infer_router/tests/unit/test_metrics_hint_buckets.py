"""AgentHints histograms need buckets in their own units."""

from openjiuwentools.infer_router.monitoring.metrics import metrics


def test_priority_histogram_has_integer_priority_buckets() -> None:
    bounds = metrics.priority_distribution._upper_bounds

    assert 0 in bounds
    assert tuple(bounds[:11]) == tuple(float(value) for value in range(0, 11))


def test_token_histogram_has_token_sized_buckets() -> None:
    bounds = metrics.estimated_output_tokens_distribution._upper_bounds

    assert 1024 in bounds
    assert 32768 in bounds
