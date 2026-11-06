"""AgentHints histograms need buckets in their own units."""

from openjiuwentools.infer_router.monitoring.metrics import metrics


def _bucket_bounds(histogram) -> list[float]:
    """Read bucket bounds through the public Prometheus collect() API."""
    return [
        sample.value
        for sample in histogram.collect()[0].samples
        if sample.name.endswith("_bucket")
    ]


def test_priority_histogram_has_integer_priority_buckets() -> None:
    bounds = _bucket_bounds(metrics.priority_distribution)

    assert 0.0 in bounds
    assert tuple(bounds[:11]) == tuple(float(value) for value in range(0, 11))


def test_token_histogram_has_token_sized_buckets() -> None:
    bounds = _bucket_bounds(metrics.estimated_output_tokens_distribution)

    assert 1024.0 in bounds
    assert 32768.0 in bounds
