# apps/evals/tests/test_regression_check.py
from __future__ import annotations

from teaching_knowledge.scripts.aggregate import AggregateResult, DimensionAggregate
from teaching_knowledge.scripts.regression_check import (
    DimensionRegression,
    RegressionReport,
    check_regression,
    format_report,
)


def _agg(
    dim_data: dict[str, tuple[float, tuple[float, float]]],
    composite: float = 2.0,
    composite_ci: tuple[float, float] | None = (1.9, 2.1),
) -> AggregateResult:
    dims = [
        DimensionAggregate(
            name=name,
            mean_process=mean,
            ci_process=ci,
            mean_outcome=mean,
            ci_outcome=ci,
            n=20,
        )
        for name, (mean, ci) in dim_data.items()
    ]
    return AggregateResult(
        dimensions=dims,
        composite_mean=composite,
        composite_ci=composite_ci,
        by_era={},
        by_skill={},
        total_rows=20,
        run_id="test",
    )


def test_flags_regression_when_ci_does_not_overlap() -> None:
    baseline = _agg({"Style": (2.5, (2.3, 2.7))})
    candidate = _agg({"Style": (1.5, (1.3, 1.7))})
    report = check_regression(baseline, candidate)
    style = next(d for d in report.dimensions if d.name == "Style")
    assert style.direction == "regressed"
    assert style.significant is True
    assert report.has_regression is True


def test_flags_improvement_when_ci_does_not_overlap() -> None:
    baseline = _agg({"Style": (1.5, (1.3, 1.7))})
    candidate = _agg({"Style": (2.5, (2.3, 2.7))})
    report = check_regression(baseline, candidate)
    style = next(d for d in report.dimensions if d.name == "Style")
    assert style.direction == "improved"
    assert style.significant is True
    assert report.has_regression is False


def test_null_delta_when_ci_overlaps() -> None:
    baseline = _agg({"Style": (2.0, (1.8, 2.2))})
    candidate = _agg({"Style": (2.1, (1.9, 2.3))})
    report = check_regression(baseline, candidate)
    style = next(d for d in report.dimensions if d.name == "Style")
    assert style.direction == "null"
    assert style.significant is False


def test_composite_delta_reported() -> None:
    baseline = _agg({"S": (2.0, (1.8, 2.2))}, composite=2.0, composite_ci=(1.9, 2.1))
    candidate = _agg({"S": (2.3, (2.1, 2.5))}, composite=2.3, composite_ci=(2.2, 2.4))
    report = check_regression(baseline, candidate)
    assert abs(report.composite_delta - 0.3) < 0.001
    assert report.composite_significant is True


def test_format_report_contains_dimension_names() -> None:
    baseline = _agg({"Alpha": (2.0, (1.8, 2.2)), "Beta": (1.5, (1.3, 1.7))})
    candidate = _agg({"Alpha": (2.1, (1.9, 2.3)), "Beta": (2.5, (2.3, 2.7))})
    text = format_report(check_regression(baseline, candidate))
    assert "Alpha" in text
    assert "Beta" in text
    assert "improved" in text.lower() or "regressed" in text.lower()
