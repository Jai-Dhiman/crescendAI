"""Tests for E2E analysis computations."""
import math


def test_cohens_d_identical_groups():
    """Cohen's d between identical groups is 0."""
    from pipeline.practice_eval.analyze_e2e import cohens_d
    assert cohens_d([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) == 0.0


def test_cohens_d_distinct_groups():
    """Cohen's d between well-separated groups is large."""
    from pipeline.practice_eval.analyze_e2e import cohens_d
    d = cohens_d([0.1, 0.15, 0.2], [0.8, 0.85, 0.9])
    assert abs(d) > 2.0


def test_cohens_d_small_sample():
    """Cohen's d with < 2 samples returns 0."""
    from pipeline.practice_eval.analyze_e2e import cohens_d
    assert cohens_d([0.5], [0.8, 0.9]) == 0.0


def test_bootstrap_ci_contains_mean():
    """Bootstrap CI should contain the sample mean."""
    from pipeline.practice_eval.analyze_e2e import bootstrap_ci
    data = [0.5, 0.6, 0.7, 0.8, 0.9]
    low, high = bootstrap_ci(data, n_bootstrap=1000, seed=42)
    mean = sum(data) / len(data)
    assert low <= mean <= high


def test_bootstrap_ci_small_sample():
    """Bootstrap with N < 5 returns None."""
    from pipeline.practice_eval.analyze_e2e import bootstrap_ci
    assert bootstrap_ci([0.5, 0.6]) is None


def test_build_confusion_matrix_perfect():
    """Perfect identification gives diagonal-only matrix."""
    from pipeline.practice_eval.analyze_e2e import build_confusion_matrix
    results = [
        {"expected": "bach", "actual": "bach", "correct": True},
        {"expected": "fur_elise", "actual": "fur_elise", "correct": True},
    ]
    matrix = build_confusion_matrix(results)
    assert matrix["bach"]["bach"] == 1
    assert matrix["fur_elise"]["fur_elise"] == 1


def test_build_confusion_matrix_errors():
    """Misidentifications appear off-diagonal."""
    from pipeline.practice_eval.analyze_e2e import build_confusion_matrix
    results = [
        {"expected": "bach", "actual": "fur_elise", "correct": False},
        {"expected": "bach", "actual": "bach", "correct": True},
    ]
    matrix = build_confusion_matrix(results)
    assert matrix["bach"]["bach"] == 1
    assert matrix["bach"]["fur_elise"] == 1
