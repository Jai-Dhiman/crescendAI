import numpy as np

from masterclass_experiments.evaluation import leave_one_video_out_cv


def test_lovo_cv_returns_aggregate_metrics():
    rng = np.random.default_rng(42)
    n = 30
    X = rng.standard_normal((n, 5))
    y = rng.integers(0, 2, size=n)
    video_ids = np.array(["v1"] * 10 + ["v2"] * 10 + ["v3"] * 10)

    results = leave_one_video_out_cv(X, y, video_ids)

    assert "auc" in results
    assert "accuracy" in results
    assert "precision" in results
    assert "recall" in results
    assert "per_segment" in results
    assert isinstance(results["auc"], float)
    assert len(results["per_segment"]) == n


def test_lovo_cv_per_segment_has_required_fields():
    rng = np.random.default_rng(42)
    n = 20
    X = rng.standard_normal((n, 5))
    y = rng.integers(0, 2, size=n)
    video_ids = np.array(["v1"] * 10 + ["v2"] * 10)

    results = leave_one_video_out_cv(X, y, video_ids)

    seg = results["per_segment"][0]
    assert "video_id" in seg
    assert "y_true" in seg
    assert "y_pred_proba" in seg
