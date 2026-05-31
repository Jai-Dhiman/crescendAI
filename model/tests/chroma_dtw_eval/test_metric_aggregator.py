from chroma_dtw_eval.metric_aggregator import (
    Baseline, ChunkResult, GuardSet, aggregate,
)


def _gold(err_frames):
    return [ChunkResult(kind="gold", error_frames=e, cost=0.15, abstain=False)
            for e in err_frames]


def test_primary_is_pct_within_50ms_at_50hz():
    # 50ms tolerance at 50 Hz = ±2.5 frames. Use err_frames=2 (pass) and err_frames=5 (fail).
    results = _gold([0, 1, 2, 5, 5])
    baseline = Baseline(primary=0.0, guards=GuardSet(g1=100.0, g2=0.0, g3=100.0, g4=0.0, g5=100.0))
    m = aggregate(results, baseline=baseline, frame_rate_hz=50.0, tolerance_ms=50.0)
    assert m.primary == 60.0
    assert "primary" not in m.regressed


def test_regressed_lists_only_regressing_guards():
    # Primary worse than baseline => regress.
    results = _gold([5, 5, 5, 5, 5])
    baseline = Baseline(primary=80.0, guards=GuardSet(g1=0.0, g2=0.9, g3=0.0, g4=100.0, g5=0.0))
    m = aggregate(results, baseline=baseline, frame_rate_hz=50.0, tolerance_ms=50.0)
    assert "primary" in m.regressed
