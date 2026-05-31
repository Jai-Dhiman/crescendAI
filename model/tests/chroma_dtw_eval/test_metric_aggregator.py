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


def test_g1_fires_when_amateur_teleport_rate_rises():
    # g1 regresses UP (higher = worse): all amateur chunks have bar_distance > 5.0 => g1=100.0
    # baseline.guards.g1 = 50.0, so 100.0 > 50.0 + 1.0 => "g1" in regressed
    results = [
        ChunkResult(kind="amateur", error_frames=None, cost=0.0, abstain=False, bar_distance_from_forward=6.0),
        ChunkResult(kind="amateur", error_frames=None, cost=0.0, abstain=False, bar_distance_from_forward=7.0),
    ]
    baseline = Baseline(primary=0.0, guards=GuardSet(g1=50.0, g2=0.5, g3=0.0, g4=0.0, g5=0.0))
    m = aggregate(results, baseline=baseline, frame_rate_hz=50.0, tolerance_ms=50.0)
    assert "g1" in m.regressed


def test_g4_fires_when_synth_stitch_accuracy_falls():
    # g4 regresses DOWN (lower = worse): all synth chunks have stitch_error_frames > tol => g4=0.0
    # baseline.guards.g4 = 80.0, so 0.0 < 80.0 - 1.0 => "g4" in regressed
    results = [
        ChunkResult(kind="synthetic_practice", error_frames=None, cost=0.0, abstain=False, stitch_error_frames=10.0),
        ChunkResult(kind="synthetic_practice", error_frames=None, cost=0.0, abstain=False, stitch_error_frames=12.0),
    ]
    baseline = Baseline(primary=0.0, guards=GuardSet(g1=0.0, g2=0.5, g3=0.0, g4=80.0, g5=0.0))
    m = aggregate(results, baseline=baseline, frame_rate_hz=50.0, tolerance_ms=50.0)
    assert "g4" in m.regressed


def test_g5_fires_when_real_practice_self_consistency_drops():
    # g5 regresses UP (higher = worse): all real_practice chunks have bar_distance > 5.0 => g5=100.0
    # baseline.guards.g5 = 50.0, so 100.0 > 50.0 + 1.0 => "g5" in regressed
    results = [
        ChunkResult(kind="real_practice", error_frames=None, cost=0.0, abstain=False, bar_distance_from_forward=8.0),
        ChunkResult(kind="real_practice", error_frames=None, cost=0.0, abstain=False, bar_distance_from_forward=9.0),
    ]
    baseline = Baseline(primary=0.0, guards=GuardSet(g1=0.0, g2=0.5, g3=0.0, g4=0.0, g5=50.0))
    m = aggregate(results, baseline=baseline, frame_rate_hz=50.0, tolerance_ms=50.0)
    assert "g5" in m.regressed
