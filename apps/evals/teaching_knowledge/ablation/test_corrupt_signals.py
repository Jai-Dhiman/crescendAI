import pytest
from teaching_knowledge.ablation.corrupt_signals import corrupt

CORPUS = [
    [{"dimension": "dynamics", "score": 0.8, "deviation_from_mean": 0.25, "direction": "above_average"}],
    [{"dimension": "timing", "score": 0.3, "deviation_from_mean": -0.18, "direction": "below_average"}],
    [{"dimension": "pedaling", "score": 0.6, "deviation_from_mean": 0.14, "direction": "above_average"}],
]


def test_shuffle_deterministic_same_seed():
    src = CORPUS[0]
    a = corrupt(src, mode="shuffle", seed=42, all_top_moments=CORPUS)
    b = corrupt(src, mode="shuffle", seed=42, all_top_moments=CORPUS)
    assert a == b


def test_shuffle_returns_other_session_signals():
    src = CORPUS[0]
    out = corrupt(src, mode="shuffle", seed=42, all_top_moments=CORPUS)
    assert out in [CORPUS[1], CORPUS[2]]


def test_shuffle_raises_on_singleton_corpus():
    with pytest.raises(ValueError, match="cannot shuffle"):
        corrupt(CORPUS[0], mode="shuffle", seed=42, all_top_moments=[CORPUS[0]])
