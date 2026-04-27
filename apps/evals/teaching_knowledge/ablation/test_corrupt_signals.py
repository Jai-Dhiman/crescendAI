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


def test_marginal_deterministic_same_seed():
    src = CORPUS[0]
    a = corrupt(src, mode="marginal", seed=42, all_top_moments=CORPUS)
    b = corrupt(src, mode="marginal", seed=42, all_top_moments=CORPUS)
    assert a == b


def test_marginal_preserves_dim_names():
    src = CORPUS[0]
    out = corrupt(src, mode="marginal", seed=42, all_top_moments=CORPUS)
    assert {m["dimension"] for m in out} == {m["dimension"] for m in src}


def test_marginal_scores_come_from_corpus():
    src = CORPUS[0]
    out = corrupt(src, mode="marginal", seed=42, all_top_moments=CORPUS)
    for moment in out:
        observed = [m["score"] for tm in CORPUS for m in tm if m["dimension"] == moment["dimension"]]
        assert moment["score"] in observed


def test_flip_inverts_scores():
    src = [
        {"dimension": "dynamics", "score": 0.8, "deviation_from_mean": 0.25, "direction": "above_average"},
        {"dimension": "timing", "score": 0.3, "deviation_from_mean": -0.18, "direction": "below_average"},
    ]
    out = corrupt(src, mode="flip", seed=0, all_top_moments=[src])
    assert out[0]["score"] == pytest.approx(0.2)
    assert out[1]["score"] == pytest.approx(0.7)
    assert out[0]["direction"] == "below_average"
    assert out[1]["direction"] == "above_average"


def test_flip_deterministic():
    src = [{"dimension": "dynamics", "score": 0.8, "deviation_from_mean": 0.25, "direction": "above_average"}]
    a = corrupt(src, mode="flip", seed=0, all_top_moments=[src])
    b = corrupt(src, mode="flip", seed=999, all_top_moments=[src])
    assert a == b
