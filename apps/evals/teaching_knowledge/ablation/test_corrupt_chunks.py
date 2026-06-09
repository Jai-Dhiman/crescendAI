# apps/evals/teaching_knowledge/ablation/test_corrupt_chunks.py
import pytest

from teaching_knowledge.ablation.corrupt_signals import corrupt_chunks

DIMS = ["dynamics", "timing", "pedaling"]


def _chunks(score: float, n: int = 2) -> list[dict]:
    return [
        {
            "chunk_index": i,
            "predictions": {d: score for d in DIMS},
            "midi_notes": [{"pitch": 60}],
            "pedal_events": [],
        }
        for i in range(n)
    ]


def test_flip_inverts_every_prediction() -> None:
    src = _chunks(0.7)
    out = corrupt_chunks(src, mode="flip", seed=42, all_chunks=[src])
    assert all(v == 0.3 for ch in out for v in ch["predictions"].values())
    # midi_notes / pedal_events preserved (only the MuQ signal is perturbed).
    assert out[0]["midi_notes"] == [{"pitch": 60}]
    assert out[0]["chunk_index"] == 0


def test_flip_does_not_mutate_source() -> None:
    src = _chunks(0.7)
    corrupt_chunks(src, mode="flip", seed=42, all_chunks=[src])
    assert src[0]["predictions"]["dynamics"] == 0.7


def test_shuffle_borrows_another_recordings_chunks() -> None:
    a = _chunks(0.7)
    b = _chunks(0.2)
    out = corrupt_chunks(a, mode="shuffle", seed=42, all_chunks=[a, b])
    # Deterministic and drawn from a different recording (only b qualifies).
    assert out is b
    assert corrupt_chunks(a, mode="shuffle", seed=42, all_chunks=[a, b]) is b


def test_shuffle_raises_on_singleton_corpus() -> None:
    a = _chunks(0.7)
    with pytest.raises(ValueError, match="no other recordings"):
        corrupt_chunks(a, mode="shuffle", seed=42, all_chunks=[a])


def test_marginal_draws_scores_from_corpus_pool() -> None:
    a = _chunks(0.7)
    b = _chunks(0.2)
    out = corrupt_chunks(a, mode="marginal", seed=42, all_chunks=[a, b])
    pool = {0.7, 0.2}
    assert all(v in pool for ch in out for v in ch["predictions"].values())
    # Same dims preserved.
    assert set(out[0]["predictions"].keys()) == set(DIMS)


def test_marginal_deterministic_same_seed() -> None:
    a = _chunks(0.7)
    b = _chunks(0.2)
    x = corrupt_chunks(a, mode="marginal", seed=7, all_chunks=[a, b])
    y = corrupt_chunks(a, mode="marginal", seed=7, all_chunks=[a, b])
    assert x == y


def test_unknown_mode_raises() -> None:
    a = _chunks(0.7)
    with pytest.raises(ValueError, match="unknown mode"):
        corrupt_chunks(a, mode="nonsense", seed=1, all_chunks=[a])
