"""
Tests for the pedagogy relevance classifier.
"""

import pytest

from teacher_model.relevance_classifier import (
    ClassifierMetrics,
    PedagogyRelevanceClassifier,
)

# Module-level fixture so the model is only loaded once per test session
@pytest.fixture(scope="module")
def classifier() -> PedagogyRelevanceClassifier:
    return PedagogyRelevanceClassifier()


def test_classifier_scores_teaching_moment_high(
    classifier: PedagogyRelevanceClassifier,
) -> None:
    """Known teaching moments must score above 0.5."""
    teaching_moments = [
        (
            "Keep your wrist free and focus on the follow-through from the opening of "
            "the hand to the note, then pull back and recover."
        ),
        (
            "Make sure the left-hand notes stay really quiet, below the dynamic level "
            "of the melody. Practice the left hand alone, first in rhythm."
        ),
        (
            "Work on getting a more consistent voicing and shaping in the right hand "
            "— keep the phrasing clear and the drama steady throughout the piece."
        ),
        (
            "Use your torso, back and wrist, and play with the pads of your fingers "
            "close to the keys; staying close is a smart approach for pyrotechnical pieces."
        ),
    ]
    for text in teaching_moments:
        score = classifier.score(text)
        assert score > 0.5, (
            f"Expected teaching moment to score > 0.5 but got {score:.3f}: {text[:60]}"
        )


def test_classifier_scores_irrelevant_low(
    classifier: PedagogyRelevanceClassifier,
) -> None:
    """Gear talk and other off-topic content must score below 0.4."""
    irrelevant_texts = [
        (
            "I just picked up the Yamaha P-125 and honestly it is a huge upgrade. "
            "The weighted keys feel so much more realistic for the price point."
        ),
        (
            "The Spotify algorithm has completely changed how people discover "
            "classical music. Short clips drive millions of streams."
        ),
        (
            "A piano technician came to tune my Yamaha U3 and noticed the damper "
            "felts were worn on the lower octaves."
        ),
        (
            "Lang Lang gave an absolutely breathtaking performance of the Goldberg "
            "Variations last night."
        ),
    ]
    for text in irrelevant_texts:
        score = classifier.score(text)
        assert score < 0.4, (
            f"Expected irrelevant text to score < 0.4 but got {score:.3f}: {text[:60]}"
        )


def test_classifier_batch_scoring(
    classifier: PedagogyRelevanceClassifier,
) -> None:
    """Batch scoring returns correct count and all scores are in [0, 1]."""
    texts = [
        "Keep your wrist relaxed and let the arm weight do the work.",
        "I just bought a new Kawai digital piano for my apartment.",
        "Practice this passage hands separately at a slow tempo first.",
        "The concert last night was stunning from start to finish.",
        "Distribute the weight throughout the whole hand to avoid tension.",
    ]
    scores = classifier.score_batch(texts)

    assert len(scores) == len(texts), (
        f"Expected {len(texts)} scores but got {len(scores)}"
    )
    for i, score in enumerate(scores):
        assert 0.0 <= score <= 1.0, (
            f"Score at index {i} is out of range [0, 1]: {score}"
        )


def test_classifier_batch_scoring_empty_input(
    classifier: PedagogyRelevanceClassifier,
) -> None:
    """Empty batch returns empty list without error."""
    assert classifier.score_batch([]) == []


def test_classifier_validation_metrics(
    classifier: PedagogyRelevanceClassifier,
) -> None:
    """validate() returns ClassifierMetrics with valid ranges and correct counts."""
    metrics = classifier.validate()

    assert isinstance(metrics, ClassifierMetrics)
    assert 0.0 <= metrics.precision <= 1.0, f"Precision out of range: {metrics.precision}"
    assert 0.0 <= metrics.recall <= 1.0, f"Recall out of range: {metrics.recall}"
    assert 0.0 <= metrics.f1 <= 1.0, f"F1 out of range: {metrics.f1}"
    assert 0.0 <= metrics.threshold <= 1.0, f"Threshold out of range: {metrics.threshold}"
    assert metrics.n_positives == 379, f"Expected 379 positives, got {metrics.n_positives}"
    assert metrics.n_negatives >= 200, f"Expected >= 200 negatives, got {metrics.n_negatives}"

    # Classifier must be non-trivial: F1 should be meaningfully above 0
    assert metrics.f1 > 0.5, f"F1 too low ({metrics.f1:.3f}); classifier is not working"
