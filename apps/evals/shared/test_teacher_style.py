# apps/evals/shared/test_teacher_style.py
import pytest
from shared.teacher_style import evaluate

SIGNALS = {
    "max_neg_dev": 0.2, "max_pos_dev": 0.0, "n_significant": 2,
    "drilling_present": False, "drilling_improved": False,
    "duration_min": 15.0, "mode_count": 1, "has_piece": True,
}


def test_evaluate_arithmetic():
    assert evaluate("1.5 * max_neg_dev + 0.3 * n_significant", SIGNALS) == pytest.approx(0.9)


def test_evaluate_signal_lookup():
    assert evaluate("max_neg_dev", SIGNALS) == pytest.approx(0.2)


def test_evaluate_unknown_signal_raises():
    with pytest.raises(ValueError, match="unknown signal"):
        evaluate("max_neg_dev + bogus", SIGNALS)
