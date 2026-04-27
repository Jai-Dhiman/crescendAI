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


def test_evaluate_conditional_true_branch():
    sig = {**SIGNALS, "drilling_improved": True}
    assert evaluate("1.5 if drilling_improved else 0.0", sig) == pytest.approx(1.5)


def test_evaluate_conditional_false_branch():
    sig = {**SIGNALS, "drilling_improved": False}
    assert evaluate("1.5 if drilling_improved else 0.5", sig) == pytest.approx(0.5)


def test_evaluate_compound_with_and():
    sig = {**SIGNALS, "max_neg_dev": 0.05, "max_pos_dev": 0.05}
    assert evaluate("1 if max_neg_dev < 0.1 and max_pos_dev < 0.1 else 0", sig) == pytest.approx(1.0)


def test_evaluate_real_formula_technical_corrective():
    sig = {**SIGNALS, "max_neg_dev": 0.2, "n_significant": 2, "drilling_improved": False}
    formula = "1.5 * max_neg_dev + 0.3 * n_significant - 0.5 * (1 if drilling_improved else 0)"
    assert evaluate(formula, sig) == pytest.approx(0.9)


from shared.teacher_style import select_clusters


def test_select_clusters_negative_dev_picks_technical():
    sig = {"max_neg_dev": 0.25, "max_pos_dev": 0.0, "n_significant": 3,
           "drilling_present": False, "drilling_improved": False,
           "duration_min": 15.0, "mode_count": 1, "has_piece": True}
    sel = select_clusters(sig)
    assert "Technical" in sel.primary.name


def test_select_clusters_positive_dev_picks_praise():
    sig = {"max_neg_dev": 0.0, "max_pos_dev": 0.3, "n_significant": 1,
           "drilling_present": True, "drilling_improved": True,
           "duration_min": 25.0, "mode_count": 2, "has_piece": True}
    sel = select_clusters(sig)
    assert "Positive" in sel.primary.name or "praise" in sel.primary.name.lower()


def test_select_clusters_returns_two_distinct():
    sig = {"max_neg_dev": 0.25, "max_pos_dev": 0.0, "n_significant": 3,
           "drilling_present": False, "drilling_improved": False,
           "duration_min": 15.0, "mode_count": 1, "has_piece": True}
    sel = select_clusters(sig)
    assert sel.primary.name != sel.secondary.name


def test_select_clusters_fallback_when_all_low():
    # max_neg_dev=0.1 keeps all cluster scores < 0.3 (CONFIDENCE_FLOOR):
    # Technical=0.15, Artifact=0 (mnd<0.15, dm>=20, no piece), Motivational=0 (dm>=10, mnd not<0.1), etc.
    sig = {"max_neg_dev": 0.1, "max_pos_dev": 0.0, "n_significant": 0,
           "drilling_present": False, "drilling_improved": False,
           "duration_min": 20.0, "mode_count": 1, "has_piece": False}
    sel = select_clusters(sig)
    assert "Technical" in sel.primary.name
    assert "Positive" in sel.secondary.name or "praise" in sel.secondary.name.lower()


from shared.teacher_style import format_teacher_voice_blocks


def test_format_emits_both_blocks():
    sig = {"max_neg_dev": 0.25, "max_pos_dev": 0.0, "n_significant": 3,
           "drilling_present": False, "drilling_improved": False,
           "duration_min": 15.0, "mode_count": 1, "has_piece": True}
    out = format_teacher_voice_blocks(select_clusters(sig))
    assert "<teacher_voice" in out
    assert "<also_consider" in out
    assert "Register:" in out
    assert "Tone:" in out
    assert "Exemplar:" in out


def test_format_includes_cluster_attribute():
    sig = {"max_neg_dev": 0.25, "max_pos_dev": 0.0, "n_significant": 3,
           "drilling_present": False, "drilling_improved": False,
           "duration_min": 15.0, "mode_count": 1, "has_piece": True}
    out = format_teacher_voice_blocks(select_clusters(sig))
    assert 'cluster="' in out
