"""Verify pre-registered KILL/TUNE/PROCEED rule.

Gate thresholds (updated for note-based bakeoff):
  KILL    if dtw_recall10 < 0.70
  PROCEED if best_indexable_recall10 >= 0.85 AND open_set_ok (FA<=0.05 @ TA>=0.60)
  TUNE    otherwise
"""
from piece_id_eval.decision import decide


def test_kill_when_dtw_below_threshold() -> None:
    assert decide(dtw_recall10=0.60, best_indexable_recall10=0.95, open_set_ok_flag=True) == "KILL"


def test_kill_at_exact_boundary() -> None:
    assert decide(dtw_recall10=0.699, best_indexable_recall10=0.90, open_set_ok_flag=True) == "KILL"


def test_proceed_when_all_criteria_met() -> None:
    assert decide(dtw_recall10=0.80, best_indexable_recall10=0.85, open_set_ok_flag=True) == "PROCEED"


def test_tune_when_dtw_ok_but_indexable_low() -> None:
    assert decide(dtw_recall10=0.75, best_indexable_recall10=0.80, open_set_ok_flag=True) == "TUNE"


def test_tune_when_dtw_ok_indexable_ok_but_open_set_fails() -> None:
    assert decide(dtw_recall10=0.80, best_indexable_recall10=0.90, open_set_ok_flag=False) == "TUNE"


def test_proceed_at_exact_indexable_boundary() -> None:
    assert decide(dtw_recall10=0.70, best_indexable_recall10=0.85, open_set_ok_flag=True) == "PROCEED"


def test_tune_when_dtw_exactly_at_boundary() -> None:
    assert decide(dtw_recall10=0.70, best_indexable_recall10=0.60, open_set_ok_flag=True) == "TUNE"


def test_docstring_reflects_new_open_set_thresholds() -> None:
    """The module docstring must mention the current FA/TA thresholds."""
    import piece_id_eval.decision as _mod
    doc = _mod.__doc__ or ""
    assert "0.05" in doc, "docstring missing FA threshold 0.05"
    assert "0.60" in doc, "docstring missing TA threshold 0.60"
